# coding=utf-8
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Qwen2 model with Gated Attention."""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
# Import cache classes with fallback for older transformers versions
from transformers.cache_utils import Cache, DynamicCache
try:
    from transformers.cache_utils import StaticCache
except ImportError:
    StaticCache = None
try:
    from transformers.cache_utils import SlidingWindowCache
except ImportError:
    SlidingWindowCache = None
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
# Import ROPE_INIT_FUNCTIONS with fallback for older transformers versions
try:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
except ImportError:
    # Fallback: try to import from a different location or create a minimal version
    try:
        from transformers.models.llama.modeling_llama import ROPE_INIT_FUNCTIONS
    except ImportError:
        # Create a minimal fallback with default RoPE initialization
        def _default_rope_init(config, device=None, seq_len=None, **kwargs):
            """Default RoPE initialization function"""
            max_position_embeddings = config.max_position_embeddings if hasattr(config, 'max_position_embeddings') else 2048
            base = getattr(config, 'rope_theta', 10000.0)
            dim = config.hidden_size // config.num_attention_heads if hasattr(config, 'hidden_size') else 128
            
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
            return inv_freq, 1.0
        
        ROPE_INIT_FUNCTIONS = {"default": _default_rope_init}
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

# Import Qwen2Config from transformers
try:
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
except ImportError:
    try:
        # Fallback: try to get from AutoConfig
        from transformers import AutoConfig
        # Create a dummy config class for type hints
        class Qwen2Config:
            pass
    except ImportError:
        # Final fallback: create a minimal config class
        class Qwen2Config:
            pass

logger = logging.get_logger(__name__)

# Import flash_attn functions if available
if is_flash_attn_2_available():
    try:
        from flash_attn import flash_attn_func, flash_attn_varlen_func
        from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
        try:
            from flash_attn.bert_padding import _get_unpad_data
        except ImportError:
            # Fallback: try to import from a different location
            try:
                from flash_attn import _get_unpad_data
            except ImportError:
                # If all imports fail, define a minimal version
                def _get_unpad_data(attention_mask):
                    """Minimal fallback for _get_unpad_data"""
                    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
                    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
                    max_seqlen_in_batch = seqlens_in_batch.max().item()
                    cu_seqlens = torch.zeros((seqlens_in_batch.shape[0] + 1,), device=seqlens_in_batch.device, dtype=torch.int32)
                    cu_seqlens[1:] = torch.cumsum(seqlens_in_batch, dim=0)
                    return indices, cu_seqlens, max_seqlen_in_batch
    except ImportError:
        flash_attn_func = None
        flash_attn_varlen_func = None
        index_first_axis = None
        pad_input = None
        unpad_input = None
        _get_unpad_data = None
        logger.warning_once("Could not import flash_attn. Flash attention may not work correctly.")
else:
    flash_attn_func = None
    flash_attn_varlen_func = None
    index_first_axis = None
    pad_input = None
    unpad_input = None
    _get_unpad_data = None

_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-0.5B"
_CONFIG_FOR_DOC = "Qwen2Config"


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Qwen2
class Qwen2RotaryEmbedding(nn.Module):
    def __init__(
            self,
            dim=None,
            max_position_embeddings=2048,
            base=10000,
            device=None,
            scaling_factor=1.0,
            rope_type="default",
            config: Optional[Qwen2Config] = None,
    ):
        super().__init__()
        # TODO (joao): remove the `if` below, only used for BC
        self.rope_kwargs = {}
        if config is None:
            logger.warning_once(
                "`Qwen2RotaryEmbedding` can now be fully parameterized by passing the model config through the "
                "`config` argument. All other arguments will be removed in v4.46"
            )
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
            self.max_seq_len_cached = max_position_embeddings
            self.original_max_seq_len = max_position_embeddings
        else:
            # BC: "rope_type" was originally "type"
            # Use getattr to safely access rope_scaling (may not exist in older configs)
            rope_scaling = getattr(config, "rope_scaling", None)
            if rope_scaling is not None:
                self.rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = getattr(config, "max_position_embeddings", max_position_embeddings)
            self.original_max_seq_len = self.max_seq_len_cached

        self.config = config
        # Use default if rope_type is not available in ROPE_INIT_FUNCTIONS
        if self.rope_type not in ROPE_INIT_FUNCTIONS:
            logger.warning_once(f"rope_type '{self.rope_type}' not found in ROPE_INIT_FUNCTIONS, using 'default'")
            self.rope_type = "default"
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(
                self.config, device, seq_len=seq_len, **self.rope_kwargs
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper with Gated Attention support.
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = getattr(config, "attention_dropout", 0.0)
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        
        # Gate attention flags
        self.headwise_attn_output_gate = getattr(config, "headwise_attn_output_gate", False)
        self.elementwise_attn_output_gate = getattr(config, "elementwise_attn_output_gate", False)
        
        # qkv_bias support
        # Note: In original Qwen2, q_proj/k_proj/v_proj have bias=True, o_proj has bias=False
        # qkv_bias flag is used to enable/disable bias for o_proj, but q/k/v always have bias
        qkv_bias = getattr(config, "qkv_bias", False)

        # q_proj with gate support (same as Qwen3)
        # q_proj/k_proj/v_proj always have bias=True to match original Qwen2 behavior
        if self.headwise_attn_output_gate:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim + self.num_heads, bias=True)
        elif self.elementwise_attn_output_gate:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim * 2, bias=True)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)

        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        # o_proj uses qkv_bias flag (False by default, matching original Qwen2)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=qkv_bias)
        
        if self.use_qk_norm:
            self.q_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            self.k_norm = Qwen2RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        
        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Extract gate_score from q_proj output (same as Qwen3)
        # Note: For Qwen2, we need to handle the case where q_proj output includes gate dimensions
        if self.headwise_attn_output_gate:
            # q_proj output: [bsz, q_len, num_heads * head_dim + num_heads]
            # Split into query and gate parts
            query_dim = self.num_heads * self.head_dim
            query_states, gate_score_raw = torch.split(query_states, [query_dim, self.num_heads], dim=-1)
            # Reshape query_states: [bsz, q_len, num_heads, head_dim]
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            # Reshape gate_score: [bsz, q_len, num_heads, 1]
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)
        elif self.elementwise_attn_output_gate:
            # q_proj output: [bsz, q_len, num_heads * head_dim * 2]
            # Split into query and gate parts
            query_dim = self.num_heads * self.head_dim
            query_states, gate_score_raw = torch.split(query_states, [query_dim, query_dim], dim=-1)
            # Reshape query_states: [bsz, q_len, num_heads, head_dim]
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            # Reshape gate_score: [bsz, q_len, num_heads, head_dim]
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)
        else:
            # Standard mode: [bsz, q_len, num_heads, head_dim]
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            gate_score = None
        
        # Reshape key and value states (same as transformers Qwen2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Handle position_embeddings (same as transformers Qwen2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # Apply gating (same as Qwen3)
        # attn_output: [bsz, q_len, num_heads, head_dim]
        # gate_score: [bsz, q_len, num_heads, 1] or [bsz, q_len, num_heads, head_dim]
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            # Print attention_score before gate
            # if self.layer_idx is not None and self.layer_idx < 2:  # Only print for first 2 layers to avoid spam
            #     print(f"[Qwen2Attention Layer {self.layer_idx}] BEFORE GATE - attn_weights stats:")
            #     print(f"  shape: {attn_weights.shape}")
            #     print(f"  mean: {attn_weights.mean().item():.6f}, std: {attn_weights.std().item():.6f}")
            #     print(f"  min: {attn_weights.min().item():.6f}, max: {attn_weights.max().item():.6f}")
            #     print(f"  attn_output stats - mean: {attn_output.mean().item():.6f}, std: {attn_output.std().item():.6f}")
            
            attn_output = attn_output * torch.sigmoid(gate_score)
            
            # # Print attention_score after gate
            # if self.layer_idx is not None and self.layer_idx < 2:  # Only print for first 2 layers to avoid spam
            print(f"[Qwen2Attention Layer {self.layer_idx}] AFTER GATE - attn_output stats:")
            print(f"  mean: {attn_output.mean().item():.6f}, std: {attn_output.std().item():.6f}")
            print(f"  min: {attn_output.min().item():.6f}, max: {attn_output.max().item():.6f}")
            if gate_score is not None:
                sigmoid_gate = torch.sigmoid(gate_score)
                print(f"  gate_score (sigmoid) - mean: {sigmoid_gate.mean().item():.6f}, std: {sigmoid_gate.std().item():.6f}")
                print(f"  gate_score (sigmoid) - min: {sigmoid_gate.min().item():.6f}, max: {sigmoid_gate.max().item():.6f}")

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2FlashAttention2(Qwen2Attention):
    """
    Qwen2 flash attention module with Gated Attention support. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Extract gate_score from q_proj output (same as Qwen3, but match Qwen2 structure)
        if self.headwise_attn_output_gate:
            # q_proj output: [bsz, q_len, num_heads * head_dim + num_heads]
            query_dim = self.num_heads * self.head_dim
            query_states, gate_score_raw = torch.split(query_states, [query_dim, self.num_heads], dim=-1)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)
        elif self.elementwise_attn_output_gate:
            # q_proj output: [bsz, q_len, num_heads * head_dim * 2]
            query_dim = self.num_heads * self.head_dim
            query_states, gate_score_raw = torch.split(query_states, [query_dim, query_dim], dim=-1)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            gate_score = None
            
        # Reshape key and value states (same as transformers Qwen2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Handle position_embeddings (same as transformers Qwen2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # Reshape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)
        
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # # Calculate attention scores before Flash Attention (for debugging)
        # # Note: Flash Attention doesn't return attention weights, so we compute them manually
        # # After transpose(1, 2):
        # # query_states: [bsz, q_len, num_heads, head_dim]
        # # key_states: [bsz, q_len, num_key_value_heads, head_dim]
        # if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
        #     # Transpose back to [bsz, num_heads, q_len, head_dim] for attention score calculation
        #     query_states_for_attn = query_states.transpose(1, 2)  # [bsz, num_heads, q_len, head_dim]
        #     # Repeat key_states to match num_heads
        #     key_states_for_attn = repeat_kv(key_states.transpose(1, 2), self.num_key_value_groups)  # [bsz, num_heads, q_len, head_dim]
        #     # Compute attention scores: Q @ K^T / sqrt(head_dim)
        #     # Result: [bsz, num_heads, q_len, q_len]
        #     attn_scores = torch.matmul(query_states_for_attn, key_states_for_attn.transpose(2, 3)) / math.sqrt(self.head_dim)
        #     # Use both logger and print to ensure output
        #     msg_attn_before = f"[Qwen2FlashAttention2 Layer {self.layer_idx}] BEFORE GATE - attention_score stats:"
        #     logger.info(msg_attn_before)
        #     print(msg_attn_before, flush=True)
            
        #     msg_attn_shape = f"  attention_score shape: {attn_scores.shape}"
        #     logger.info(msg_attn_shape)
        #     print(msg_attn_shape, flush=True)
            
        #     msg_attn_stats = f"  attention_score mean: {attn_scores.mean().item():.6f}, std: {attn_scores.std().item():.6f}"
        #     logger.info(msg_attn_stats)
        #     print(msg_attn_stats, flush=True)
            
        #     msg_attn_minmax = f"  attention_score min: {attn_scores.min().item():.6f}, max: {attn_scores.max().item():.6f}"
        #     logger.info(msg_attn_minmax)
        #     print(msg_attn_minmax, flush=True)
            
        #     # Also print softmax of attention scores
        #     attn_scores_softmax = nn.functional.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
        #     msg_attn_softmax_stats = f"  attention_score (softmax) mean: {attn_scores_softmax.mean().item():.6f}, std: {attn_scores_softmax.std().item():.6f}"
        #     logger.info(msg_attn_softmax_stats)
        #     print(msg_attn_softmax_stats, flush=True)
            
        #     msg_attn_softmax_minmax = f"  attention_score (softmax) min: {attn_scores_softmax.min().item():.6f}, max: {attn_scores_softmax.max().item():.6f}"
        #     logger.info(msg_attn_softmax_minmax)
        #     print(msg_attn_softmax_minmax, flush=True)

        # Use self._flash_attention_forward method (defined in this class)
        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=False,  # Qwen2 doesn't use sliding window
        )

        # Apply gating (same as Qwen3)
        # attn_output from flash attention: [bsz, q_len, num_heads, head_dim]
        # gate_score: [bsz, q_len, num_heads, 1] or [bsz, q_len, num_heads, head_dim]
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            attn_output = attn_output * torch.sigmoid(gate_score)
            
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        """Unpad input for flash attention (copied from transformers Qwen2FlashAttention2)"""
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape
        q_batch_size, q_seq_len, q_num_heads, q_head_dim = query_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            # Verify query_layer shape matches expected shape
            if q_seq_len == kv_seq_len and q_batch_size == batch_size and q_num_heads == num_heads and q_head_dim == head_dim:
                query_layer = index_first_axis(
                    query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
                )
                cu_seqlens_q = cu_seqlens_k
                max_seqlen_in_batch_q = max_seqlen_in_batch_k
                indices_q = indices_k
            else:
                # Shape mismatch: use unpad_input instead
                # Use the same attention_mask since query_length == kv_seq_len
                query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)
                return (
                    query_layer,
                    key_layer,
                    value_layer,
                    indices_q,
                    (cu_seqlens_q, cu_seqlens_k),
                    (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
                )
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.
        (Copied from transformers Qwen2FlashAttention2)
        """
        if flash_attn_func is None or flash_attn_varlen_func is None:
            raise RuntimeError(
                "Flash attention is required but not available. "
                "Please install flash-attn: pip install flash-attn"
            )
        
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
            causal = self.is_causal and query_length != 1

        # Decide whether to use SWA or not by layer index.
        if use_sliding_windows and self.layer_idx >= self.config.max_window_layers:
            use_sliding_windows = False

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, self.config.sliding_window),
                )
        
        return attn_output


class Qwen2SdpaAttention(Qwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention with Gated Attention support.
    This module inherits from `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # Extract gate_score from q_proj output (same as Qwen3, but match Qwen2 structure)
        if self.headwise_attn_output_gate:
            # q_proj output: [bsz, q_len, num_heads * head_dim + num_heads]
            query_dim = self.num_heads * self.head_dim
            query_states, gate_score_raw = torch.split(query_states, [query_dim, self.num_heads], dim=-1)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, 1)
        elif self.elementwise_attn_output_gate:
            # q_proj output: [bsz, q_len, num_heads * head_dim * 2]
            query_dim = self.num_heads * self.head_dim
            query_states, gate_score_raw = torch.split(query_states, [query_dim, query_dim], dim=-1)
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            gate_score = gate_score_raw.view(bsz, q_len, self.num_heads, self.head_dim)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            gate_score = None
            
        # Reshape key and value states (same as transformers Qwen2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # Handle position_embeddings (same as transformers Qwen2)
        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # key_states: bs, head, q_len, head_dim
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        # Apply gating (same as Qwen3)
        # attn_output: [bsz, q_len, num_heads, head_dim]
        # gate_score: [bsz, q_len, num_heads, 1] or [bsz, q_len, num_heads, head_dim]
        if self.headwise_attn_output_gate or self.elementwise_attn_output_gate:
            # Print attention_score before gate
            if self.layer_idx is not None and self.layer_idx < 2:  # Only print for first 2 layers to avoid spam
                print(f"[Qwen2SdpaAttention Layer {self.layer_idx}] BEFORE GATE - attn_output stats:")
                print(f"  shape: {attn_output.shape}")
                print(f"  mean: {attn_output.mean().item():.6f}, std: {attn_output.std().item():.6f}")
                print(f"  min: {attn_output.min().item():.6f}, max: {attn_output.max().item():.6f}")
            
            attn_output = attn_output * torch.sigmoid(gate_score)
            
            # Print attention_score after gate
            if self.layer_idx is not None and self.layer_idx < 2:  # Only print for first 2 layers to avoid spam
                print(f"[Qwen2SdpaAttention Layer {self.layer_idx}] AFTER GATE - attn_output stats:")
                print(f"  mean: {attn_output.mean().item():.6f}, std: {attn_output.std().item():.6f}")
                print(f"  min: {attn_output.min().item():.6f}, max: {attn_output.max().item():.6f}")
                if gate_score is not None:
                    sigmoid_gate = torch.sigmoid(gate_score)
                    print(f"  gate_score (sigmoid) - mean: {sigmoid_gate.mean().item():.6f}, std: {sigmoid_gate.std().item():.6f}")
                    print(f"  gate_score (sigmoid) - min: {sigmoid_gate.min().item():.6f}, max: {sigmoid_gate.max().item():.6f}")

        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value

QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "flash_attention_2": Qwen2FlashAttention2,
    "sdpa": Qwen2SdpaAttention,
}


# Note: For a complete implementation, we would need to add Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM, etc.
# However, since InternVL uses transformers' Qwen2Model directly and only patches the attention class,
# we can create a minimal implementation that focuses on the attention classes.
# The full model classes can be imported from transformers and patched with our gate attention classes.

# Export the attention classes for patching
__all__ = [
    "Qwen2Attention",
    "Qwen2FlashAttention2", 
    "Qwen2SdpaAttention",
    "QWEN2_ATTENTION_CLASSES",
]

