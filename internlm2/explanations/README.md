num_attention_heads = 16

LLM 生成答案不是基于 aggregated attention，而是逐层处理：
Layer 0:
  输入 hidden_states[0] 
  → 计算 attention weights
  → 用 weights 更新 hidden_states[0] 
  → 输出 hidden_states[0] (更新后)

Layer 1:
  输入 hidden_states[1] = hidden_states[0] (从 Layer 0 来的)
  → 计算 attention weights
  → 用 weights 更新 hidden_states[1]
  → 输出 hidden_states[1] (更新后)

...

Layer 23:
  输入 hidden_states[23] = hidden_states[22] (从 Layer 22 来的)
  → 计算 attention weights
  → 用 weights 更新 hidden_states[23]
  → 输出 hidden_states[23] (最终结果，用于生成)





Hidden States = 每个 token 的向量表示
形状：[batch_size, seq_len, hidden_dim]
例如：[1, 445, 2048] 表示 1 个样本、445 个 token，每个 token 用 2048 维向量表示、

每一层（decoder_layer）会：
    计算 Attention：基于当前 hidden_states 计算 attention weights
    应用 Attention：用 attention weights 加权聚合信息，更新 hidden_states
    Feed-Forward：通过 MLP 进一步处理 hidden_states
    输出：返回更新后的 hidden_states 给下一层




总结
    单个 layer 的 attention 高度集中在少数 patch 上是正常的
    不同层关注不同区域，形成互补
    Aggregated attention 综合所有层的信息，显示整体关注模式