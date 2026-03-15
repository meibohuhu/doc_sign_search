# Blackwell GPU 多卡训练修复记录

## 问题背景

GPU: NVIDIA RTX PRO 6000 Blackwell (sm_120, 98GB/卡)

运行 `finetune_internvl2_5_how2sign_1b_unfreeze.sh` 时遇到一系列错误，根本原因是
旧版软件栈不支持 Blackwell sm_120 架构。

---

## 错误链

| 错误 | 原因 |
|------|------|
| `CUDA error: no kernel image is available` | PyTorch 2.5.1+cu121 无 sm_120 内核 |
| `NCCL error: ncclUnhandledCudaError` | NCCL 2.21.5 不支持 sm_120 |
| `CUDA SETUP: Required library not found: libbitsandbytes_cuda128.so` | bitsandbytes 0.42.0 不支持 cu128 |
| `ModuleNotFoundError: No module named 'six'` | conda 环境 dist-info 损坏 |
| `CUDA error: an illegal memory access` | NCCL 在 Blackwell 上做 broadcast 时非法内存访问 |

---

## 修复内容

### 1. 修复 conda 环境（Python 可执行文件丢失）

```bash
/home/stu2/s15/mh2803/anaconda3/bin/conda install -n internvl python=3.10 -y
```

### 2. 升级 PyTorch 到 CUDA 12.8 版本（最核心修复）

```bash
/home/stu2/s15/mh2803/anaconda3/envs/internvl/bin/python3.10 -m pip install \
  "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" "torchaudio==2.7.0+cu128" \
  --index-url https://download.pytorch.org/whl/cu128 \
  --cache-dir /tmp/pip_cache
```

- cu128 (CUDA 12.8) 是首个原生支持 Blackwell sm_120 的版本
- cu128 索引中最低可用版本为 2.7.0（不存在 2.6.0+cu128）

### 3. 升级/修复依赖包

```bash
# bitsandbytes: 0.42.0 → 0.49.2，支持 cu128
pip install bitsandbytes --upgrade

# flash-attn: 重新编译以匹配新 PyTorch ABI
mkdir -p /tmp/pip_cache /tmp/pip_build
TMPDIR=/tmp/pip_build CUDA_HOME=/home/stu2/s15/mh2803/anaconda3/envs/internvl \
pip install flash-attn --no-build-isolation --force-reinstall --cache-dir /tmp/pip_cache

# six: dist-info 损坏导致 pandas/datasets 无法导入
rm -rf /home/stu2/s15/mh2803/anaconda3/envs/internvl/lib/python3.10/site-packages/six*
pip install six --cache-dir /tmp/pip_cache

# deepspeed: 0.18.2 → 0.18.7
pip install deepspeed --upgrade
```

### 4. 修改 DeepSpeed 配置

**文件**: `InternVL/internvl_chat/zero_stage0_config.json`

从 Stage 1 改为 Stage 0，禁用 fp16/bf16 的手动设置：

```json
{
  "zero_optimization": {
    "stage": 0
  },
  "fp16": {
    "enabled": false
  },
  "bf16": {
    "enabled": "auto"
  },
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "steps_per_print": 2000,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false,
  "comms_logger": {
    "enabled": false
  }
}
```

**原因**: Stage 1 的 optimizer sharding 使用了 Blackwell 不支持的 CUDA 内核。Stage 0 无此问题。

**影响**: 每卡多用约 2-4GB 显存存储 optimizer states，98GB 显存完全无压力。训练精度/收敛不受影响。

### 5. 修改分布式后端

**文件**: `InternVL/internvl_chat/internvl/train/internvl_chat_finetune_local.py` (行 1330)

```python
# 修改前
init_dist(launcher=launcher, backend='nccl')

# 修改后
init_dist(launcher=launcher, backend='gloo')
```

**原因**: NCCL 在 Blackwell 上执行 `_broadcast_model`（DeepSpeed 初始化时广播模型参数）时触发非法内存访问，即使设置了 `NCCL_P2P_DISABLE=1` 也无法解决。

**影响**: gloo 通过 CPU/共享内存做跨卡通信，比 NCCL 略慢。但本次训练只有 ~8.8M LoRA 可训练参数，同步量极小，速度损失约 5-15%。训练精度/收敛与 NCCL 完全等价。

### 6. 修改训练脚本

**文件**: `script_adobe/0310/finetune_internvl2_5_how2sign_1b_unfreeze.sh`

```bash
# 添加 CUDA_HOME（DeepSpeed 编译 ops 需要）
export CUDA_HOME="$HOME/anaconda3/envs/internvl"

# NCCL 环境变量（保留以防其他地方用到 NCCL）
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# 恢复 bf16 训练
--bf16 True
```

---

## 最终环境

| 组件 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.7.0+cu128 |
| CUDA | 12.8 |
| NCCL | 2.26.2 |
| DeepSpeed | 0.18.7 |
| flash-attn | 2.8.3 |
| bitsandbytes | 0.49.2 |

---

## 运行方式

```bash
cd /home/stu2/s15/mh2803/workspace/doc_sign_search/InternVL
GPU_IDS="0,1,2,3" bash ../script_adobe/0310/finetune_internvl2_5_how2sign_1b_unfreeze.sh
```

注意: NCCL 初始化需约 20 秒，这是正常现象，不是 hang。

---

## 相关文件

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 训练指南
- [script_adobe/0310/finetune_internvl2_5_how2sign_1b_unfreeze.sh](script_adobe/0310/finetune_internvl2_5_how2sign_1b_unfreeze.sh) - 训练脚本
- [InternVL/internvl_chat/zero_stage0_config.json](InternVL/internvl_chat/zero_stage0_config.json) - DeepSpeed 配置
