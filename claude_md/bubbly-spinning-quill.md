# SFT Training Strategy: Multi-Dataset Mixed Training

## Context

**目标：** 提高 How2Sign benchmark 分数（当前最好：BLEU-1 33.37%, BLEU-4 7.64%, ROUGE-L 27.96%）

**历史最高（eval文件）：** BLEU-1 34.6%, BLEU-4 8.1%, ROUGE-L 29.1%（`04_04_02` reward权重模型）

**数据情况：**
- How2Sign (H2S): 32,622 samples — 精确翻译，目标domain
- OpenASL: 135,990 samples — 精确翻译，不同domain（ASL但场景/signer不同）
- YouTube: 173,263 samples — weakly supervised，已filter，比较noisy

**框架：** InternVL2.5-1B，meta_path JSON支持多个dataset key + repeat_time（float subsample / int repeat）

**重要约束：** 最终一定要做RL，SFT checkpoint是RL的起点。

---

## 实验计划：Ablation Study

### 目标
通过控制变量实验，验证每个dataset对H2S benchmark的贡献，找到最优数据混合配置。

### 实验设计（每次只变一个变量）

| 实验 | H2S | OpenASL | YouTube | Total | 目的 |
|------|-----|---------|---------|-------|------|
| **A (baseline)** | 32k (×1) | 0 | 0 | 32k | 对照组，纯H2S |
| **B** | 32k (×1) | 68k (×0.5) | 0 | 100k | OpenASL的价值 |
| **C** | 32k (×1) | 0 | 43k (×0.25) | 75k | YouTube的价值 |
| **D** | 32k (×1) | 68k (×0.5) | 43k (×0.25) | 143k | 三者混合 |

**分析方法：**
- B vs A = OpenASL 的增益
- C vs A = YouTube 的增益
- D vs A = 混合的总增益
- D vs B/C = 三者混合 vs 只加一个

### 训练超参（所有实验统一）
- `num_train_epochs: 6`
- `learning_rate: 4e-5`
- `save_strategy: epoch`，`save_total_limit: 6`
- Freeze: `freeze_llm=True, freeze_backbone=False, freeze_mlp=True, use_llm_lora=16`
- 每个epoch存checkpoint，eval选最优

### 执行优先级

1. **先跑 A + D**（并行）— A是baseline对照，D是最终方案的快速验证
2. **再跑 B + C**（并行）— 验证OpenASL和YouTube各自的独立贡献
3. 根据结果决定是否需要更大数据量的实验

---

## Stage 2：H2S Domain Fine-tuning（可选）

**是否需要Stage 2取决于Stage 1的结果：**

- 如果Stage 1最优checkpoint的H2S分数 > baseline → 可以直接用给RL，或跑短Stage 2看是否还能涨
- 如果Stage 1分数 ≈ baseline → 跑Stage 2做domain alignment
- 如果Stage 1分数 < baseline → 混合数据有害，回到纯H2S

**验证Stage 2价值的方法：**
- 用Stage 1的中间checkpoint（epoch 2-3）做Stage 2
- 如果Stage 2有提升 → 两阶段策略有效
- 如果Stage 2没提升 → 直接用Stage 1最优checkpoint给RL

**Stage 2训练超参：**
- `num_train_epochs: 5`（H2S在epoch 5左右最优，loss降到1-1.5是sweet spot）
- `learning_rate: 2e-5`（比Stage 1小一半，精细调整避免灾难性遗忘）
- 从Stage 1 checkpoint继续，`--model_name_or_path` 指向Stage 1 output dir
- `save_strategy: epoch`，重点关注epoch 3/4/5的eval分数
- 选eval分数最好的checkpoint → 给RL

**Stage 2数据配置（`train_stage2_meta.json`）：**
```json
{
  "how2sign": {
    "root": "/scratch/mh2803/train_crop_videos_224/",
    "annotation": "InternVL/data/how2sign/segmented_train_val_combined.jsonl",
    "data_augment": false,
    "repeat_time": 1,
    "length": 32622
  }
}
```

---

## 后续扩展实验

找到最优配置后，可以尝试scale up：

| 方案 | H2S | OpenASL | YouTube | Total |
|------|-----|---------|---------|-------|
| Scale 1 | 65k (×2) | 68k (×0.5) | 43k (×0.25) | 176k |
| Scale 2 | 65k (×2) | 68k (×0.5) | 87k (×0.5) | 220k |
| Scale 3 | 130k (×4) | 136k (×1) | 87k (×0.5) | 353k |
| Full | 130k (×4) | 136k (×1) | 173k (×1) | 439k |

---

## 已创建的文件

1. **`script_adobe/train_stage1_meta.json`** — Stage 1数据配置（当前正在跑的实验）
2. **`script_adobe/train_stage2_meta.json`** — Stage 2数据配置（H2S only）
3. **`script_adobe/0317/finetune_stage1_broad.sh`** — Stage 1训练脚本
4. **`script_adobe/0317/finetune_stage2_h2s.sh`** — Stage 2训练脚本

---

## 关键决策说明

**为什么不用loss weighting？**
直接用repeat_time控制sampling比例，效果等价，且不需要改training code。

**关于repeat_time：**
- `repeat_time > 1`（int）：在Dataset初始化时复制数据列表，每个epoch都遍历全部
- `repeat_time < 1`（float）：截取前N条数据
- 注意：原代码 `assert isinstance(repeat_time, int)` 会在JSON解析为float时报错，已修复为 `int(repeat_time)`

**关于use_data_resampling：**
- `--use_data_resampling True` 启用WeightedConcatDataset（按sqrt(len)做weighted sampling，实现batch级别混合）
- 当前实测发现该参数未生效（Num examples = 原始总数），待排查
- 替代方案：通过repeat_time调整各dataset比例至接近1:1:1

**关于overfit风险：**
- Stage 1中H2S和其他数据混合训练，overfit风险较低
- Stage 2是主要overfit风险点，通过save_strategy=epoch + eval选最优checkpoint缓解
- H2S经验：epoch 5左右最优，loss 1-1.5是sweet spot

**YouTube数据：** 因为noisy，不应出现在Stage 2，只用于Stage 1提供视觉多样性。

---

## 验证方式

每个实验跑完后：
1. Eval每个epoch的checkpoint在H2S benchmark上的分数
2. 对比BLEU-1/BLEU-4/ROUGE-L
3. 选最优checkpoint
4. 与baseline（BLEU-1 33.37%）对比
5. 最优checkpoint → Stage 2（可选）→ RL
