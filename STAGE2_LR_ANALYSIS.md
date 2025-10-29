# Stage 2 学习率调整分析

## 当前配置

```bash
--freeze_vision_tower True    # Vision被冻结
--freeze_llm True             # LLM被冻结（只用LoRA）
--freeze_merger False         # Merger可训练
--learning_rate 3e-5          # LoRA学习率
--merger_lr 3e-5              # Merger/Projector学习率
```

## 为什么可以增加学习率？

### 1. **参数量大幅减少**
- **Stage 1**: 训练Vision + Merger + LoRA (~数千万参数)
- **Stage 2**: 只训练Merger + LoRA (~数百万参数)
- 更少的参数 → 可以承受更高的学习率

### 2. **任务更简单**
- **Stage 1**: 学习视觉特征 + 对齐 (复杂)
- **Stage 2**: 只做对齐 (相对简单)
- Vision特征已经固定，只需要学习映射关系

### 3. **梯度更稳定**
- Vision冻结后，梯度只流向merger和LoRA
- 梯度路径更短，更稳定
- 可以用更高的学习率

## 学习率选择建议

### 保守方案（推荐先试）
```bash
--learning_rate 5e-5    # 从3e-5增加到5e-5（约1.67倍）
--merger_lr 5e-5        # Merger也相应提高
```
**优点**：稳定，风险低
**预期**：训练更快，BLEU提升速度加快

### 中等方案
```bash
--learning_rate 7e-5    # 从3e-5增加到7e-5（约2.3倍）
--merger_lr 7e-5
```
**优点**：平衡速度和稳定性
**预期**：明显加快对齐速度

### 激进方案
```bash
--learning_rate 1e-4    # 从3e-5增加到1e-4（约3.3倍）
--merger_lr 1e-4
```
**优点**：最快速度
**风险**：
- Loss可能不稳定
- 可能过拟合
- 需要更频繁的checkpoint保存

## 推荐方案

**建议先用 5e-5 或 7e-5**：

1. **先从 5e-5 开始**（如果稳定，再考虑提高）
2. **监控前500步**：
   - Loss应该稳步下降
   - 如果loss震荡或上升，降低到 3e-5
   - 如果下降很快且稳定，可以尝试提高到 7e-5
3. **配合warmup**：使用 `--warmup_steps 200` 帮助稳定训练

## 修改脚本示例

### 保守版本（5e-5）
```bash
--learning_rate 5e-5 \
--merger_lr 5e-5 \
--warmup_steps 200 \
```

### 中等版本（7e-5）
```bash
--learning_rate 7e-5 \
--merger_lr 7e-5 \
--warmup_steps 300 \  # 稍微增加warmup
```

### 激进版本（1e-4）- 不推荐立即使用
```bash
--learning_rate 1e-4 \
--merger_lr 1e-4 \
--warmup_steps 400 \     # 增加warmup保护
--max_grad_norm 0.5 \    # 降低gradient clip
```

## 监控指标

训练时关注：
1. **Loss趋势**：应该稳步下降，不要剧烈震荡
2. **梯度范数**：不应该太大（< 1.0）
3. **BLEU提升速度**：每1000步评估一次，应该明显快于Stage 1

## 结论

✅ **推荐先从 5e-5 或 7e-5 开始**
⚠️ **1e-4 可能太激进，除非你能密切监控训练**

**最佳实践**：
- 先用 5e-5 训练1000步
- 如果稳定，可以提高到 7e-5
- 只有在非常有经验且能实时监控时才用 1e-4

