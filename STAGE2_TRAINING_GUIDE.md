# Stage 2 训练指南：Vision-Text 对齐

## 策略说明

### 为什么这样做有效？

**Stage 1 已完成：**
- ✅ Vision encoder已经学习了sign language的视觉特征（虽然可能不够完美）
- ✅ Vision LR (1e-5) 训练了6000步，特征已经有一定基础

**Stage 2 目标：**
- 🔒 **Freeze Vision Encoder**：保持已学到的视觉特征，不再变化
- 🎯 **训练 Merger (Projector)**：学习如何将vision特征映射到text空间
- 🎯 **训练 LoRA**：学习如何生成正确的sign language翻译

**为什么这样有效：**
1. **解耦问题**：先学视觉特征，再学对齐，比同时学更容易
2. **稳定训练**：Fixed vision encoder提供稳定的输入，merger更容易收敛
3. **更少参数**：只训练merger+LoRA，训练更快，显存占用更少
4. **专门优化对齐**：merger专注于vision-text bridge，学习效率更高

---

## 使用方法

### 1. 修改Checkpoint路径

在 `finetune_qwen2vl_how2sign_4xa100_stage2_alignment.sh` 中：

```bash
# 修改这一行，指向你的Stage 1 checkpoint
STAGE1_CHECKPOINT="/shared/rc/llm-gen-agent/mhu/qwen2.5vl/1018/qwen2vl_how2sign_4xa100_filtered_32batchsize_robust/checkpoint-6000"
```

**选择哪个checkpoint？**
- 通常选最后一个（checkpoint-6000）
- 或者选loss最低的checkpoint
- 如果中途有最好的BLEU checkpoint，也可以选那个

### 2. 确认输出目录

```bash
--output_dir /shared/rc/llm-gen-agent/mhu/qwen2.5vl/1018/qwen2vl_how2sign_4xa100_stage2_alignment/
```

这个目录会自动创建，保存Stage 2的checkpoints。

### 3. 运行训练

```bash
sbatch qwenvl/Qwen2-VL-Finetune/scripts/how2sign_train_update/finetune_qwen2vl_how2sign_4xa100_stage2_alignment.sh
```

---

## 关键配置说明

### Stage 2 训练参数

```bash
--freeze_vision_tower True     # 🔒 Freeze vision encoder
--freeze_llm True              # 🔒 Keep LLM frozen (LoRA only)
--freeze_merger False          # ✅ Train merger/projector
--merger_lr 3e-5               # Projector学习率（与LoRA相同）
--learning_rate 3e-5           # LoRA学习率
--lora_rank 64                 # 增加到64（比Stage 1的32更大）
--lora_alpha 128               # Alpha = 2 * rank
--max_steps 3000               # 训练3000步（对齐不需要太久）
```

### 为什么增加LoRA rank？

- Stage 1用的是rank=32，容量可能不够
- Stage 2提高到64，提供更多参数空间学习对齐
- 因为vision已冻结，显存压力不大，可以增加rank

---

## 预期效果

### 训练过程中

**Loss趋势：**
- 初始loss可能比Stage 1结束时稍高（因为只训练merger+LoRA）
- 应该快速下降，因为任务更简单（只对齐，不需要学视觉）

**BLEU改善：**
- 应该能在1000步内看到BLEU提升
- 目标是：训练集BLEU 6% → 15-25%+
- 推理集BLEU 3% → 10-20%+

### 评估策略

**建议每1000步评估一次：**
- 在training subset上评估BLEU
- 如果BLEU开始plateau，可以考虑停止

---

## 故障排查

### 问题1：Checkpoint加载失败

**错误：** `Checkpoint not found` 或 `Can't load checkpoint`

**解决：**
1. 确认checkpoint路径正确
2. 确认checkpoint目录包含：
   - `adapter_model.bin` (LoRA weights)
   - `non_lora_state_dict.bin` (non-LoRA weights)
   - `training_state.json`
3. 如果是DeepSpeed checkpoint，确保使用DeepSpeed加载

### 问题2：Loss不下降

**可能原因：**
1. Vision encoder freeze失败
2. Merger没有被正确训练

**检查：**
```bash
# 在训练日志中查看
grep "requires_grad" <log_file>
# 确认vision参数requires_grad=False
```

### 问题3：BLEU不提升

**可能原因：**
1. Merger学习率太低
2. LoRA容量不够
3. 训练步数不够

**解决方案：**
- 尝试增加 `--merger_lr` 到 `5e-5`
- 增加 `--lora_rank` 到 128
- 增加 `--max_steps` 到 5000

---

## 下一步（可选Stage 3）

如果Stage 2效果不错但还不够完美，可以考虑：

**Stage 3: Fine-tuning everything**
```bash
--freeze_vision_tower False   # 重新unfreeze
--vision_lr 5e-6               # 很小的LR微调vision
--merger_lr 1e-5               
--lora继续训练
```

但通常Stage 2就足够了！

---

## 时间估计

- **Stage 2训练时间**：约12-24小时（3000步，4xA100）
- **评估时间**：每次评估约30分钟

**建议工作流：**
1. 提交Stage 2训练
2. 每1000步检查一次BLEU
3. 如果3000步后仍不够，可以继续训练

---

## 总结

**这个策略的优势：**
✅ 充分利用Stage 1学到的视觉特征
✅ 专注于对齐问题（这是主要瓶颈）
✅ 训练更快、更稳定
✅ 预期BLEU提升显著

**风险：**
⚠️ 如果Stage 1的vision特征太差，Stage 2可能救不回来（但根据你的情况，应该没问题）
⚠️ 如果merger容量不够，可能需要增加参数

祝训练顺利！🎉

