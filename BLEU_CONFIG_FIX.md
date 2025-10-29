# BLEU评估配置修复说明

## 问题
之前使用的 `sacreBLEU` 配置**没有明确指定 `-lc --tokenize 13a`**参数。

## 修复内容

### 修复前
```python
bleu = sacrebleu.corpus_bleu(predictions, [ground_truths])
```

**问题**：
- ❌ 默认 `lowercase=False`（没有使用 `-lc`）
- ✅ 默认 `tokenize='13a'`（但未显式指定）

### 修复后
```python
bleu = sacrebleu.corpus_bleu(
    predictions, 
    [ground_truths],
    lowercase=True,      # -lc flag: lowercase the data
    tokenize='13a'       # --tokenize 13a: use 13a tokenization
)
```

**修复后**：
- ✅ 明确指定 `lowercase=True`（对应 `-lc`）
- ✅ 明确指定 `tokenize='13a'`（对应 `--tokenize 13a`）

## 已修复的文件

1. **`evaluation/ssvp_evaluation.py`**
   - `calculate_bleu_scores_ssvp()` 函数
   - 主要评估模块

2. **`scripts/add_bleu_scores_ssvp.py`**
   - `calculate_individual_bleu_ssvp()` 函数
   - 批处理BLEU计算脚本

## 影响

### 评估结果变化
使用 `lowercase=True` 后，BLEU分数可能会有变化：
- **大写/小写匹配**现在会被正确识别
- 例如："Hello" vs "hello" 现在会被视为匹配

### 建议操作

1. **重新计算之前的评估结果**：
   ```bash
   # 如果之前的结果使用了错误的配置，需要重新计算
   python scripts/add_bleu_scores_ssvp.py <previous_results.json>
   ```

2. **验证新配置**：
   ```python
   from evaluation.ssvp_evaluation import calculate_bleu_scores_ssvp
   
   predictions = ['Hello world.']
   ground_truths = ['hello world.']
   
   results = calculate_bleu_scores_ssvp(predictions, ground_truths)
   # 现在应该正确匹配（lowercase=True）
   ```

## 关于去BPE

sacreBLEU的 `tokenize='13a'` 会自动处理：
- 标点符号标准化
- 文本分词
- 大小写转换（如果 `lowercase=True`）

**BPE tokens**通常在tokenizer解码时通过 `skip_special_tokens=True` 已经去除，无需额外处理。

## 验证方法

```python
import sacrebleu

# 测试配置
pred = ['Hello World.']
ref = [['hello world.']]

# 新的正确配置
bleu = sacrebleu.corpus_bleu(pred, ref, lowercase=True, tokenize='13a')
print(f"BLEU score: {bleu.score}")
# 应该给出高分（因为lowercase=True后会匹配）
```

---

**状态**: ✅ 已修复
**日期**: 2025-01-29

