# 训练日志中有问题的Video文件总结

## 问题类型
所有问题都是：**文件不存在** (`No such file or directory`)

错误信息：
- `ERROR opening: /mnt/localssd/doc_sign_search/train_crop_videos_224/[filename].mp4, No such file or directory`
- `Failed to load video locally (decord: Error reading ...; opencv: cv2.VideoCapture failed to open file)`

## 有问题的Video文件列表

共发现 **13个** 有问题的video文件，每个文件在训练过程中都失败了 **6次**（可能是在不同的epoch中重复尝试加载）。

### 文件列表：

1. `13SBIjnC5qA_11-8-rgb_front.mp4` - 失败6次
2. `16W1jnZzfi4_25-8-rgb_front.mp4` - 失败6次
3. `1aJwX9nRlmk_21-2-rgb_front.mp4` - 失败6次
4. `279MO2nwC_E_7-2-rgb_front.mp4` - 失败6次
5. `5HDlLzELoeg_15-8-rgb_front.mp4` - 失败6次
6. `_93xgBE3NgA_2-5-rgb_front.mp4` - 失败6次
7. `a5yNwUSiYpA_11-5-rgb_front.mp4` - 失败6次
8. `a5yNwUSiYpA_12-5-rgb_front.mp4` - 失败6次
9. `dC0nsP_KIw4_8-8-rgb_front.mp4` - 失败6次
10. `DnWpxYlY4cc_6-5-rgb_front.mp4` - 失败6次
11. `EI6DNHOvn40_4-8-rgb_front.mp4` - 失败6次
12. `EI6DNHOvn40_5-8-rgb_front.mp4` - 失败6次
13. `EI6DNHOvn40_6-8-rgb_front.mp4` - 失败6次

## 统计信息

- **总失败次数**: 78次（13个文件 × 6次）
- **唯一有问题的文件数**: 13个
- **数据集**: how2sign
- **视频路径**: `/mnt/localssd/doc_sign_search/train_crop_videos_224/`

## 建议

1. **检查文件是否存在**: 确认这13个文件是否真的存在于指定路径
2. **检查文件权限**: 确认文件是否有读取权限
3. **检查路径映射**: 确认训练时的路径 `/mnt/localssd/doc_sign_search/train_crop_videos_224/` 是否正确映射到实际文件位置
4. **从训练数据中移除**: 如果这些文件确实不存在，建议从训练数据集中移除这些条目，避免重复失败

## 相关Video ID模式

注意到有些video ID有相似的模式：
- `EI6DNHOvn40_*` - 有3个不同的segment（4-8, 5-8, 6-8）
- `a5yNwUSiYpA_*` - 有2个不同的segment（11-5, 12-5）

这可能表明某些原始视频的多个segment都缺失了。


