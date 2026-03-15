How2Sign 数据集：
聚合了 2,192 个唯一视频
从 31,165 个句子片段合并
平均每个视频：1,293 个字符，251 个单词
输出文件：data_aggregation_output/how2sign_video_texts.json (2.8 MB)

YouTube 数据集：
聚合了 9,541 个视频
合并了 title + description + captions
平均每个视频：3,234 个字符，546 个单词
输出文件：data_aggregation_output/youtube_video_texts.json (62 MB)


Sign1news
原始视频数：395
删除的视频：45（caption_languages 为多语言且 caption_timestamps 有问题）
total_sentences: 13457

cd /home/mh2803/projects/sign_language_llm/data_aggregation_output
MAX_VIDEOS=6923
USE_STRICT=true BATCH_SIZE=20 python stage2_llm_how2sign_filter.py
www的句子，（）的不裁。**的不裁
(expressive gesture), http://, &gt
 []
