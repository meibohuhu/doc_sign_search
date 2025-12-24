# YouTube Video Download Script - 依赖和安装说明

## 概述

这个脚本用于从 YouTube 下载视频，支持批量下载，自动检测和转换视频格式。

## 系统要求

- Linux 系统
- Conda 环境管理器
- Python 3.x

## 必需的库和工具

### 1. Conda 环境

脚本需要在 `internvl` conda 环境中运行。

```bash
# 激活 conda 环境
conda activate internvl
```

### 2. Python 包

#### yt-dlp
用于下载 YouTube 视频的核心工具。

```bash
# 在 conda internvl 环境中安装
conda activate internvl
pip install yt-dlp
# 或者
conda install -c conda-forge yt-dlp
```

**验证安装：**
```bash
python -m yt_dlp --version
```

### 3. FFmpeg

用于视频格式转换和合并（将 TS 格式转换为 MP4）。

```bash
# 在 conda internvl 环境中安装
conda activate internvl
conda install -c conda-forge ffmpeg
```

**验证安装：**
```bash
ffmpeg -version
```

### 4. libiconv

FFmpeg 的依赖库，用于字符编码转换。

```bash
# 在 conda internvl 环境中安装
conda activate internvl
conda install -c conda-forge libiconv
```

**验证安装：**
```bash
# 检查库文件是否存在
ls $CONDA_PREFIX/lib/libiconv.so*
```

## Python 标准库

脚本使用的 Python 标准库（无需额外安装）：
- `os` - 操作系统接口
- `subprocess` - 子进程管理
- `sys` - 系统相关参数和函数
- `time` - 时间相关功能
- `pathlib` - 路径操作

## 完整安装步骤

### 方法 1：使用 Conda（推荐）

```bash
# 1. 激活 conda 环境
conda activate internvl

# 2. 安装所有必需的包
conda install -c conda-forge yt-dlp ffmpeg libiconv

# 3. 验证安装
python -m yt_dlp --version
ffmpeg -version
```

### 方法 2：混合安装

```bash
# 1. 激活 conda 环境
conda activate internvl

# 2. 使用 conda 安装 ffmpeg 和 libiconv
conda install -c conda-forge ffmpeg libiconv

# 3. 使用 pip 安装 yt-dlp
pip install yt-dlp

# 4. 验证安装
python -m yt_dlp --version
ffmpeg -version
```

## 环境变量设置

脚本会自动设置以下环境变量，但也可以手动设置：

```bash
# 设置 LD_LIBRARY_PATH（脚本会自动设置）
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# 可选：设置输入文件和输出目录
export YOUTUBE_DOWNLOAD_INPUT_FILE="/path/to/video_ids.txt"
export YOUTUBE_DOWNLOAD_OUTPUT_DIR="/path/to/output/directory"
```

## 使用方法

### 基本用法

```bash
cd /home/mh2803/projects/sign_language_llm/data_aggregation_output
bash start_download.sh
```

### 指定输入文件和输出目录

```bash
bash start_download.sh [input_file] [output_dir]
```

**示例：**
```bash
# 使用默认值
bash start_download.sh

# 指定输入文件
bash start_download.sh youtube_video_ids_diff.txt

# 指定输入文件和输出目录
bash start_download.sh \
  youtube_video_ids_diff.txt \
  /shared/rc/llm-gen-agent/mhu/videos/youtube_asl/downloads
```

## 故障排除

### 问题 1: `ffmpeg: error while loading shared libraries: libiconv.so.2`

**解决方案：**
```bash
conda activate internvl
conda install -c conda-forge libiconv
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
```

### 问题 2: `No module named 'yt_dlp'`

**解决方案：**
```bash
conda activate internvl
pip install yt-dlp
# 或
conda install -c conda-forge yt-dlp
```

### 问题 3: `ffmpeg: command not found`

**解决方案：**
```bash
conda activate internvl
conda install -c conda-forge ffmpeg
```

### 问题 4: 下载的视频是 TS 格式而不是 MP4

**原因：** YouTube 可能只提供 TS 格式的流。

**解决方案：** 脚本会自动尝试转换为 MP4。如果转换失败，确保已安装 ffmpeg 和 libiconv。

## 文件说明

- `youtube_download.py` - 主下载脚本
- `start_download.sh` - 启动脚本（设置环境并运行 Python 脚本）
- `youtube_video_ids_stage2.txt` - 视频 ID 列表（严格过滤）
- `youtube_video_ids_stage2_notstrict.txt` - 视频 ID 列表（非严格过滤）
- `youtube_video_ids_diff.txt` - 差集视频 ID 列表
- `www.youtube.com_cookies.txt` - YouTube cookies 文件（可选，用于登录下载）

## 注意事项

1. **Cookies 文件**：如果需要下载私有或受限视频，需要提供 cookies 文件。
2. **网络连接**：确保网络连接稳定，下载大量视频可能需要较长时间。
3. **磁盘空间**：确保输出目录有足够的磁盘空间。
4. **速率限制**：脚本在每次下载之间有 2 秒延迟，以避免触发 YouTube 的速率限制。

## 版本信息

- Python: 3.x
- yt-dlp: 2025.12.08 或更高版本
- ffmpeg: 5.0.1 或更高版本
- libiconv: 1.17 或更高版本

## 更新日志

- 2024-12-23: 初始版本
  - 支持批量下载
  - 自动格式检测和转换
  - 支持自定义输入文件和输出目录

