#!/bin/bash
HUGGINGFACE_TOKEN=""


# setup huggingface ###############################################################################
echo "正在登录Huggingface..."
# 使用正确的登录方式
huggingface-cli login --token "$HUGGINGFACE_TOKEN"

# download and extract videos ###############################################################################
echo "正在下载和解压视频..."
DOWNLOAD_SCRIPT="/home/stu2/s15/mh2803/workspace/doc_sign_search/download_and_extract_videos_from_hf.py"
VIDEO_DATASET_NAME="${VIDEO_DATASET_NAME:-PhoenixHu/sign_mllm_how_224_train_val}"
VIDEO_OUTPUT_DIR="${VIDEO_OUTPUT_DIR:-/scratch/mh2803/train_crop_videos_224}"
FORCE_DOWNLOAD_VIDEOS="${FORCE_DOWNLOAD_VIDEOS:-0}"

# 检查下载脚本是否存在
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "⚠️  警告: 未找到下载脚本 $DOWNLOAD_SCRIPT"
    echo "   跳过视频下载步骤"
else
    # 检查视频是否已存在
    if [ -d "$VIDEO_OUTPUT_DIR" ] && [ "$(ls -A $VIDEO_OUTPUT_DIR 2>/dev/null)" ]; then
        echo "📁 视频目录已存在且不为空: $VIDEO_OUTPUT_DIR"
        echo "   跳过下载步骤（如需重新下载，请删除该目录或设置 FORCE_DOWNLOAD_VIDEOS=1）"
        
        # 如果设置了强制下载，则继续下载
        if [ "${FORCE_DOWNLOAD_VIDEOS:-0}" != "1" ]; then
            echo "✅ 使用现有视频文件"
        else
            echo "🔄 强制下载模式：将重新下载视频"
            python "$DOWNLOAD_SCRIPT" \
                --dataset_name "$VIDEO_DATASET_NAME" \
                --output_dir "$VIDEO_OUTPUT_DIR" \
                --token "$HUGGINGFACE_TOKEN" \
                --remove_archives
        fi
    else
        echo "📥 开始下载视频数据集: $VIDEO_DATASET_NAME"
        echo "📁 输出目录: $VIDEO_OUTPUT_DIR"
        
        # 运行下载脚本（使用conda环境中的python）
        python "$DOWNLOAD_SCRIPT" \
            --dataset_name "$VIDEO_DATASET_NAME" \
            --output_dir "$VIDEO_OUTPUT_DIR" \
            --token "$HUGGINGFACE_TOKEN" \
            --remove_archives
        
        if [ $? -eq 0 ]; then
            echo "✅ 视频下载和解压完成！"
            echo "📁 视频位置: $VIDEO_OUTPUT_DIR"
        else
            echo "⚠️  警告: 视频下载失败，请检查网络连接和数据集名称"
        fi
    fi
fi

