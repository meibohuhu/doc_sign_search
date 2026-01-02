

# setup internvl conda environment ###############################################################################
echo "接受 ToS..."
$HOME/anaconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
$HOME/anaconda3/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

echo "正在创建internvl环境..."
# 检查环境是否已存在
if conda env list | grep -q "^internvl "; then
    echo "环境 internvl 已存在，跳过创建"
else
    echo "正在创建新环境 internvl..."
    $HOME/anaconda3/bin/conda create -n internvl python=3.10 -y
fi

echo "正在进入项目目录..."
cd /code/doc_sign_search

echo "正在激活internvl环境并安装包..."
# 激活环境并安装包
conda activate internvl

############ 安装包
echo "正在安装构建依赖（psutil, packaging, ninja）..."
pip install psutil packaging ninja

echo "正在安装PyTorch with CUDA 12.1..."
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

echo "正在安装flash-attn..."
if pip install flash-attn==2.5.7 --no-build-isolation; then
    echo "✅ flash-attn 安装成功"
else
    echo "⚠️  第一次安装失败，尝试使用 --no-deps 选项..."
    pip install flash-attn==2.5.7 --no-build-isolation --no-deps || echo "⚠️  flash-attn 安装失败，但可以继续（某些功能可能较慢）"
fi

echo "正在使用requirements_internvl.txt安装其他依赖..."
pip install -r /code/doc_sign_search/requirements_internvl.txt

pip cache purge


