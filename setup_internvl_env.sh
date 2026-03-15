source /home/stu2/s15/mh2803/anaconda3/etc/profile.d/conda.sh
conda activate internvl

# 安装基础依赖
/home/stu2/s15/mh2803/anaconda3/envs/internvl/bin/pip install psutil packaging ninja

# 安装 PyTorch
/home/stu2/s15/mh2803/anaconda3/envs/internvl/bin/pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121



# 安装其他依赖（如果 requirements_internvl.txt 存在）
/home/stu2/s15/mh2803/anaconda3/envs/internvl/bin/pip install -r /home/stu2/s15/mh2803/workspace/doc_sign_search/requirements_internvl.txt

# 安装 flash-attn
source /home/stu2/s15/mh2803/anaconda3/etc/profile.d/conda.sh
conda activate internvl
conda install -c conda-forge cuda-toolkit=13.1
export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH
source /home/stu2/s15/mh2803/anaconda3/etc/profile.d/conda.sh && conda activate internvl && /home/stu2/s15/mh2803/anaconda3/envs/internvl/bin/pip install flash-attn --no-build-isolation --no-cache-dir
##### check flash attention
python -c "import flash_attn; print(flash_attn.__version__)"

