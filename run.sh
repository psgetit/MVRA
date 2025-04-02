#!/bin/sh
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-80:1  # 请求 1 块 A100 GPU
#SBATCH --time=3:00:00   # 最大运行时间 3 小时
#SBATCH --job-name=gpu_job
#SBATCH --output=run.log
#SBATCH --mem=64G         # 设置内存为 64GB

# Step 1: 确保 GPU 可用
echo "Checking GPU availability..."
nvidia-smi

# 这里不需要显式设置 CUDA_VISIBLE_DEVICES，SLURM 会自动分配 GPU

export CUDA_LAUNCH_BLOCKING=1  # 强制同步 GPU 操作，方便调试

# 单卡训练，不需要分布式初始化
python3 train_llama.py ScienceQA sgd riemannian

echo "All tasks completed successfully!"
