#!/bin/sh
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=a100-80:1  # Request 1 A100-40 GPU
#SBATCH --time=3:00:00   # Maximum allowed time
#SBATCH --job-name=gpu_job
#SBATCH --output=dataset.log
#SBATCH --mem=64G            # 设置内存为64GB
# Step 1: 确保 GPU 可用
echo "Checking GPU availability..."
nvidia-smi

# Step 3: 检查 PyTorch CUDA 支持
echo "Checking PyTorch and CUDA availability..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

# Step 4: 优化显存分配
# 设置 PyTorch 显存增长选项，避免爆显存
export CUDA_VISIBLE_DEVICES=0  # 限制使用第 0 块 GPU
export CUDA_LAUNCH_BLOCKING=1  # 强制同步 GPU 操作，方便调试

# Step 5: 调整批量大小或梯度累积
echo "Running tasks with optimized settings..."

python3 prepare_data.py --save_path "./dataset/scienceqa"

echo "All tasks completed successfully!"
