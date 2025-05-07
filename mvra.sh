#!/bin/sh
#SBATCH --mail-type=END,FAIL
#SBATCH --partition=gpu
#SBATCH --gpus=h100-96:1
#SBATCH --time=3:00:00   # 最大运行时间 3 小时
#SBATCH --job-name=gpu_job
#SBATCH --output=mvra.log
#SBATCH --mem=64G         # 设置内存为 64GB

# Step 1: 确保 GPU 可用
echo "Checking GPU availability..."
#watch -n 2 -d nvidia-smi

# 这里不需要显式设置 CUDA_VISIBLE_DEVICES，SLURM 会自动分配 GPU

#export CUDA_LAUNCH_BLOCKING=1  # 强制同步 GPU 操作，方便调试

# 单卡训练，不需要分布式初始化
#python train.py --rank 512 --method vera
#python train.py --rank 1024 --method vera
#python train.py --rank 512 --method vera --d 0.5
#python train.py --rank 1024 --method vera --d 0.5
#python train.py --rank 8 --method lora
#python train.py --rank 16 --method lora
#python train.py --rank 32 --method lora
#python train.py --rank 512 --method vera --number_expert 20 --top_k 10 --d 0.1
#python train.py --rank 1024 --method vera --number_expert 20 --top_k 10 --d 0.1
#python train.py --rank 8 --method lora --number_expert 8 --top_k 4
#python train.py --rank 16 --method lora --number_expert 8 --top_k 4
#python train.py --rank 512 --method vera --number_expert 8 --top_k 4 --d 0.5
#python train.py --rank 1024 --method vera --number_expert 8 --top_k 4 --d 0.5
#python train.py --rank 512 --method vera --number_expert 8 --top_k 4 --d 1
#python train.py --rank 1024 --method vera --number_expert 8 --top_k 4 --d 1
#python train.py --rank 512 --method vera --number_expert 8 --top_k 4 --d 0.5
#python train.py --rank 1024 --method vera --number_expert 8 --top_k 4 --d 0.5
#python train.py --rank 2048 --method vera --number_expert 8 --top_k 4 --d 0.5 --batch_size_256_plus_20 6
#python3 train.py --rank 8 --d 0.5 --method lora --number_expert 12 --top_k 6 --initial_gate_lr 3e-8 --initial_expert_lr 1e-5 --batch_size_256_plus_20 8
#python3 train.py --rank 16 --d 0.5 --method lora --number_expert 12 --top_k 6 --initial_gate_lr 3e-8 --initial_expert_lr 1e-5 --batch_size_256_plus_20 8
#python3 train.py --rank 512 --d 0.5 --method vera --number_expert 12 --top_k 3 --initial_gate_lr 3e-8 --initial_expert_lr 1e-5 --batch_size_256_plus_20 8
#python3 train.py --rank 8 --method lora --number_expert 10 --top_k 3 --initial_gate_lr 1e-7 --initial_expert_lr 5e-5 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python3 train.py --rank 16 --method lora --number_expert 10 --top_k 3 --initial_gate_lr 1e-7 --initial_expert_lr 5e-5 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python3 train.py --rank 8 --method lora --initial_gate_lr 5e-7 --initial_expert_lr 5e-5 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python3 train.py --rank 16 --method lora --initial_gate_lr 5e-7 --initial_expert_lr 5e-5 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 512 --method vera --d 0.5 --initial_gate_lr 5e-7 --initial_expert_lr 5e-4 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 1024 --method vera --d 0.5 --initial_gate_lr 5e-7 --initial_expert_lr 5e-4 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 512 --method vera --d 0.5 --number_expert 10 --top_k 3 --initial_gate_lr 5e-7 --initial_expert_lr 5e-4 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 1024 --method vera --d 0.5 --number_expert 10 --top_k 3 --initial_gate_lr 5e-7 --initial_expert_lr 5e-4 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 256 --method vera --d 0.5 --initial_gate_lr 5e-7 --initial_expert_lr 5e-4 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 256 --method vera --d 0.5 --number_expert 16 --top_k 5 --initial_gate_lr 5e-7 --initial_expert_lr 5e-4 --warm_up 300 --total_gradient_update_steps 1000 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 32 --method vera --d 0.5 --initial_gate_lr 5e-8 --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 32 --method vera --d 0.5 --number_expert 12 --top_k 5 --initial_gate_lr 5e-8 --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 16 --method lora --number_expert 10 --top_k 5 --d 0.5 --initial_gate_lr 3e-8 --initial_expert_lr 5e-5 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 1000
#python train.py --rank 0 --method base --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 0 --method base --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000
#python train.py --rank 256 --method vera --number_expert 10 --top_k 5 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 1000
#python train.py --rank 256 --method vera --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 1000
#python train.py --rank 8 --method lora --d 0.5 --initial_gate_lr 3e-8 --initial_expert_lr 5e-5 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 1000
#python train.py --rank 8 --method lora --number_expert 10 --top_k 5 --d 0.5 --initial_gate_lr 3e-8 --initial_expert_lr 5e-5 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 1000
#python train.py --rank 16 --method lora --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 16 --method lora --number_expert 10 --top_k 2 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 512 --method vera --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 1000 --dataset_name 'CommonsenseQA'
#python train.py --rank 512 --method vera --number_expert 10 --top_k 1 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 850
#python train.py --rank 512 --method vera --number_expert 10 --top_k 2 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 850
#python train.py --rank 512 --method vera --number_expert 10 --top_k 3 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 850
#python train.py --rank 512 --method vera --number_expert 10 --top_k 4 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 850
#python train.py --rank 512 --method vera --number_expert 10 --top_k 5 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 850
#python train.py --rank 512 --method vera --number_expert 10 --top_k 6 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 850
#python train.py --rank 512 --method vera --number_expert 10 --top_k 1 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 5e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 850 --dataset_name 'CommonsenseQA'
#python train.py --rank 512 --method vera --number_expert 10 --top_k 2 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 800 --eval_steps 1000 --scheduler_max_steps 850 --dataset_name 'CommonsenseQA'
#python train.py --rank 512 --method vera --number_expert 10 --top_k 3 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 500 --eval_steps 1000 --scheduler_max_steps 550 --dataset_name 'CommonsenseQA'
#python train.py --rank 512 --method vera --number_expert 10 --top_k 4 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 500 --eval_steps 1000 --scheduler_max_steps 550 --dataset_name 'CommonsenseQA'
#python train.py --rank 512 --method vera --number_expert 10 --top_k 5 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 500 --eval_steps 1000 --scheduler_max_steps 550 --dataset_name 'CommonsenseQA'
#python train.py --rank 512 --method vera --number_expert 10 --top_k 6 --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 100 --total_gradient_update_steps 500 --eval_steps 1000 --scheduler_max_steps 550 --dataset_name 'CommonsenseQA'
#python train_new.py --rank 256 --method vera --d 0.5 --number_expert 10 --top_k 1 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1500 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1520 --res_path 'new_test1.csv'
#python train_new1.py --rank 256 --method vera --d 0.5 --number_expert 10 --top_k 2 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1200 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1220 --res_path 'new_test1.csv' --batch_size_256_plus_10 4
##python train_new.py --rank 256 --method vera --d 0.5 --number_expert 10 --top_k 1 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1200 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1220
#python train_new1.py --rank 512 --method vera --d 0.5 --number_expert 10 --top_k 3 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1200 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1220 --res_path 'new_test1.csv' --batch_size_256_plus_10 4
#python train_new1.py --rank 512 --method vera --d 0.5 --number_expert 10 --top_k 2 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1200 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1220 --res_path 'new_test1.csv' --batch_size_256_plus_10 4
#python train.py --rank 64 --method vera --d 0.5 --number_expert 12 --top_k 1 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 800 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 820 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 64 --method vera --d 0.5 --number_expert 16 --top_k 1 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 800 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 820 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 64 --method vera --d 0.5 --number_expert 12 --top_k 2 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 800 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 820 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 64 --method vera --d 0.5 --number_expert 16 --top_k 2 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 800 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 820 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 64 --method vera --d 0.5 --number_expert 12 --top_k 3 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 800 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 820 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 64 --method vera --d 0.5 --number_expert 16 --top_k 3 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 800 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 820 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train_new1.py --rank 32 --method vera --d 0.5 --contra_rate 0.1 --number_expert 12 --top_k 3 --initial_gate_lr 8e-8 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1500 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1500 --res_path 'results/new_test1.csv' --batch_size_256_plus_10 8 --accumulated_steps_256_plus_10 2
#python train_new1.py --rank 32 --method vera --d 0.5 --contra_rate 0.1 --number_expert 12 --top_k 4 --initial_gate_lr 8e-8 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1500 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1500 --res_path 'results/new_test1.csv' --batch_size_256_plus_10 8 --accumulated_steps_256_plus_10 2
#python train.py --rank 16 --method lora --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 800 --eval_steps 2000 --dataset_name 'SST-2' --scheduler_max_steps 1010 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 256 --method vera --d 0.5 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1000 --eval_steps 2000 --dataset_name 'SST-2' --scheduler_max_steps 1010 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 16 --method lora --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1000 --eval_steps 2000 --dataset_name 'ScienceQA' --scheduler_max_steps 1010 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 256 --method vera --d 0.5 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1000 --eval_steps 2000 --dataset_name 'ScienceQA' --scheduler_max_steps 1010 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 16 --method lora --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1000 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1010 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 256 --method vera --d 0.5 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1000 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1010 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4

python train.py --rank 32 --method vera --d 0.5 --number_expert 8 --top_k 1 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1500 --eval_steps 2000 --dataset_name 'SST-2' --scheduler_max_steps 1510 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
python train.py --rank 32 --method vera --d 0.5 --number_expert 8 --top_k 1 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1500 --eval_steps 2000 --dataset_name 'ScienceQA' --scheduler_max_steps 1510 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
python train.py --rank 32 --method vera --d 0.5 --number_expert 8 --top_k 1 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1500 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1510 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4
#python train.py --rank 16 --method lora --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1000 --eval_steps 2000 --dataset_name 'SST-2' --scheduler_max_steps 1010 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4

#python train_new.py --rank 512 --method vera --d 0.5 --number_expert 10 --top_k 1 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1200 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1220

echo "All tasks completed successfully!"
#***SBATCH --mail-type=END,FAIL
#***sbatch -J gpujob --gres=gpu:h100-47:1 job.sh
python train_new1.py --rank 32 --method vera --d 0.5 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1500 --eval_steps 2000 --dataset_name ScienceQA --scheduler_max_steps 1510 --batch_size_256_plus_10 2 --accumulated_steps_256_plus_10 4