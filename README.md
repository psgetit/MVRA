## Install
1. Clone this repository
   ```bash
   git clone https://github.com/THUDM/MoELoRA_Riemannian.git
   ```
2. Install dependencies
   ```bash
   conda create -n moelora_riemannian python=3.10 -y
   conda activate moelora_riemannian
   pip install -r requirements.txt
   ```
## Prepare ScienceQA Data (for example)

1. Prepare the datasets by this script:
   ```bash
   python prepare_data.py \
     --save_path "./dataset/scienceqa" 
   ```

2. Organize your datasets in the following structure:
   ```
   MoELoRA_Riemannian/dataset/
   ├── scienceqa/
   │   ├── science_qa.hf
   │   ├── scienceqa_train.json
   │   ├── scienceqa_test.json
   │   └── ...
   └── ...
   ```

## How to Run
eg. python train.py --rank 32 --method vera --d 0.5 --number_expert 8 --top_k 1 --initial_gate_lr 1e-7 --initial_expert_lr 1e-4 --warm_up 60 --total_gradient_update_steps 1500 --eval_steps 2000 --dataset_name 'CommonsenseQA' --scheduler_max_steps 1510 --batch_size_256_plus_10 4 --accumulated_steps_256_plus_10 4


