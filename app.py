from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


# 引入你现有的函数
from train_new1 import main as train_main, CANDIDATE_DATASETS

app = FastAPI(
    title="MVRA Training API",
    description="用 FastAPI 部署 MVRA 模型训练接口",
    version="1.0.0"
)

# ✅ 添加 CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 或 ["http://localhost:3000"] 更安全
    allow_credentials=True,
    allow_methods=["*"],  # 支持 OPTIONS、POST 等所有方法
    allow_headers=["*"],
)

class TrainConfig(BaseModel):
    method: str = Field(..., description="微调方法, e.g., 'lora' or 'vera'")
    target_modules: List[str] = Field(..., description="目标模块列表")
    number_expert: int = Field(..., description="专家数量 N")
    expert_rank: int = Field(..., description="低秩分解秩 r")
    top_k: int = Field(..., description="Top-k 激活专家数")
    d_initial: float = Field(..., description="专家初始 d 参数均值")
    initial_expert_lr: float = Field(..., description="专家学习率")
    initial_gate_lr: float = Field(..., description="门控网络学习率")
    warm_up: int = Field(..., description="预热步数")
    total_gradient_update_steps: int = Field(..., description="总梯度更新步数")
    eval_steps: int = Field(..., description="评估频率")
    max_gate_grad_norm: float = Field(..., description="门控梯度裁剪范数")
    dropout: float = Field(..., description="dropout 概率")
    contra_rate: float = Field(..., description="对比学习权重")
    dataset_name: str = Field(..., description="数据集名称")
    res_path: str = Field("results/final_results.csv", description="结果保存路径")
    batch_sizes: int = Field(..., description="batch size")
    accumulated_steps: int = Field(..., description="累积步数")
    scheduler_max_steps: int = Field(..., description="调度器最大步数")
@app.get("/metadata")
def get_metadata():
    return {
        "methods": ["VeRA", "LoRA"],
        "target_modules": "q_proj,k_proj,v_proj,o_proj",
        "max_window_length": 256,
        "max_generated_tokens": 20,
    }

@app.post("/train")
def train_endpoint(config: TrainConfig):
    # 设置全局变量
    global MyDataset, DATASET_NAME, DATASET_PATH
    if config.dataset_name not in CANDIDATE_DATASETS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的数据集名称: {config.dataset_name}，请从以下列表中选择: {list(CANDIDATE_DATASETS.keys())}"
        )
    else:
        MyDataset, DATASET_PATH = CANDIDATE_DATASETS[config.dataset_name]
    DATASET_NAME = config.dataset_name
    if DATASET_NAME not in CANDIDATE_DATASETS:
        raise HTTPException(status_code=400, detail="Unknown dataset_name")
    MyDataset, DATASET_PATH = CANDIDATE_DATASETS[DATASET_NAME]


    # 准备 candidate_configs
    candidate_configs = {
        "method": config.method,
        "target_modules": config.target_modules,
        "number_expert": config.number_expert,
        "expert_rank": config.expert_rank,
        "top_k": config.top_k,
        "d_initial": config.d_initial,
        "initial_expert_lr": config.initial_expert_lr,
        "initial_gate_lr": config.initial_gate_lr,
        "warm_up": config.warm_up,
        "total_gradient_update_steps": config.total_gradient_update_steps,
        "eval_steps": config.eval_steps,
        "dropout": config.dropout,
        "per_gpu_batch_size": config.batch_sizes,
        "accumulated_steps": config.accumulated_steps,
        "scheduler_max_steps": config.scheduler_max_steps,
        "max_gate_grad_norm": config.max_gate_grad_norm,
        "res_path": config.res_path,
        "contra_rate": config.contra_rate,
        'max_window_length': MyDataset.MAX_SAMPLE_INPUT_LENGTH,  # 128/256/...
        'max_generated_tokens': MyDataset.MAX_SAMPLE_OUTPUT_LENGTH  # 10/...
    }

    # 调用训练主函数
    # try:
    result = train_main(candidate_configs, DATASET_NAME, MyDataset, DATASET_PATH)
    return {"status": "success", "result": result}
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
