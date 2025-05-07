import os
import argparse
import time
from instruct.utils import get_parameters_count
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import VeraConfig, LoraConfig, get_peft_model
import peft
import csv
from moe_vera_new1 import MoEVeRA
from moe_lora import MoELoRA
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from dataset import CoLADataset
from dataset import ScienceQADataset
from dataset import CommonsenseQADataset
from dataset import SST2Dataset
from dataset import MRPCDataset
from dataset import QQPDataset
from dataset import QNLIDataset
from dataset import STSBDataset
from dataset import WNLIDataset
from dataset import MixedQADataset
from dataset import ImageText2TextDataset
from dataset import VMCBenchDataset
from datetime import datetime
from utils import init_lora_A, init_lora_B, init_lambda_b, init_lambda_d, init_lambda_d_random
import random
model_name = "meta-llama/Llama-3.2-1B-Instruct"
hf_token = "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"
MyDataset = None
DATASET_PATH = None
DATASET_NAME = None
glue_flag = 0

CANDIDATE_DATASETS = {
    "ScienceQA": [ScienceQADataset, "./dataset/scienceqa/science_qa.hf"],
    "CommonsenseQA": [CommonsenseQADataset, "./dataset/CommonsenseQA"],

    "CoLA": [CoLADataset, "./dataset/glue_data/CoLA"],
    "SST-2": [SST2Dataset, "./dataset/glue_data/SST-2"],
    "MRPC": [MRPCDataset, "./dataset/glue_data/MRPC"],
    "QQP": [QQPDataset, "./dataset/glue_data/QQP"],
    "QNLI": [QNLIDataset, "./dataset/glue_data/QNLI"],
    "STS-B": [STSBDataset, "./dataset/glue_data/STS-B"],
    "WNLI": [WNLIDataset, "./dataset/glue_data/WNLI"],

    "MixedQA": [MixedQADataset, "./datasets"],

    "ImageText2Text": [ImageText2TextDataset, "./dataset/visual7w"],
    "VMCBench": [VMCBenchDataset, "./dataset/VMCBench/data"]
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def save_training_results(csv_file_path, training_params, test_metrics, training_time, params_trainable, params_total):
    """
    Save the training parameters and test results into a CSV file.

    Parameters:
        csv_file_path (str): The path to the CSV file.
        training_params (dict): A dictionary of training parameters, e.g., {'batch_size': 32, 'number_experts': 4}.
        test_metrics (dict): A dictionary of test metrics, e.g., {'accuracy': 0.85}.
        training_time (float): Total training time in seconds.
        params_trainable (int): The number of trainable parameters.
        params_total (int): The total number of model parameters.

    Returns:
        None
    """
    # Construct CSV field names (column headers): combine keys from training_params and test_metrics,
    # and add 'training_time', 'params_trainable', and 'params_total' columns.
    fieldnames = list(training_params.keys()) + ['training_time', 'params_trainable', 'params_total'] + list(test_metrics.keys())

    # Check if the CSV file already exists.
    file_exists = os.path.isfile(csv_file_path)

    # Open the CSV file in append mode.
    with open(csv_file_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # If the file does not exist, write the header.
        if not file_exists:
            writer.writeheader()

        # Merge all data into one row using dictionary unpacking.
        row_data = {**training_params,
                    'training_time': training_time,
                    'params_trainable': params_trainable,
                    'params_total': params_total,
                    **test_metrics
                   }

        # Write the data row to the CSV file.
        writer.writerow(row_data)
    return row_data



def print_time_taken(start_time, step_name):
    end_time = time.time()
    print(f"{step_name} took {end_time - start_time:.4f} seconds")
    return end_time


def optimizer_set(model, initial_expert_lr, initial_gate_lr, number_experts, adapter_type):
    if adapter_type == "vera":
        expert_params = [p for n, p in model.named_parameters() if 'vera_lambda' in n]
    elif adapter_type == "lora":
        expert_params = [p for n, p in model.named_parameters() if 'lora' in n]
    else:
        expert_params = 0

    if not expert_params:
        raise ValueError("No expert parameters found for optimization.")

    optimizer = {"expert": AdamW(expert_params, lr=initial_expert_lr)}
    if number_experts > 1:
        gate_params = [p for n, p in model.named_parameters() if 'gate.' in n] if number_experts > 1 else []
        optimizer["gate"] = AdamW(gate_params, lr=initial_gate_lr)
    return optimizer

def scheduler_set(optimizer, warmup_steps, total_steps, number_experts):
    scheduler = {
        "expert": get_linear_schedule_with_warmup(
            optimizer["expert"],
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    }

    if number_experts > 1:
        scheduler["gate"] = get_linear_schedule_with_warmup(
            optimizer["gate"],
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    return scheduler

def generate_random_near_center(center: float, delta: float = 0.4) -> float:
    """
    生成一个以center为中心、在±delta范围内的随机浮点数。

    :param center: 中心值（float）
    :param delta: 波动范围（可以是绝对值，比如0.1，表示±0.1）
    :return: 随机浮点数
    """
    return random.uniform(center - delta, center + delta)

def load_model(pretrained_model_path: object, adapter_type: str, r: int, dropout: float, target_modules: list, torch_type,
               number_experts: int = 0, top_k: int = 0, d_initial: float = 1):
    """
    Load the model with different adapters (Lora, Vera, MoEVeRA) based on the adapter_type.

    Parameters:
    - pretrained_model_path: str, path to the pretrained model
    - adapter_type: str, 'lora', 'vera', or 'mora' to specify which adapter to use
    - r: int, the rank for LoRA or Vera adapters
    - dropout: float, dropout rate for LoRA or Vera adapters
    - target_modules: list, the target layers/modules to apply the adapter
    - torch_type: torch dtype, the dtype for the model
    - number_experts: int, number of experts (for MoEVeRA only)
    - top_k: int, the top_k value for MoEVeRA (for MoEVeRA only)
    - vera_r: int, the rank for Vera adapter (for Vera only)
    - d_initial: float, initial scaling factor for Vera adapter (for Vera only)

    Returns:
    - tokenizer: the tokenizer
    - model: the loaded model with applied adapter
    """

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        token=hf_token,
        torch_dtype=torch_type,
        low_cpu_mem_usage=True,
        device_map={"": device},  # Explicitly map to the target device
    )

    # Apply adapter based on the type
    if adapter_type == "lora":
        print("Applying LoRA adapter...")
        lora_config = LoraConfig(
            r=r,  # LoRA rank (recommended 256)
            lora_dropout=dropout,
            target_modules=target_modules
        )
        model = get_peft_model(model, lora_config)
        if number_experts > 1:
            print('Replacing LoRA layers with MoELoRA')
            # Replace LoRA layers with MoELoRA and ensure device consistency
            for name, module in model.named_modules():
                if isinstance(module, peft.tuners.lora.layer.Linear):
                    module.__class__ = MoELoRA
                    module.set_moe(number_experts=number_experts, top_k=top_k)
                    for i in range(number_experts):
                        init_lora_A(module.lora_A[i])
                        init_lora_B(module.lora_B[i])
    elif adapter_type == "vera":
        print("Applying VeRA adapter...")
        vera_config = VeraConfig(
            r=r,  # VeRA rank
            vera_dropout=dropout,
            target_modules=target_modules,
            d_initial=d_initial,  # Initial scaling factor for VeRA
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, vera_config)
        if number_experts > 1:
            print('Replacing VeRA layers with MoEVeRA')
            for name, module in model.named_modules():
                if isinstance(module, peft.tuners.vera.layer.Linear):
                    module.__class__ = MoEVeRA
                    module.set_moe(number_experts=number_experts, top_k=top_k)
                    for i in range(number_experts):
                        init_lambda_b(module.vera_lambda_b[i])
                        init_lambda_d(module.vera_lambda_d[i], value=generate_random_near_center(d_initial))  # 你也可以修改 value
    elif adapter_type == "base":
        return tokenizer, model.to(device)
    else:
        raise NotImplementedError(
            "the other base adapters is not implemented yet"
        )

    model.to(device)
    # print("\Parameters participating in training")
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.shape}, {param.device}, requires_grad={param.requires_grad}")
    # print("\nParameters not participating in training")
    # for name, param in model.named_parameters():
    #     if not param.requires_grad:  # 改为检查 requires_grad=False
    #         print(f"{name}: {param.shape}, {param.device}, requires_grad={param.requires_grad}")

    return tokenizer, model
def load_data(MyDataset, data_path, tokenizer, max_length, per_gpu_batch_size):
    train_dataset, val_dataset, test_dataset = MyDataset.load_dataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length
    )

    def custom_collate_fn(batch, device):
        from torch.utils.data import default_collate
        batch = default_collate(batch)
        return {k: v.to(device) for k, v in batch.items()}

    train_loader, val_loader, test_loader = (
        DataLoader(
            dataset_split,
            batch_size=per_gpu_batch_size,
            sampler=RandomSampler(dataset_split),
            collate_fn=lambda b: custom_collate_fn(b, device)
        ) for dataset_split in [train_dataset, val_dataset, test_dataset]
    )

    return train_loader, val_loader, test_loader



def valid(gradient_update_steps, epoch, val_loader, model):
    print('---valid---')
    with torch.no_grad():
        valid_loss = []
        # 在单卡训练时直接使用 tqdm
        tqdm_val_loader = tqdm(val_loader)
        for batch in tqdm_val_loader:
            outputs = model(**batch)
            loss = outputs.loss.item()
            valid_loss.append(loss)
        avg_valid_loss = sum(valid_loss) / len(valid_loss)
        print(f"Steps:{gradient_update_steps}; Epoch:{epoch}; Validation Loss:{avg_valid_loss}\n", end="")
        print('---valid---')


def test(test_loader, model, tokenizer, max_window_length, max_generated_tokens):
    print('---test---')
    global MyDataset
    with torch.no_grad():
        test_outputs, test_inputs, test_labels = [], [], []
        # 在单卡训练时直接使用 tqdm
        tqdm_test_loader = tqdm(test_loader)
        for batch in tqdm_test_loader:
            # Generate the output for each batch
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=max_generated_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
                temperature=None,
                top_p=None
            )

            if outputs.size(1) < max_window_length + max_generated_tokens:
                format_outputs = torch.ones(outputs.size(0), max_window_length + max_generated_tokens,
                                            dtype=outputs.dtype).to(device) * tokenizer.pad_token_id
                format_outputs[:, :outputs.size(1)] = outputs
                test_outputs.append(format_outputs)
            else:
                test_outputs.append(outputs)

            test_inputs.append(batch["input_ids"])
            test_labels.append(batch["labels"])

        outputs = torch.cat(test_outputs, dim=0)
        inputs = torch.cat(test_inputs, dim=0)
        labels = torch.cat(test_labels, dim=0)

        # Evaluate the model
        metrics = MyDataset.evaluate(inputs=inputs, labels=labels, preds=outputs, tokenizer=tokenizer)
        print(f"Accuracy:{metrics}\n", end="")
    print('---test---')
    return {"Accuracy": metrics}


import wandb


def train(train_loader, val_loader, test_loader, model, batch_size, accumulated_steps, optimizer, scheduler,
          max_gate_grad_norm, total_gradient_update_steps, eval_steps, generate_param,
          number_experts, top_k, r, contra_rate, early_stop_patience=100):

    print('\n---training---\n')
    wandb.init(
        entity="psget1t-national-university-of-singapore-students-union",
        project="gui",
        name=f'n:{number_experts}_k:{top_k}_r:{r}_c{contra_rate}',
        config={
            "batch_size": batch_size,
            "accumulated_steps": accumulated_steps,
            "eval_steps": eval_steps,
            "max_gate_grad_norm": max_gate_grad_norm,
            "number_experts": number_experts,
            "early_stop_patience": early_stop_patience
        })

    gradient_update_steps = 0
    total_steps = 0
    max_epoch = int((total_gradient_update_steps * accumulated_steps) / len(train_loader)) + 1

    tokenizer = generate_param["tokenizer"]
    max_window_length = generate_param["max_window_length"]
    max_generated_tokens = generate_param["max_generated_tokens"]

    model.train()
    if number_experts > 1:
        optimizer["expert"].zero_grad()
        optimizer["gate"].zero_grad()
    else:
        optimizer["expert"].zero_grad()

    loss_item = 0.0
    start_time = datetime.now()
    best_train_loss = float("inf")
    last_improvement_step = 0

    for epoch in range(max_epoch):
        print(f"\nEpoch {epoch + 1}/{max_epoch}")
        print(f"Memory Usage - Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        print(f"Memory Usage - Cached: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        for batch_idx, batch in enumerate(train_loader):
            if gradient_update_steps >= total_gradient_update_steps:
                break

            outputs = model(**batch)
            if number_experts > 1 and contra_rate > 0:
                contrast_losses = [m.contra_loss for m in model.modules() if isinstance(m, MoEVeRA)]
                L_les = [m.L_l for m in model.modules() if isinstance(m, MoEVeRA)]
                total_contrast = sum(contrast_losses)
                total_L_l = sum(L_les)
                loss = ((1 - 3*contra_rate) * outputs.loss + contra_rate * total_contrast) / accumulated_steps
                print(f"main_loss: {outputs.loss:.4f}")
                print(f"contrast_loss: {total_contrast:.4f}")
                print(f"loss: {loss:.4f}")
            else:
                loss = outputs.loss / accumulated_steps

            loss.backward()
            loss_item += loss.item()
            total_steps += 1

            if total_steps % accumulated_steps == 0:
                if number_experts > 1:
                    torch.nn.utils.clip_grad_norm_(
                        [param for name, param in model.named_parameters() if 'gate.' in name],
                        max_norm=max_gate_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        [param for name, param in model.named_parameters() if 'expert' in name],
                        max_norm=max_gate_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if number_experts > 1:
                    optimizer["expert"].step()
                    optimizer["gate"].step()
                    scheduler["expert"].step()
                    scheduler["gate"].step()
                else:
                    optimizer["expert"].step()
                    scheduler["expert"].step()

                if number_experts > 1:
                    optimizer["expert"].zero_grad()
                    optimizer["gate"].zero_grad()
                else:
                    optimizer["expert"].zero_grad()

                gradient_update_steps += 1
                time_elapsed = str(datetime.now() - start_time).split(".")[0]

                if number_experts > 1:
                    lr_expert = scheduler["expert"].get_last_lr()[0]
                    lr_gate = scheduler["gate"].get_last_lr()[0]
                    lr_info = f"LR Expert: {lr_expert:.2e} | LR Gate: {lr_gate:.2e}"
                else:
                    lr = scheduler["expert"].get_last_lr()[0]
                    lr_info = f"LR: {lr:.2e}"

                print(
                    f"[{time_elapsed}] Step: {gradient_update_steps:4d}/{total_gradient_update_steps} | "
                    f"Loss: {loss_item:.4f} | {lr_info}"
                )

                log_data = {
                    "train/loss": loss_item,
                    "train/step": gradient_update_steps,
                    "train/time_elapsed": float((datetime.now() - start_time).total_seconds())
                }
                if number_experts > 1:
                    log_data["lr/expert"] = lr_expert
                    log_data["lr/gate"] = lr_gate
                    if contra_rate > 0:
                        log_data["loss/contrastive"] = total_contrast.item()
                        log_data["loss/L_l"] = total_L_l.item()
                        log_data["loss/main"] = outputs.loss.item()
                else:
                    log_data["lr"] = lr

                # 记录每位专家使用次数（叠加所有 module 中的 use_expert）
                if number_experts > 1:
                    expert_total_use = None
                    for m in model.modules():
                        if isinstance(m, MoEVeRA):
                            if expert_total_use is None:
                                expert_total_use = m.use_expert.clone()
                            else:
                                expert_total_use += m.use_expert
                    for eid, count in enumerate(expert_total_use.tolist()):
                        log_data[f"expert_usage/eid_{eid}"] = count

                wandb.log(log_data)

                if loss_item < best_train_loss:
                    best_train_loss = loss_item
                    last_improvement_step = gradient_update_steps
                elif gradient_update_steps - last_improvement_step >= early_stop_patience:
                    print(f"\nEarly stopping at step {gradient_update_steps}. No improvement in {early_stop_patience} steps.")
                    training_time = (datetime.now() - start_time).total_seconds()
                    model.eval()
                    test_metrics = test(test_loader, model, tokenizer, max_window_length, max_generated_tokens)
                    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
                    wandb.log({"training_time": training_time})
                    wandb.finish()
                    return test_metrics, training_time, model

                loss_item = 0.0

                if gradient_update_steps % eval_steps == 0:
                    model.eval()
                    valid(gradient_update_steps, epoch, val_loader, model)
                    model.train()

    training_time = (datetime.now() - start_time).total_seconds()
    model.eval()
    test_metrics = test(test_loader, model, tokenizer, max_window_length, max_generated_tokens)
    wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
    wandb.log({"training_time": training_time})
    wandb.finish()

    return test_metrics, training_time, model





def main(candidate_configs, DATASET_NAME, MyDataset, DATASET_PATH):
    print(f"---start at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}---")
    print(candidate_configs)

    max_window_length = candidate_configs["max_window_length"]  # 128/256/...
    max_generated_tokens = candidate_configs["max_generated_tokens"]  # 10/...
    initial_expert_learning_rate = candidate_configs["initial_expert_lr"]
    initial_gate_learning_rate = candidate_configs["initial_gate_lr"]
    warmup_steps = candidate_configs["warm_up"]
    total_gradient_update_steps = candidate_configs["total_gradient_update_steps"]
    per_gpu_batch_size = candidate_configs["per_gpu_batch_size"]
    accumulated_steps = candidate_configs["accumulated_steps"]
    scheduler_max_steps = candidate_configs["scheduler_max_steps"]
    eval_steps = candidate_configs["eval_steps"]
    max_gate_grad_norm = candidate_configs["max_gate_grad_norm"]
    d_initial = candidate_configs["d_initial"]
    method = candidate_configs["method"]
    target_modules = candidate_configs["target_modules"]
    dropout = candidate_configs["dropout"]
    number_experts = candidate_configs["number_expert"]
    top_k = candidate_configs["top_k"]
    r = candidate_configs["expert_rank"]
    contra_rate = candidate_configs["contra_rate"]

    print(model_name)
    tokenizer, model = load_model(
        pretrained_model_path=model_name,
        adapter_type=method,
        r=r,
        d_initial=d_initial,
        dropout=dropout,
        target_modules=target_modules,
        number_experts=number_experts,
        top_k=top_k,
        torch_type=torch.float16,
    )

    print(f"当前显存占用 in train: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
    print(f"当前显存缓存 in train: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

    train_loader, val_loader, test_loader = load_data(
        MyDataset=MyDataset,
        data_path=DATASET_PATH,
        tokenizer=tokenizer,
        max_length=max_window_length,
        per_gpu_batch_size=per_gpu_batch_size,
    )

    if r == 0:
        model.eval()
        test_metrics = test(test_loader, model, tokenizer, max_window_length, max_generated_tokens)
        params_trainable = get_parameters_count(model, requires_grad=True)
        params_total = get_parameters_count(model, requires_grad=False)
        print(f"Trainable parameters: {params_trainable}")
        print(f"Total number of parameters: {params_total}")
        # 收集训练参数
        training_params = {
            'pretrained_model_path': model_name,
            'adapter_type': 'base',
            'dataset': DATASET_NAME,
            'r': r,
            'initial_expert_lr': 0,
            'initial_gate_lr': 0,
            'd_initial': 0,
            'dropout': 0,
            'target_modules': 0,
            'number_experts': 0,
            'top_k': 0,
            'batch_size': per_gpu_batch_size,
            'accumulated_steps': 0,
            'max_gate_grad_norm': 0,
            'total_gradient_update_steps': 0,
        }

        # 指定 CSV 文件路径
        csv_file_path = 'final_results.csv'

        # 保存到 CSV
        res = save_training_results(csv_file_path, training_params, test_metrics, 0, params_trainable, params_total)
        res['save_path'] = 'None'
        return res

    optimizer = optimizer_set(model, initial_expert_learning_rate, initial_gate_learning_rate, number_experts, method)
    scheduler = scheduler_set(optimizer, warmup_steps, scheduler_max_steps, number_experts)

    print('start training...')
    test_metrics, training_time, model = train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        model=model,
        batch_size=per_gpu_batch_size,
        accumulated_steps=accumulated_steps,
        optimizer=optimizer,
        scheduler=scheduler,
        max_gate_grad_norm=max_gate_grad_norm,
        total_gradient_update_steps=total_gradient_update_steps,
        eval_steps=eval_steps,
        generate_param={
            "tokenizer": tokenizer,
            "max_window_length": max_window_length,
            "max_generated_tokens": max_generated_tokens
        },
        number_experts=number_experts,
        top_k=top_k,
        r=r,
        contra_rate=contra_rate,
        early_stop_patience=100
    )
    params_trainable = get_parameters_count(model, requires_grad=True)
    params_total = get_parameters_count(model, requires_grad=False)
    print(f"Trainable parameters: {params_trainable}")
    print(f"Total number of parameters: {params_total}")
    print(f"Time cost in training: {training_time}")
    if method == 'vera':
        if number_experts > 1:
            adapter_type = 'mvra'
        else:
            adapter_type = method
            initial_gate_learning_rate = 0
    elif method == 'lora':
        d_initial = 0
        if number_experts > 1:
            adapter_type = 'mora'
        else:
            adapter_type = method
            initial_gate_learning_rate = 0
    else:
        print('not supported method')
    save_path = "./saved_lora/" + "_".join([
        'llama3.2-3b',
        adapter_type,
        DATASET_NAME,
        str(r),
        str(dropout),
        str(number_experts),
        str(top_k),
    ])
    os.makedirs(save_path, exist_ok=True)


    # 收集训练参数
    training_params = {
        'pretrained_model_path': model_name,
        'adapter_type': adapter_type,
        'dataset': DATASET_NAME,
        'r': r,
        'contra_rate': contra_rate,
        'initial_expert_lr': initial_expert_learning_rate,
        'initial_gate_lr': initial_gate_learning_rate,
        'd_initial': d_initial,
        'dropout': dropout,
        'target_modules': target_modules,
        'number_experts': number_experts,
        'top_k': top_k,
        'batch_size': per_gpu_batch_size,
        'accumulated_steps': accumulated_steps,
        'max_gate_grad_norm': max_gate_grad_norm,
        'total_gradient_update_steps': total_gradient_update_steps,

    }

    # 指定 CSV 文件路径
    csv_file_path = candidate_configs['res_path']

    # 保存到 CSV
    res =  save_training_results(csv_file_path, training_params, test_metrics, training_time, params_trainable, params_total)

    print("Saving the trained model...")
    torch.save(model.state_dict(), os.path.join(save_path, "trained_model.pth"))
    res['save_path'] = os.path.join(save_path, "trained_model.pth")

    print('\n---done---\n')
    return res


if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Configure training hyperparameters.")

    # General hyperparameters
    parser.add_argument("--number_expert", type=int, default=0, help="Number of experts.")
    parser.add_argument("--rank", type=int, required=True, help="Rank of each expert.")
    parser.add_argument("--top_k", type=int, default=0, help="Number of top experts to select.")
    parser.add_argument("--initial_expert_lr", type=float, default=5e-5, help="Initial learning rate for experts.")
    parser.add_argument("--initial_gate_lr", type=float, default=3e-6, help="Initial learning rate for gate.")
    parser.add_argument("--warm_up", type=int, default=100, help="Warm-up steps.")
    parser.add_argument("--total_gradient_update_steps", type=int, default=1000, help="Total gradient update steps.")
    parser.add_argument("--eval_steps", type=int, default=400, help="Evaluation steps.")
    parser.add_argument("--max_gate_grad_norm", type=float, default=1.0, help="Maximum gate gradient norm.")
    parser.add_argument("--dropout", type=float, default=0.05, help="dropout probability.")
    parser.add_argument("--d", type=float, default=1.0, help="initial d")

    parser.add_argument("--dataset_name", type=str, default='ScienceQA', help="Dataset Name")
    parser.add_argument("--method", type=str, default='lora', help="way to finetune")
    parser.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj",
                        help="Comma-separated target modules.")
    parser.add_argument("--res_path", type=str, default='results/final_results.csv', help="path to save results")
    parser.add_argument("--contra_rate", type=float, default=0, help="contrastive learning")

    # Batch size and accumulated steps
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size")

    parser.add_argument("--accumulated_steps", type=int, default=2, help="Accumulated steps ")


    # Scheduler steps
    parser.add_argument("--scheduler_max_steps", type=int, default=int(1000 / 9 * 10),
                        help="Max steps for the scheduler.")

    # 解析命令行参数
    args = parser.parse_args()
    DATASET_NAME = args.dataset_name
    MyDataset = CANDIDATE_DATASETS[DATASET_NAME][0]
    DATASET_PATH = CANDIDATE_DATASETS[DATASET_NAME][1]
    if args.dataset_name not in CANDIDATE_DATASETS.keys():
        data_flag = 1
    else:
        data_flag = 0

    # 将参数配置到字典中
    candidate_configs = {
        "method": args.method,
        "target_modules": args.target_modules.split(","),
        "number_expert": args.number_expert,
        "expert_rank": args.rank,
        "top_k": args.top_k,
        "d_initial": args.d,
        "initial_expert_lr": args.initial_expert_lr,
        "initial_gate_lr": args.initial_gate_lr,
        "warm_up": args.warm_up,
        "total_gradient_update_steps": args.total_gradient_update_steps,
        "eval_steps": args.eval_steps,
        "dropout": args.dropout,
        "per_gpu_batch_size": args.batch_size,
        "accumulated_steps": args.accumulated_steps,
        "scheduler_max_steps": args.scheduler_max_steps,
        "max_gate_grad_norm": args.max_gate_grad_norm,
        "res_path" : args.res_path,
        "contra_rate": args.contra_rate,
        'max_window_length': MyDataset.MAX_SAMPLE_INPUT_LENGTH,  # 128/256/...
        'max_generated_tokens': MyDataset.MAX_SAMPLE_OUTPUT_LENGTH  # 10/...
    }

    main(candidate_configs, DATASET_NAME, MyDataset, DATASET_PATH)
