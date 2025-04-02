import os
import sys
import peft
import torch
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import get_linear_schedule_with_warmup, BitsAndBytesConfig
from transformers import AutoProcessor, LlavaForConditionalGeneration

from moe_lora import MoELoRA
from utils import plot_gradients, plot_values
from custom_optimizer import AdamW, SGD, AdamWr, SGDr
from utils import init_lora_A, init_lora_B, init_gate, wrap_print_function

from dataset import CoLADataset
from dataset import ScienceQADataset
from dataset import BoolQDataset
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
model_name = "meta-llama/Llama-3.2-3B-Instruct"
hf_token = "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"
MyDataset = None
DATASET_PATH = None
DATASET_NAME = None
CANDIDATE_DATASETS = {
    "ScienceQA":        [ScienceQADataset,      "./dataset/scienceqa/science_qa.hf"],
    "CommonsenseQA":    [CommonsenseQADataset,  "./dataset/CommonsenseQA"],

    "CoLA":             [CoLADataset,           "./dataset/glue_data/CoLA"],
    "SST-2":            [SST2Dataset,           "./dataset/glue_data/SST-2"],
    "MRPC":             [MRPCDataset,           "./dataset/glue_data/MRPC"],
    "QQP":              [QQPDataset,            "./dataset/glue_data/QQP"],
    "QNLI":             [QNLIDataset,           "./dataset/glue_data/QNLI"],
    "STS-B":            [STSBDataset,           "./dataset/glue_data/STS-B"],
    "WNLI":             [WNLIDataset,           "./dataset/glue_data/WNLI"],

    "MixedQA":          [MixedQADataset,        "./datasets"],

    "ImageText2Text":   [ImageText2TextDataset, "./dataset/visual7w"],
    "VMCBench":         [VMCBenchDataset,       "./dataset/VMCBench/data"]
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print_log = None

def load_model(pretrained_model_path, lora_r, lora_alpha, lora_dropout, lora_modules, number_experts, top_k, torch_type, descent_strategy):

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side='left'

    #processor = AutoProcessor.from_pretrained(pretrained_model_path)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit = True,                        # 或者 load_in_8bit=True，根据需要设置
        #llm_int8_threshold = 6.0,
        #llm_int8_has_fp16_weight = False,
        llm_int8_enable_fp32_cpu_offload = True,
        bnb_4bit_compute_dtype = torch.float16,     # 计算时使用的精度
        bnb_4bit_quant_type = "nf4",                # 采用nf4量化法，默认为fp4
        bnb_4bit_use_double_quant = True,           # 是否采用双量化
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_path,
        token=hf_token,
        torch_dtype = torch_type,
        low_cpu_mem_usage = True,
        quantization_config = quantization_config
    )
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r = lora_r,
        lora_alpha = lora_alpha,
        target_modules = lora_modules,
        lora_dropout = lora_dropout,
        bias = "none",
        task_type = "CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    for name, module in model.named_modules():
        if isinstance(module, peft.tuners.lora.LoraLayer):
            module.__class__ = MoELoRA
            module.set_moe(number_experts = number_experts, top_k = top_k)
            module.descent_strategy = descent_strategy
            for i in range(number_experts):
                init_lora_A(module.lora_A[i])
                init_lora_B(module.lora_B[i])
                init_gate(module.gate)
            #print_log(name)
    global AdamWr, SGDr
    AdamWr.number_experts = number_experts
    SGDr.number_experts = number_experts
    return tokenizer, model

# def distribution_initialize():
#     local_rank = os.environ['LOCAL_RANK']
#     torch.distributed.init_process_group(backend = 'nccl')
#     torch.distributed.barrier()
#     world_size = torch.distributed.get_world_size()
#     rank = torch.distributed.get_rank()
#     device = torch.device("cuda", rank)
#     return world_size, local_rank, device

from tqdm import tqdm
import torch

def valid(gradient_update_steps, epoch, val_loader, model):
    with torch.no_grad():
        valid_loss = []
        # 在单卡训练时直接使用 tqdm
        tqdm_val_loader = tqdm(val_loader)
        for batch in tqdm_val_loader:
            outputs = model(**batch)
            loss = outputs.loss.item()
            valid_loss.append(loss)
        avg_valid_loss = sum(valid_loss)/len(valid_loss)
        print('---Validation---')
        print(f"Steps:{gradient_update_steps}; Epoch:{epoch}; Validation Loss:{avg_valid_loss}\n", end="")
        print('---Validation---')

def test(gradient_update_steps, epoch, test_loader, model, tokenizer, device, max_window_length, max_generated_tokens):
    global MyDataset
    with torch.no_grad():
        test_outputs, test_inputs, test_labels = [], [], []
        # 在单卡训练时直接使用 tqdm
        tqdm_test_loader = tqdm(test_loader)
        for batch in tqdm_test_loader:
            outputs = model.generate(
                input_ids = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                #pixel_values = batch["pixel_values"].to(device),
                max_new_tokens = max_generated_tokens,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id,
                do_sample = False,
                temperature = None,
                top_p = None
            )
            if outputs.size(1) < max_window_length + max_generated_tokens:
                format_outputs = torch.ones(outputs.size(0), max_window_length + max_generated_tokens, dtype = outputs.dtype).to(device) * tokenizer.pad_token_id
                format_outputs[:, :outputs.size(1)] = outputs
                test_outputs.append(format_outputs)
            else:
                test_outputs.append(outputs)
            test_inputs.append(batch["input_ids"])
            test_labels.append(batch["labels"])
        outputs = torch.cat(test_outputs, dim = 0)
        inputs = torch.cat(test_inputs, dim = 0)
        labels = torch.cat(test_labels, dim = 0)
        metrics = MyDataset.evaluate(inputs = inputs, labels = labels, preds = outputs, tokenizer = tokenizer)
        print('---Test---')
        print(f"Steps:{gradient_update_steps}; Epoch:{epoch}; Accuracy:{metrics}\n", end="")
        print('---Test---')

from datetime import datetime

def train(train_loader, val_loader, test_loader, model, per_gpu_batch_size, accumulated_steps, optimizer, scheduler, max_gate_grad_norm, total_gradient_update_steps, eval_steps, generate_param):
    gradient_update_steps = 0
    total_steps = 0
    max_epoch = int((total_gradient_update_steps * accumulated_steps) / len(train_loader)) + 1
    tokenizer = generate_param["tokenizer"]
    device = generate_param["device"]
    max_window_length = generate_param["max_window_length"]
    max_generated_tokens = generate_param["max_generated_tokens"]
    num_bit = len(str(per_gpu_batch_size * len(train_loader)))
    fmt = '%'+str(num_bit)+'d'
    model.train()
    optimizer["expert"].zero_grad()
    optimizer["gate"].zero_grad()
    loss_item = 0.0
    start_time = datetime.now()
    print(f"total_gradient_update_steps:{total_gradient_update_steps}")
    print(f"accumulated_steps:{accumulated_steps}")
    print(f"len(train_loader):{len(train_loader)}")

    for epoch in range(max_epoch):
        #tqdm_train_loader = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epoch}")
        #for batch in tqdm_train_loader:
    # 训练逻辑

    # for epoch in range(max_epoch):
        for batch in train_loader:
            if gradient_update_steps == total_gradient_update_steps:
                test(gradient_update_steps, 999, test_loader, model, tokenizer, device, max_window_length,
                     max_generated_tokens)
                return model

            outputs = model(**batch)
            print(f"当前显存占用 in train: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
            print(f"当前显存缓存 in train: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")
            loss = outputs.loss
            loss = loss / accumulated_steps
            loss.backward()
            loss_item = loss_item + loss.item()
            total_steps += 1

            if total_steps % accumulated_steps == 0:
                if max_gate_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_([param for name, param in model.named_parameters() if name.find('gate.') != -1], max_norm=max_gate_grad_norm)
                optimizer["expert"].step()
                optimizer["gate"].step()
                scheduler["expert"].step()
                scheduler["gate"].step()
                optimizer["expert"].zero_grad()
                optimizer["gate"].zero_grad()
                gradient_update_steps += 1

                time_str = str(datetime.now() - start_time).split(".")[0]
                print(f"[{time_str}] Step:{'%4d'%gradient_update_steps}/{total_gradient_update_steps}; Epoch:{'%2d'%epoch}({fmt%(per_gpu_batch_size*(total_steps % len(train_loader)))}/{per_gpu_batch_size*len(train_loader)}); Loss:{loss_item};\tLR:{[g['lr'] for part in optimizer for g in optimizer[part].param_groups]}\n", end = "")
                loss_item = 0.0

                if gradient_update_steps % eval_steps == 0:
                    model.eval()
                    valid(gradient_update_steps, epoch, val_loader, model)
                    #test(gradient_update_steps, epoch, test_loader, model, tokenizer, device, max_window_length, max_generated_tokens)
                    model.train()
                    start_time = datetime.now()
    # test(gradient_update_steps, 999, test_loader, model, tokenizer, device, max_window_length, max_generated_tokens)



def main(optimize_strategy, descent_strategy):

    # distribution configuration
    #world_size, local_rank, device = distribution_initialize()
    global MyDataset
    global DATASET_NAME
    #global print_log

    save_path = "./saved_lora/" + "_".join([
        'baseLoRA',
        DATASET_NAME,
        optimize_strategy,
        descent_strategy,
        datetime.now().strftime("%y%m%d.%H.%M")
    ])
    os.makedirs(save_path, exist_ok=True)
    #torch_rank = torch.distributed.get_rank()
    #print_log = wrap_print_function(file_path = save_path + "/log."+str(torch_rank))
    #print_log(f">> world size:{world_size}; torch rank:{torch_rank}; OS local rank:{local_rank}; device:{device};\n", end = "")

    candidate_configs = {

        # "sgd": {
        #     "number_expert": 20,        # 18
        #     "expert_rank": 4,           # 4
        #     "top_k": 10,                # 9
        #     "initial_expert_lr": 3e-5,  #3e-5,
        #     "initial_gate_lr": 3e-8,
        #     "warm_up": 0,
        #     "total_gradient_update_steps": 2000,    # 500   #"training_epochs": 20,
        #     "eval_steps": 100,                      # 100
        #     "per_gpu_batch_size": {                 # SGD takes less memory, so we use large batch.
        #         512+10: 10,
        #         256+10: 20,                         # For those dataset max_window_length=256 & max_generated_token=10 & 3090GPU
        #         128+10: 40,                         # For those dataset max_window_length=128 & max_generated_token=10 & 3090GPU
        #         64+10:  80,                          # For those dataset max_window_length=64 & max_generated_token=10 & 3090GPU
        #
        #         140+10: 5,
        #         400+10: 2
        #     },
        #     "accumulated_steps": {                  # total_steps = acculumated_steps * total_gradient_update_steps
        #         512+10: 8,
        #         256+10: 4,
        #         128+10: 2,
        #         64+10:  1,
        #
        #         140+10: 8,
        #         400+10: 2
        #     },                                      # logic batch size = 80
        #     "scheduler_max_steps": int(2000/9*10),  # so that the learning rate will decrease to 0.1*Initial_LR
        #     "max_gate_grad_norm": 1.0
        # },
        "sgd": {
            "number_expert": 8,
            "expert_rank": 4,
            "top_k": 4,  # 10 -> 8
            "initial_expert_lr": 5e-5,  # 3e-5 -> 5e-5
            "initial_gate_lr": 3e-7,  # 3e-8 -> 3e-7
            "warm_up": 0,
            "total_gradient_update_steps": 64,  # 2000 -> 1000
            "eval_steps": 64,

            "per_gpu_batch_size": {
                512 + 10: 32,  # 10 -> 20
                256 + 10: 72,  # 20 -> 40
                128 + 10: 128,  # 40 -> 80
                64 + 10: 256,  # 80 -> 160
                140 + 10: 16,  # 5 -> 10
                400 + 10: 8  # 2 -> 4
            },

            "accumulated_steps": {
                512 + 10: 4,  # 8 -> 4
                256 + 10: 2,  # 4 -> 2
                128 + 10: 1,  # 2 -> 1
                64 + 10: 1,
                140 + 10: 4,  # 8 -> 4
                400 + 10: 1  # 2 -> 1
            },

            "scheduler_max_steps": int(1000 / 9 * 10),  # 2000 -> 1000
            "max_gate_grad_norm": 1.0
        },


        "adamw": {
            "number_expert": 20,        #50         # For AdamW, we only managed improves when training lots of experts.
            "expert_rank": 4,           #4
            "top_k": 10,                #25
            "initial_expert_lr": 1e-5,  #3e-5,
            "initial_gate_lr": 3e-8,
            "warm_up": 0,
            "total_gradient_update_steps": 2000,    # 250   #"training_epochs": 5,
            "eval_steps": 100,                      # 50
            "per_gpu_batch_size": {                 # AdamW takes more memory, so we use small batch.
                512+10: 5,
                256+10: 10,                          # For those dataset max_window_length=256 & max_generated_token=10 & 3090GPU
                128+10: 20,                         # For those dataset max_window_length=128 & max_generated_token=10 & 3090GPU
                64+10:  40,                          # For those dataset max_window_length=64 & max_generated_token=10 & 3090GPU

                140+10: 2,
                400+10: 1
            },
            "accumulated_steps": {                  # total_steps = acculumated_steps * total_gradient_update_steps
                512+10: 16,
                256+10: 8,
                128+10: 4,
                64+10:  2,

                140+10: 20,
                400+10: 4
            },                                      # logic batch size = 80
            "scheduler_max_steps": int(2000/9*10),  # so that the learning rate will decrease to 0.1*Initial_LR
            "max_gate_grad_norm": 1.0
        }

    }
    # model configuration
    tokenizer, model = load_model(
        pretrained_model_path = model_name,
        lora_r = candidate_configs[optimize_strategy]["expert_rank"],
        lora_alpha = 16,
        lora_dropout = 0.05,
        lora_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        number_experts = candidate_configs[optimize_strategy]["number_expert"],
        top_k = candidate_configs[optimize_strategy]["top_k"],
        torch_type = torch.float16,
        descent_strategy = descent_strategy
    )

    model = model.to(device)


    # training configuration
    max_window_length = MyDataset.MAX_SAMPLE_INPUT_LENGTH           # 128/256/...
    max_generated_tokens = MyDataset.MAX_SAMPLE_OUTPUT_LENGTH       # 10/...
    test_before_training = True                                    # True/False
    eval_before_training = True                                    # True/False
    initial_expert_learning_rate = candidate_configs[optimize_strategy]["initial_expert_lr"]
    initial_gate_learning_rate = candidate_configs[optimize_strategy]["initial_gate_lr"]                                  
    warmup_steps = candidate_configs[optimize_strategy]["warm_up"]
    total_gradient_update_steps = candidate_configs[optimize_strategy]["total_gradient_update_steps"]
    per_gpu_batch_size = candidate_configs[optimize_strategy]["per_gpu_batch_size"][max_window_length+max_generated_tokens]
    accumulated_steps = candidate_configs[optimize_strategy]["accumulated_steps"][max_window_length+max_generated_tokens]
    scheduler_max_steps = candidate_configs[optimize_strategy]["scheduler_max_steps"]
    eval_steps = candidate_configs[optimize_strategy]["eval_steps"]
    max_gate_grad_norm = candidate_configs[optimize_strategy]["max_gate_grad_norm"]

    #world_batch_size = per_gpu_batch_size * world_size

    # dataset configuration
    global DATASET_PATH
    train_dataset, val_dataset, test_dataset = MyDataset.load_dataset(
        data_path = DATASET_PATH,
        tokenizer = tokenizer,
        max_length = max_window_length
    )
    from torch.utils.data import DataLoader, RandomSampler

    # 创建数据加载器，单卡训练时不使用分布式采样器
    train_loader, val_loader, test_loader = (
        DataLoader(
            dataset_split,
            batch_size=per_gpu_batch_size,
            sampler=RandomSampler(dataset_split)  # 使用普通的随机采样器
        ) for dataset_split in [train_dataset, val_dataset, test_dataset]
    )

    # optimizer configuration
    expert_parameters = [param for name, param in model.named_parameters() if name.find('lora_') != -1]
    gate_parameters = [param for name, param in model.named_parameters() if name.find('gate.') != -1]
    optimizer_expert = ({
        "adamw|riemannian": AdamWr,
        "sgd|riemannian": SGDr,
        "adamw|moe-riemannian": AdamWr,
        "sgd|moe-riemannian": SGDr,
    }[f"{optimize_strategy}|{descent_strategy}"])(expert_parameters, lr = initial_expert_learning_rate)
    
    optimizer_gate = ({"sgd": SGD, "adamw": AdamW})[optimize_strategy](gate_parameters, lr = initial_gate_learning_rate)
    #optimizer_gate = SGD(gate_parameters, lr = initial_gate_learning_rate)
    
    scheduler_expert = get_linear_schedule_with_warmup(
        optimizer_expert, 
        num_warmup_steps = warmup_steps, 
        num_training_steps = scheduler_max_steps
    )
    scheduler_gate = get_linear_schedule_with_warmup(
        optimizer_gate, 
        num_warmup_steps = warmup_steps, 
        num_training_steps = scheduler_max_steps
    )
    optimizer = {"gate": optimizer_gate, "expert": optimizer_expert}
    scheduler = {"gate": scheduler_gate, "expert": scheduler_expert}

    # model.eval()
    # if eval_before_training:
    #     print('\n---eval_before_training---\n')
    #     valid(0, -1, val_loader, model)
    #     #torch.distributed.barrier()
    # if test_before_training:
    #     print('\n---test_before_training---\n')
    #     test(0, -1, test_loader, model, tokenizer, device, max_window_length, max_generated_tokens)
        #torch.distributed.barrier()

    #_ = input("Wait to stop ...")
    #exit()
    print('\n---training---\n')

    train(
        train_loader = train_loader, 
        val_loader = val_loader, 
        test_loader = test_loader, 
        model = model, 
        per_gpu_batch_size = per_gpu_batch_size,
        accumulated_steps = accumulated_steps, 
        optimizer = optimizer, 
        scheduler = scheduler,
        max_gate_grad_norm = max_gate_grad_norm, 
        total_gradient_update_steps = total_gradient_update_steps, 
        eval_steps = eval_steps,
        generate_param = {
            "tokenizer": tokenizer, 
            "device": device, 
            "max_window_length": max_window_length, 
            "max_generated_tokens": max_generated_tokens
        }
    )
    print("Saving the trained model...")
    torch.save(model.state_dict(), os.path.join(save_path, "trained_model.pth"))
    print('\n---done---\n')


if __name__ == "__main__":
    print(len(sys.argv))
    print(sys.argv)
    assert len(sys.argv) == 4

    assert sys.argv[1] in CANDIDATE_DATASETS
    DATASET_NAME = sys.argv[1]
    MyDataset = CANDIDATE_DATASETS[DATASET_NAME][0]
    DATASET_PATH = CANDIDATE_DATASETS[DATASET_NAME][1]

    assert sys.argv[2] in {"sgd", "adamw"}
    optimize_strategy = sys.argv[2]

    assert sys.argv[3] in {"riemannian", "ourmethod"}
    descent_strategy = sys.argv[3] if (sys.argv[3] != "ourmethod") else "moe-riemannian"

    main(optimize_strategy, descent_strategy)
