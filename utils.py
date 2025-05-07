import torch
import numpy as np

def plot_values(model):
    print("Model Weights:")
    names, values = [], []
    for name, param in model.named_parameters():
        names.append(name)
        values.append(param.detach().cpu().numpy() if param is not None else [0])
    for i in range(len(names)):
        min_value = np.min(values[i])
        max_value = np.max(values[i])
        if np.isnan(max_value) or np.isnan(min_value):
            print(f"Rank:{torch.distributed.get_rank()}; {names[i]}; Values has NaN.")

def plot_gradients(model):
    print("Model Gradients:")
    names, grads = [], []
    for name, param in model.named_parameters():
        names.append(name)
        grads.append(param.grad.detach().cpu().numpy() if param.grad is not None else [0])
    for i in range(len(names)):
        min_value = np.min(grads[i])
        max_value = np.max(grads[i])
        if np.isnan(max_value) or np.isnan(min_value):
            print(f"Rank:{torch.distributed.get_rank()}; {names[i]}; Grads has NaN.")

def init_lora_A(lora_A_module):
    torch.nn.init.kaiming_uniform_(lora_A_module.weight, a=5**0.5)

def init_lora_B(lora_B_module):
    torch.nn.init.zeros_(lora_B_module.weight)

def init_lambda_b(lambda_b_tensor: torch.Tensor) -> torch.Tensor:
    # 初始化为 0，或者可以考虑 small normal 初始化
    return torch.zeros_like(lambda_b_tensor)

def init_lambda_d(lambda_d_tensor: torch.Tensor, value: float) -> torch.Tensor:
    return torch.full_like(lambda_d_tensor, fill_value=value)
def init_lambda_d_random(lambda_d_tensor: torch.Tensor, value: float, std: float = 0.05) -> torch.Tensor:
    """
    以 value 为均值、std 为标准差初始化 lambda_d_tensor 的值，构成高斯分布。

    Args:
        lambda_d_tensor (torch.Tensor): 目标张量，用于确定形状和设备
        value (float): 高斯分布均值（d_init）
        std (float): 高斯分布标准差（默认为 0.05）

    Returns:
        torch.Tensor: 初始化后的张量
    """
    return torch.normal(mean=value, std=std, size=lambda_d_tensor.shape, device=lambda_d_tensor.device, dtype=lambda_d_tensor.dtype)

#

# def init_lora_A(lora_A_module, mean=0.0, std=0.02):
#     """ 初始化 lora_A 参数，使其符合高斯分布 """
#     torch.nn.init.normal_(lora_A_module.weight, mean=mean, std=std)
#
# def init_lora_B(lora_B_module, mean=0.0, std=0.02):
#     """ 初始化 lora_B 参数，使其符合高斯分布 """
#     torch.nn.init.normal_(lora_B_module.weight, mean=mean, std=std)


def init_scaling_b(scaling_b_list):
    """ 初始化 scaling_b 使其所有元素为 1 """
    for param in scaling_b_list:
        torch.nn.init.ones_(param)

def init_scaling_d(scaling_d_list):
    """ 初始化 scaling_d 使其所有元素为 0 """
    for param in scaling_d_list:
        torch.nn.init.zeros_(param)


def init_gate(gate_module):
    # no need to init gate
    pass

def wrap_print_function(file_path: str):
    f = open(file_path, "w+")
    def print_log(log, end="\n"):
        f.write(str(log) + end)
        f.flush()
        print(log, end = end)
    return print_log