import copy
import peft
import torch
from typing import Any
import torch.nn.functional as F

class MoEVeRA(peft.tuners.vera.layer.Linear):
    def set_moe(self, number_experts: int, top_k: int):
        """
        设置 MoE-VeRA 模型中专家数量和参与前向计算的专家数（top_k）。
        初始化每个专家专有的参数：
          - vera_lambda_b：专家特有的线性变换乘性因子
          - vera_lambda_d：专家特有的缩放乘性因子
        """
        assert top_k <= number_experts, "top_k 必须不大于专家总数"
        self.number_experts = number_experts
        self.top_k = top_k
        self.expert_loss = 0
        self.L_l = 0
        self.contra_loss = 0

        # 初始化专家专有参数
        self.vera_lambda_b = torch.nn.ParameterList([
            torch.nn.Parameter(copy.deepcopy(self.vera_lambda_b.default))
            for _ in range(self.number_experts)
        ])
        self.vera_lambda_d = torch.nn.ParameterList([
            torch.nn.Parameter(copy.deepcopy(self.vera_lambda_d.default))
            for _ in range(self.number_experts)
        ])

        # 创建门控网络并确保设备一致
        self.gate = torch.nn.Linear(
            in_features=self.in_features,
            out_features=self.number_experts,
            bias=False,
            dtype=self.vera_lambda_b[0].dtype
        )

        # 定义专家ID列表，避免修改 self.active_adapters
        # 更新 active_adapters 列表，确保列表中存放的是 0 ~ number_experts-1 的专家ID
        old_adapters = len(self.active_adapters)
        for i in range(self.number_experts):
            self.active_adapters.append(i)
        for i in range(old_adapters):
            self.active_adapters.pop(0)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        """
        前向传播：
          1. 调用基础层获得初始结果。
          2. 通过门控网络计算专家权重，选取 top_k 专家并归一化。
          3. 使用共享 VeRA 参数（vera_A, vera_B）和专家特有参数（vera_lambda_b, vera_lambda_d）
             计算每个专家输出，按门控权重加权累加。
          4. 计算对比学习损失以鼓励专家间正交化。
          5. 计算负载均衡损失以鼓励专家负载均衡。
        返回：
          - 加权输出结果
          - 对比学习损失
          - 负载均衡损失
        """
        # 获取基础层输出
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        # 计算门控权重
        x_gate = x.to(self.gate.weight.dtype)
        g = self.gate(x_gate).to(torch_result_dtype)
        g = F.softmax(g, dim=-1)
        _, selected_expert_ids = torch.topk(g, self.top_k, dim=-1)
        one_hot_selected_ids = F.one_hot(selected_expert_ids, num_classes=self.number_experts).sum(dim=2)
        g = g * one_hot_selected_ids
        g = g / g.sum(dim=-1, keepdim=True)

        # 应用 dropout 一次，共享给所有专家
        dropout = self.vera_dropout['default']
        x_dropped = dropout(x).to(self.vera_lambda_b[0].dtype)

        # 计算专家输出
        raw_expert_outputs = []
        for active_adapter in self.active_adapters:
            if active_adapter >= len(self.vera_lambda_b):
                raise RuntimeError("Expert ID overflow")

            # 获取当前专家的参数
            lambda_b = self.vera_lambda_b[active_adapter]
            lambda_d = self.vera_lambda_d[active_adapter]

            # 由于 self.vera_A 和 self.vera_B 是 BufferDict，不可直接切片，所以先通过键 "default" 获取共享参数，再进行切片
            # 并转换共享参数数据类型为 lambda_b.dtype
            sliced_A = self.vera_A["default"][:, :self.in_features].to(lambda_b.dtype)
            sliced_B = self.vera_B["default"][:self.out_features, :].to(lambda_b.dtype)

            # 对输入先进行 dropout 并转换为 lambda_b.dtype
            dropout = self.vera_dropout['default']
            x_dropped = dropout(x).to(lambda_b.dtype)
            # 计算专家输出：
            # 1. 通过共享 VeRA 参数计算中间结果
            expert_intermediate = F.linear(x_dropped, sliced_A)
            # 2. 与专家专属的 lambda_d 相乘
            expert_intermediate = lambda_d * expert_intermediate
            # 3. 通过共享 VeRA B 参数计算输出，再与专家专有的 lambda_b 相乘
            expert_output = lambda_b * F.linear(expert_intermediate, sliced_B)
            raw_expert_outputs.append(expert_output)

            # 根据门控权重加权专家输出
            gate_value = g[:, :, active_adapter].unsqueeze(-1)
            weighted_expert_output = gate_value * expert_output
            result = result + weighted_expert_output

        # 计算对比学习损失
        reps = []
        for out in raw_expert_outputs:
            mean_vec = out.mean(dim=(0, 1))  # [H]
            reps.append(F.normalize(mean_vec, dim=0))
        reps = torch.stack(reps, dim=0)  # [E, H]
        sim = reps @ reps.T  # [E, E]
        eye = torch.eye(self.number_experts, device=sim.device)
        off_diag = sim * (1 - eye)
        self.expert_loss = off_diag.abs().sum() / (self.number_experts * (self.number_experts - 1))
        result = result.to(torch_result_dtype)

        # 计算负载均衡损失
        P = g.mean(dim=[0, 1])  # [number_experts]
        flat_selected_ids = selected_expert_ids.view(-1)  # [batch_size * seq_len * top_k]
        expert_counts = torch.bincount(flat_selected_ids, minlength=self.number_experts).float()  # [number_experts]
        self.use_expert = expert_counts.clone().detach()  # 保存为成员变量，记录每个专家被选中的次数
        T = flat_selected_ids.numel()  # 总的分配数量
        f = expert_counts / T  # [number_experts]
        self.L_l = self.number_experts * torch.sum(f * P)
        self.contra_loss = self.L_l*0.1 + self.expert_loss

        return result