import copy
import peft
import torch
from typing import Any

class MoEVeRA(peft.tuners.lora.layer.Linear):

    def set_moe(self, number_experts: int, top_k: int):
        """
        Initialize MoE-VeRA with multiple experts.
        """
        assert top_k <= number_experts
        self.number_experts = number_experts
        self.top_k = top_k
        self.descent_strategy = None

        # Initialize shared frozen matrices A and B for VeRA
        self.lora_A = torch.nn.Parameter(torch.randn(self.lora_A.default.weight.size(1), self.lora_A.default.weight.size(0)), requires_grad=False)  # A ∈ R^{m x r}, frozen
        self.lora_B = torch.nn.Parameter(torch.randn(self.lora_B.default.weight.size(1), self.lora_B.default.weight.size(0)), requires_grad=False)  # B ∈ R^{r x n}, frozen

        # Initialize scaling vectors b and d for each expert
        self.scaling_b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(self.lora_B.size(1)))  # b ∈ R^{m}, trainable
            for i in range(self.number_experts)
        ])
        self.scaling_d = torch.nn.ParameterList([
            torch.nn.Parameter(torch.ones(self.lora_A.size(1)))  # d ∈ R^{r}, trainable
            for i in range(self.number_experts)
        ])

        # Initialize gate for MoE
        self.gate = torch.nn.Linear(
            self.lora_A.size(0),  # Input dimension is r
            self.number_experts,  # Output dimension is number of experts
            bias=False,
            dtype=self.lora_A.dtype
        )

        # Update active adapters
        old_adapters = len(self.active_adapters)
        for i in range(self.number_experts):
            self.active_adapters.append(i)
        for i in range(old_adapters):
            self.active_adapters.pop(0)
        self.use_dora = {
            i: self.use_dora['default']
            for i in range(self.number_experts)
        }

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:

        """
        Forward pass for MoE-VeRA.
        """
        #assert self.descent_strategy in {"riemannian", "moe-riemannian"}
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        if self.disable_adapters:
            # If not using VeRA
            if self.merged:
                raise NotImplementedError(
                    "MoE-VeRA shouldn't be merged. Some errors may happen."
                )
            result = self.base_layer(x, *args, **kwargs)

        elif adapter_names is not None:
            # If using mixed VeRA
            raise NotImplementedError(
                "Should run 'result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)'" +
                "but mixed_batch_forward() hasn't been implemented for MoE yet."
            )

        elif self.merged:
            # If VeRA is already merged into base model
            raise NotImplementedError(
                "MoE-VeRA shouldn't be merged now. Some errors may happen."
            )

        else:
            # If using normal MoE-VeRA forwarding
            result = self.base_layer(x, *args, **kwargs)
            torch_result_dtype = result.dtype

            # Compute gate
            g = self.gate(x).to(torch_result_dtype)
            g = torch.nn.functional.softmax(g, dim=-1)
            _, selected_expert_ids = torch.topk(g, self.top_k, dim=-1)
            one_hot_selected_ids = torch.nn.functional.one_hot(
                selected_expert_ids,
                num_classes=self.number_experts
            ).sum(dim=2)
            g = g * one_hot_selected_ids
            g = g / g.sum(dim=-1, keepdim=True)

            # Run experts
            for active_adapter in self.active_adapters:
                if active_adapter >= self.number_experts:
                    # If expert ID overflows the total number of experts, raise an error
                    raise RuntimeError

                # Get scaling vectors for the current expert
                b = self.scaling_b[active_adapter]  # Scaling vector b ∈ R^{m}
                d = self.scaling_d[active_adapter]  # Scaling vector d ∈ R^{r}

                # Compute VeRA's ΔW = Λb B Λd A
                #3072*3272 3072*4 4*4 4*3072
                Lambda_b = torch.diag(b)  # Λb = diag(b) ∈ R^{m x m}
                Lambda_d = torch.diag(d)  # Λd = diag(d) ∈ R^{r x r}
                # print(f"Lambda_b shape: {Lambda_b.shape}")
                # print(f"self.lora_B shape: {self.lora_B.shape}")
                # print(f"Lambda_d shape: {Lambda_d.shape}")
                # print(f"self.lora_A shape: {self.lora_A.shape}")
                # print(f"self.x shape: {x.shape}")
                delta_W = self.lora_A @ Lambda_d @ self.lora_B @ Lambda_b


                # Apply dropout and scaling
                dropout = self.lora_dropout['default']
                scaling = self.scaling['default']
                x = x.to(self.lora_A.dtype)
                expert_output = (dropout(x)@delta_W) * scaling

                # Weight expert output by gate value
                gate_value = g[:, :, active_adapter].unsqueeze(-1)
                if self.descent_strategy == "moe-riemannian":
                    sqrt_gate_value_const = (gate_value**0.5).clone().detach()
                    expert_output_const = expert_output.clone().detach()
                    weighted_expert_output = sqrt_gate_value_const * expert_output + (gate_value - sqrt_gate_value_const) * expert_output_const
                else:
                    weighted_expert_output = gate_value * expert_output

                # Add to result
                result = result + weighted_expert_output

            result = result.to(torch_result_dtype)
        return result