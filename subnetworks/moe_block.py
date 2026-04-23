from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0


class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(x.dtype)


def swiglu(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    return x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)


class GPTOSSMoeBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size,
            config.num_experts,
            device=device,
            dtype=dtype,
        )

        if config.intermediate_size % self.world_size != 0:
            raise ValueError("intermediate_size must be divisible by world_size")

        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=dtype,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=dtype,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=device,
                dtype=dtype,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t + mlp2_bias
        t = torch.einsum("bec,be->bc", t, expert_weights)
        return x + t


__all__ = [
    "GPTOSSMoeBlock",
    "ModelConfig",
    "RMSNorm",
    "swiglu",
]
