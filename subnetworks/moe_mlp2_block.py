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


class GPTOSSMoeMLP2Block(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        if config.intermediate_size % self.world_size != 0:
            raise ValueError("intermediate_size must be divisible by world_size")
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

    def forward(
        self,
        expert_activations: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_indices: torch.Tensor,
    ) -> torch.Tensor:
        t = torch.einsum(
            "beck,bek->bec",
            self.mlp2_weight[expert_indices, ...],
            expert_activations,
        )
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t = t + self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("bec,be->bc", t, expert_weights)
        return t


__all__ = [
    "GPTOSSMoeMLP2Block",
    "ModelConfig",
]
