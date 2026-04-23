from dataclasses import dataclass

import torch


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


def swiglu(
    x: torch.Tensor,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> torch.Tensor:
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    x_glu = x_glu.clamp(max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    return x_glu * torch.sigmoid(alpha * x_glu) * (x_linear + 1)


class GPTOSSMoeMLP1Block(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.swiglu_limit = config.swiglu_limit
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2,
                    config.hidden_size,
                ),
                device=device,
                dtype=dtype,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2),
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: torch.Tensor, expert_indices: torch.Tensor) -> torch.Tensor:
        t = torch.einsum("beck,bk->bec", self.mlp1_weight[expert_indices, ...], x)
        t = t + self.mlp1_bias[expert_indices, ...]
        return swiglu(t, limit=self.swiglu_limit)


__all__ = [
    "GPTOSSMoeMLP1Block",
    "ModelConfig",
    "swiglu",
]
