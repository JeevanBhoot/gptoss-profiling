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


class GPTOSSHeadBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.unembedding(x)
        return x


__all__ = [
    "GPTOSSHeadBlock",
    "ModelConfig",
    "RMSNorm",
]
