from dataclasses import dataclass

import torch

from .attention_block import GPTOSSAttentionBlock
from .moe_block import GPTOSSMoeBlock


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


class GPTOSSTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        layer_idx: int = 0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.attention = GPTOSSAttentionBlock(
            config,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        self.moe = GPTOSSMoeBlock(
            config,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        x = self.moe(x)
        return x


__all__ = [
    "GPTOSSTransformerBlock",
    "ModelConfig",
]
