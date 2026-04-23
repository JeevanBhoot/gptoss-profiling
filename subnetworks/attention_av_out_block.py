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


class GPTOSSAttentionAVOutBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, weights: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_mult = self.num_attention_heads // self.num_key_value_heads
        v = v[:, :, None, :].expand(-1, -1, q_mult, -1)
        attn = torch.einsum("hmqk,khmd->qhmd", weights, v)
        attn = attn.reshape(attn.shape[0], -1)
        return self.out(attn)


__all__ = [
    "GPTOSSAttentionAVOutBlock",
    "ModelConfig",
]
