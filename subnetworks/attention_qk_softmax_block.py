import math
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


class GPTOSSAttentionQKSoftmaxBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        layer_idx: int = 0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.q_mult = config.num_attention_heads // config.num_key_value_heads
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=dtype)
        )

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        num_tokens, num_heads, _, _ = q.shape
        k = k[:, :, None, :].expand(-1, -1, self.q_mult, -1)
        sinks = self.sinks.reshape(num_heads, self.q_mult, 1, 1).expand(
            -1, -1, num_tokens, -1
        )

        mask = torch.triu(q.new_full((num_tokens, num_tokens), -float("inf")), diagonal=1)
        if self.sliding_window > 0:
            mask = mask + torch.tril(
                mask.new_full((num_tokens, num_tokens), -float("inf")),
                diagonal=-self.sliding_window,
            )

        qk = torch.einsum("qhmd,khmd->hmqk", q, k)
        qk = qk * self.sm_scale
        qk = qk + mask[None, None, :, :]
        qk = torch.cat([qk, sinks], dim=-1)
        return torch.softmax(qk, dim=-1)[..., :-1]


__all__ = [
    "GPTOSSAttentionQKSoftmaxBlock",
    "ModelConfig",
]
