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


def sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sinks: torch.Tensor,
    sm_scale: float,
    sliding_window: int = 0,
) -> torch.Tensor:
    n_tokens, n_heads, q_mult, _ = q.shape
    k = k[:, :, None, :].expand(-1, -1, q_mult, -1)
    v = v[:, :, None, :].expand(-1, -1, q_mult, -1)
    sinks = sinks.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)

    mask = torch.triu(q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask = mask + torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")),
            diagonal=-sliding_window,
        )

    qk = torch.einsum("qhmd,khmd->hmqk", q, k)
    qk = qk * sm_scale
    qk = qk + mask[None, None, :, :]
    qk = torch.cat([qk, sinks], dim=-1)
    weights = torch.softmax(qk, dim=-1)[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", weights, v)
    return attn.reshape(n_tokens, -1)


class GPTOSSAttentionSDPABlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        layer_idx: int = 0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=dtype)
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        t = self.out(t)
        return t


__all__ = [
    "GPTOSSAttentionSDPABlock",
    "ModelConfig",
    "sdpa",
]
