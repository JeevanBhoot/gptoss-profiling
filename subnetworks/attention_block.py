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


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: float,
        *,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> tuple[float, torch.Tensor]:
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0
            d_half = self.head_dim / 2
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq
            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq
        return concentration, inv_freq

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        positions = torch.arange(num_tokens, dtype=torch.float32, device=query.device)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration

        query_shape = query.shape
        key_shape = key.shape
        query = _apply_rotary_emb(query.view(num_tokens, -1, self.head_dim), cos, sin)
        key = _apply_rotary_emb(key.view(num_tokens, -1, self.head_dim), cos, sin)
        return query.reshape(query_shape), key.reshape(key_shape)


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


class GPTOSSAttentionBlock(torch.nn.Module):
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
        self.num_key_value_heads = config.num_key_value_heads
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=dtype)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size,
            qkv_dim,
            device=device,
            dtype=dtype,
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=dtype,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=torch.device(device) if device is not None else None,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        return x + self.out(t)


__all__ = [
    "GPTOSSAttentionBlock",
    "ModelConfig",
    "RMSNorm",
    "RotaryEmbedding",
    "sdpa",
]
