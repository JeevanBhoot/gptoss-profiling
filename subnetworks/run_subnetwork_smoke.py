from __future__ import annotations

import argparse

import torch
import torch.distributed as dist

from .attention_block import GPTOSSAttentionBlock
from .attention_av_out_block import GPTOSSAttentionAVOutBlock
from .attention_qk_softmax_block import GPTOSSAttentionQKSoftmaxBlock
from .attention_qkv_block import GPTOSSAttentionQKVBlock
from .attention_sdpa_block import GPTOSSAttentionSDPABlock
from .head_block import GPTOSSHeadBlock
from .moe_block import GPTOSSMoeBlock
from .moe_mlp1_block import GPTOSSMoeMLP1Block
from .moe_mlp2_block import GPTOSSMoeMLP2Block
from .transformer_block import GPTOSSTransformerBlock, ModelConfig


GPT_OSS_20B_CONFIG = ModelConfig(
    num_hidden_layers=24,
    num_experts=32,
    experts_per_token=4,
)


GPT_OSS_120B_CONFIG = ModelConfig(
    num_hidden_layers=36,
    num_experts=128,
    experts_per_token=4,
)


TINY_DEBUG_CONFIG = ModelConfig(
    num_hidden_layers=2,
    num_experts=4,
    experts_per_token=2,
    vocab_size=1024,
    hidden_size=128,
    intermediate_size=128,
    head_dim=32,
    num_attention_heads=4,
    num_key_value_heads=1,
    sliding_window=16,
    initial_context_length=128,
    rope_theta=10000.0,
    rope_scaling_factor=1.0,
)


PRESETS = {
    "20b": GPT_OSS_20B_CONFIG,
    "120b": GPT_OSS_120B_CONFIG,
    "tiny": TINY_DEBUG_CONFIG,
}


DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def init_reference_parameters(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    with torch.no_grad():
        for name, param in module.named_parameters():
            if "scale" in name:
                param.fill_(1.0)
            elif "bias" in name or "sinks" in name:
                param.zero_()
            else:
                param.normal_(mean=0.0, std=0.02)
    module.eval()
    return module


def make_hidden_state_input(
    config: ModelConfig,
    num_tokens: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    x = torch.randn(num_tokens, config.hidden_size, dtype=torch.float32)
    return x.to(device=device, dtype=dtype)


def make_attention_state_inputs(
    config: ModelConfig,
    num_tokens: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q = torch.randn(
        num_tokens,
        config.num_key_value_heads,
        config.num_attention_heads // config.num_key_value_heads,
        config.head_dim,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    k = torch.randn(
        num_tokens,
        config.num_key_value_heads,
        config.head_dim,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    v = torch.randn(
        num_tokens,
        config.num_key_value_heads,
        config.head_dim,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    return q, k, v


def make_attention_weights_v_inputs(
    config: ModelConfig,
    num_tokens: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    q_mult = config.num_attention_heads // config.num_key_value_heads
    weights = torch.softmax(
        torch.randn(
            config.num_key_value_heads,
            q_mult,
            num_tokens,
            num_tokens,
            device=device,
            dtype=torch.float32,
        ),
        dim=-1,
    ).to(dtype)
    v = torch.randn(
        num_tokens,
        config.num_key_value_heads,
        config.head_dim,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    return weights, v


def make_expert_indices_input(
    config: ModelConfig,
    num_tokens: int,
    *,
    device: str | torch.device,
    seed: int = 0,
) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(
        low=0,
        high=config.num_experts,
        size=(num_tokens, config.experts_per_token),
        device=device,
        dtype=torch.int64,
    )


def make_moe_mlp2_inputs(
    config: ModelConfig,
    num_tokens: int,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    expert_activations = torch.randn(
        num_tokens,
        config.experts_per_token,
        config.intermediate_size // world_size,
        dtype=torch.float32,
    ).to(device=device, dtype=dtype)
    expert_weights = torch.softmax(
        torch.randn(
            num_tokens,
            config.experts_per_token,
            device=device,
            dtype=torch.float32,
        ),
        dim=1,
    ).to(dtype)
    expert_indices = torch.randint(
        low=0,
        high=config.num_experts,
        size=(num_tokens, config.experts_per_token),
        device=device,
        dtype=torch.int64,
    )
    return expert_activations, expert_weights, expert_indices


def build_module(
    block: str,
    config: ModelConfig,
    *,
    layer_idx: int,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int,
) -> torch.nn.Module:
    if block == "attention":
        module = GPTOSSAttentionBlock(
            config,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
    elif block == "attention-av-out":
        module = GPTOSSAttentionAVOutBlock(
            config,
            device=device,
            dtype=dtype,
        )
    elif block == "attention-qk-softmax":
        module = GPTOSSAttentionQKSoftmaxBlock(
            config,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
    elif block == "attention-qkv":
        module = GPTOSSAttentionQKVBlock(
            config,
            device=device,
            dtype=dtype,
        )
    elif block == "attention-sdpa":
        module = GPTOSSAttentionSDPABlock(
            config,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
    elif block == "moe":
        module = GPTOSSMoeBlock(
            config,
            device=device,
            dtype=dtype,
        )
    elif block == "moe-mlp1":
        module = GPTOSSMoeMLP1Block(
            config,
            device=device,
            dtype=dtype,
        )
    elif block == "moe-mlp2":
        module = GPTOSSMoeMLP2Block(
            config,
            device=device,
            dtype=dtype,
        )
    elif block == "transformer":
        module = GPTOSSTransformerBlock(
            config,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
    elif block == "head":
        module = GPTOSSHeadBlock(
            config,
            device=device,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unsupported block: {block}")
    return init_reference_parameters(module, seed=seed)


def make_inputs(
    block: str,
    config: ModelConfig,
    *,
    num_tokens: int,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    if block in {"attention", "attention-qkv", "moe", "transformer", "head"}:
        return (
            make_hidden_state_input(
                config,
                num_tokens,
                device=device,
                dtype=dtype,
                seed=seed,
            ),
        )
    if block == "attention-av-out":
        return make_attention_weights_v_inputs(
            config,
            num_tokens,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    if block == "attention-qk-softmax":
        q, k, _ = make_attention_state_inputs(
            config,
            num_tokens,
            device=device,
            dtype=dtype,
            seed=seed,
        )
        return q, k
    if block == "attention-sdpa":
        return make_attention_state_inputs(
            config,
            num_tokens,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    if block == "moe-mlp1":
        return (
            make_hidden_state_input(
                config,
                num_tokens,
                device=device,
                dtype=dtype,
                seed=seed,
            ),
            make_expert_indices_input(
                config,
                num_tokens,
                device=device,
                seed=seed + 1,
            ),
        )
    if block == "moe-mlp2":
        return make_moe_mlp2_inputs(
            config,
            num_tokens,
            device=device,
            dtype=dtype,
            seed=seed,
        )
    raise ValueError(f"Unsupported block: {block}")


def maybe_compile(
    module: torch.nn.Module,
    *,
    enabled: bool,
    backend: str | None,
) -> torch.nn.Module:
    if not enabled:
        return module
    if backend is None:
        return torch.compile(module)
    return torch.compile(module, backend=backend)


def run_block_smoke(
    block: str,
    config: ModelConfig,
    *,
    num_tokens: int,
    layer_idx: int,
    device: str | torch.device,
    dtype: torch.dtype,
    seed: int,
    compile_model: bool,
    backend: str | None,
) -> torch.Tensor:
    module = build_module(
        block,
        config,
        layer_idx=layer_idx,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    module = maybe_compile(module, enabled=compile_model, backend=backend)
    inputs = make_inputs(
        block,
        config,
        num_tokens=num_tokens,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    with torch.no_grad():
        return module(*inputs)


def format_output(output: object) -> str:
    if isinstance(output, torch.Tensor):
        return (
            f"shape={tuple(output.shape)} dtype={output.dtype} device={output.device}"
        )
    if isinstance(output, tuple):
        parts = []
        for index, value in enumerate(output):
            if isinstance(value, torch.Tensor):
                parts.append(
                    f"out{index}=shape{tuple(value.shape)} dtype={value.dtype} device={value.device}"
                )
            else:
                parts.append(f"out{index}={type(value).__name__}")
        return " ".join(parts)
    return type(output).__name__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test GPT-OSS subnetworks.")
    parser.add_argument(
        "--block",
        choices=[
            "attention",
            "attention-av-out",
            "attention-qk-softmax",
            "attention-qkv",
            "attention-sdpa",
            "moe",
            "moe-mlp1",
            "moe-mlp2",
            "transformer",
            "head",
            "all",
        ],
        default="all",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="tiny",
    )
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bfloat16",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--backend")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PRESETS[args.preset]
    dtype = DTYPES[args.dtype]
    blocks = (
        [
            "attention",
            "attention-av-out",
            "attention-qk-softmax",
            "attention-qkv",
            "attention-sdpa",
            "moe",
            "moe-mlp1",
            "moe-mlp2",
            "transformer",
            "head",
        ]
        if args.block == "all"
        else [args.block]
    )

    for block in blocks:
        output = run_block_smoke(
            block,
            config,
            num_tokens=args.tokens,
            layer_idx=args.layer_idx,
            device=args.device,
            dtype=dtype,
            seed=args.seed,
            compile_model=args.compile,
            backend=args.backend,
        )
        print(
            f"{block}: {format_output(output)}"
        )


if __name__ == "__main__":
    main()
