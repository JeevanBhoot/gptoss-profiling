from __future__ import annotations

import argparse
import gc
import sys

import torch

from .run_subnetwork_smoke import (
    DTYPES,
    PRESETS,
    format_output,
    run_block_smoke,
)


BLOCKS = [
    "attention",
    "moe",
    "head",
    "transformer",
    "attention-qkv",
    "attention-qk-softmax",
    "attention-av-out",
    "moe-mlp1",
    "moe-mlp2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile-smoke GPT-OSS subnetworks sequentially."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Target device. Use cuda on DGX Spark.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default="tiny",
        help="Use tiny for compile smoke tests.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPES),
        default="bfloat16",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=1,
        help="Keep this very small on DGX Spark.",
    )
    parser.add_argument("--layer-idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        default=None,
        help="Optional torch.compile backend override.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of forward passes after wrapping with torch.compile.",
    )
    return parser.parse_args()


def clear_device_memory(device: str) -> None:
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available.", file=sys.stderr)
        return 1

    config = PRESETS[args.preset]
    dtype = DTYPES[args.dtype]
    failures: list[tuple[str, str]] = []

    print(
        f"device={args.device} preset={args.preset} dtype={args.dtype} "
        f"tokens={args.tokens} runs={args.runs}"
    )

    for index, block in enumerate(BLOCKS):
        print(f"[{index + 1}/{len(BLOCKS)}] {block}")
        clear_device_memory(args.device)
        try:
            output = None
            for run_index in range(args.runs):
                output = run_block_smoke(
                    block,
                    config,
                    num_tokens=args.tokens,
                    layer_idx=args.layer_idx,
                    device=args.device,
                    dtype=dtype,
                    seed=args.seed + run_index,
                    compile_model=True,
                    backend=args.backend,
                )
                if args.device.startswith("cuda"):
                    torch.cuda.synchronize()
            print(f"  ok  {format_output(output)}")
        except Exception as exc:
            failures.append((block, repr(exc)))
            print(f"  fail {repr(exc)}")
        finally:
            clear_device_memory(args.device)

    if failures:
        print("\nFailures:")
        for block, error in failures:
            print(f"- {block}: {error}")
        return 1

    print("\nAll blocks compiled successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
