# GPT-OSS Profiling

Profile the GPT-OSS reference PyTorch architecture across model sizes and token workloads.

The notebook uses dummy BF16 weights and the forward path in `model.py`. It does not add KV cache support, `torch.compile`, Triton, or custom kernels.

## Setup

1. Run `uv sync`.
2. Install a CUDA-compatible PyTorch build for the target machine.
3. Start Jupyter with `uv run jupyter lab`.
4. Open `notebooks/gpt_oss_reference_profile.ipynb`.

## Notebook

- Edit any `ModelConfig` field directly in the top config cell
- Start with `prefill_tokens=1`, `generated_tokens=1`, `warmup_iters=0`, `measure_iters=1`
- Run preflight before running the model
- Measure level 0, level 1, and level 2 timings
- Enable sweeps only after a safe single run
- Export optional `torch.profiler` Chrome traces

## Files

- `model.py`: reference architecture with profiling scopes
- `profiling/reference.py`: model construction, preflight checks, timing, summaries, plots
- `notebooks/gpt_oss_reference_profile.ipynb`: notebook entrypoint
