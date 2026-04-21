# GPT-OSS Profiling

Profile the GPT-OSS reference PyTorch architecture across model sizes and token workloads.

The notebook uses dummy BF16 weights and the forward path in `model.py`. It does not add KV cache support, `torch.compile`, Triton, or custom kernels.

## Setup

1. Run `uv sync`.
2. Install a CUDA-compatible PyTorch build for the target machine.
3. Start Jupyter with `uv run jupyter lab`.
4. Open `notebooks/gpt_oss_reference_profile.ipynb`.

## Notebook

- Edit model presets or override any `ModelConfig` field
- Run preflight memory checks for dense BF16 reference weights
- Measure level 0, level 1, and level 2 timings
- Sweep prefill and generated-token workloads
- Export optional `torch.profiler` Chrome traces

## Files

- `model.py`: reference architecture with profiling scopes
- `profiling/reference.py`: model construction, preflight checks, timing, summaries, plots
- `notebooks/gpt_oss_reference_profile.ipynb`: notebook entrypoint
