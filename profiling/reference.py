from __future__ import annotations

import contextlib
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch

from model import ModelConfig, Transformer, get_active_profiler, set_active_profiler


BF16_BYTES = 2
FP32_BYTES = 4

DEFAULT_PREFILL_SWEEP = [1, 8, 32, 128]
DEFAULT_GENERATED_SWEEP = [1, 2, 4]

GPT_OSS_20B_PRESET = ModelConfig(
    num_hidden_layers=24,
    num_experts=32,
    experts_per_token=4,
    vocab_size=201088,
    hidden_size=2880,
    intermediate_size=2880,
    swiglu_limit=7.0,
    head_dim=64,
    num_attention_heads=64,
    num_key_value_heads=8,
    sliding_window=128,
    initial_context_length=4096,
    rope_theta=150000.0,
    rope_scaling_factor=32.0,
    rope_ntk_alpha=1.0,
    rope_ntk_beta=32.0,
)

GPT_OSS_120B_PRESET = ModelConfig(
    num_hidden_layers=36,
    num_experts=128,
    experts_per_token=4,
    vocab_size=201088,
    hidden_size=2880,
    intermediate_size=2880,
    swiglu_limit=7.0,
    head_dim=64,
    num_attention_heads=64,
    num_key_value_heads=8,
    sliding_window=128,
    initial_context_length=4096,
    rope_theta=150000.0,
    rope_scaling_factor=32.0,
    rope_ntk_alpha=1.0,
    rope_ntk_beta=32.0,
)

PRESET_CONFIGS: dict[str, ModelConfig] = {
    "gpt-oss-20b": GPT_OSS_20B_PRESET,
    "gpt-oss-120b": GPT_OSS_120B_PRESET,
    "custom": ModelConfig(),
}


@dataclass(slots=True, frozen=True)
class WorkloadConfig:
    prefill_tokens: int
    generated_tokens: int


@dataclass(slots=True)
class TimingConfig:
    warmup_iters: int = 0
    measure_iters: int = 1
    seed: int = 0
    enable_torch_profiler: bool = False
    trace_output_dir: str | Path | None = None
    trace_file_stem: str = "gpt_oss_reference_trace"


def _as_model_config(config: ModelConfig | dict[str, Any]) -> ModelConfig:
    if isinstance(config, ModelConfig):
        return ModelConfig(**asdict(config))
    return ModelConfig(**config)


def model_config_from_preset(
    preset_name: str = "gpt-oss-20b", overrides: dict[str, Any] | None = None
) -> ModelConfig:
    if preset_name not in PRESET_CONFIGS:
        available = ", ".join(sorted(PRESET_CONFIGS))
        raise KeyError(f"Unknown preset {preset_name!r}. Available presets: {available}")
    values = asdict(PRESET_CONFIGS[preset_name])
    if overrides:
        values.update(overrides)
    return ModelConfig(**values)


def _config_metadata(config: ModelConfig, preset_name: str) -> dict[str, Any]:
    metadata = {"preset_name": preset_name}
    metadata.update(asdict(config))
    return metadata


def format_bytes(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "n/a"
    suffixes = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    value = float(num_bytes)
    for suffix in suffixes:
        if abs(value) < 1024.0 or suffix == suffixes[-1]:
            return f"{value:.2f} {suffix}"
        value /= 1024.0
    return f"{value:.2f} PiB"


def _parameter_breakdown(config: ModelConfig) -> dict[str, int]:
    qkv_dim = config.head_dim * (
        config.num_attention_heads + 2 * config.num_key_value_heads
    )
    attn_out_dim = config.head_dim * config.num_attention_heads

    embedding_params = config.vocab_size * config.hidden_size
    unembedding_params = config.vocab_size * config.hidden_size

    attn_params = (
        config.num_attention_heads
        + config.hidden_size
        + config.hidden_size * qkv_dim
        + qkv_dim
        + config.hidden_size * attn_out_dim
        + config.hidden_size
    )

    mlp_params = (
        config.hidden_size
        + config.num_experts * config.hidden_size
        + config.num_experts
        + config.num_experts * 2 * config.intermediate_size * config.hidden_size
        + config.num_experts * 2 * config.intermediate_size
        + config.num_experts * config.hidden_size * config.intermediate_size
        + config.num_experts * config.hidden_size
    )

    final_norm_params = config.hidden_size

    return {
        "embedding_params": embedding_params,
        "transformer_block_attention_params": attn_params,
        "transformer_block_mlp_params": mlp_params,
        "transformer_block_total_params": attn_params + mlp_params,
        "final_norm_params": final_norm_params,
        "unembedding_params": unembedding_params,
        "total_params": (
            embedding_params
            + config.num_hidden_layers * (attn_params + mlp_params)
            + final_norm_params
            + unembedding_params
        ),
    }


def _weight_memory_bytes(config: ModelConfig) -> dict[str, int]:
    qkv_dim = config.head_dim * (
        config.num_attention_heads + 2 * config.num_key_value_heads
    )
    attn_out_dim = config.head_dim * config.num_attention_heads

    bf16_param_count = (
        config.vocab_size * config.hidden_size
        + config.num_hidden_layers
        * (
            config.num_attention_heads
            + config.hidden_size * qkv_dim
            + qkv_dim
            + config.hidden_size * attn_out_dim
            + config.hidden_size
            + config.num_experts * config.hidden_size
            + config.num_experts
            + config.num_experts * 2 * config.intermediate_size * config.hidden_size
            + config.num_experts * 2 * config.intermediate_size
            + config.num_experts * config.hidden_size * config.intermediate_size
            + config.num_experts * config.hidden_size
        )
        + config.vocab_size * config.hidden_size
    )
    fp32_param_count = config.hidden_size * (2 * config.num_hidden_layers + 1)

    return {
        "bf16_param_count": bf16_param_count,
        "fp32_param_count": fp32_param_count,
        "dense_weight_bytes": bf16_param_count * BF16_BYTES + fp32_param_count * FP32_BYTES,
    }


def _runtime_memory_breakdown(
    config: ModelConfig, max_sequence_length: int
) -> dict[str, int | str]:
    qkv_dim = config.head_dim * (
        config.num_attention_heads + 2 * config.num_key_value_heads
    )

    hidden_state_bytes = max_sequence_length * config.hidden_size * BF16_BYTES
    attention_qkv_bytes = max_sequence_length * qkv_dim * BF16_BYTES
    attention_scores_bytes = (
        max_sequence_length
        * max_sequence_length
        * config.num_attention_heads
        * BF16_BYTES
    )
    moe_activation_bytes = (
        max_sequence_length
        * config.experts_per_token
        * (2 * config.intermediate_size + config.hidden_size)
        * BF16_BYTES
    )
    logits_bytes = max_sequence_length * config.vocab_size * BF16_BYTES

    moe_mlp1_gather_bytes = (
        max_sequence_length
        * config.experts_per_token
        * 2
        * config.intermediate_size
        * config.hidden_size
        * BF16_BYTES
    )
    moe_mlp1_bias_bytes = (
        max_sequence_length
        * config.experts_per_token
        * 2
        * config.intermediate_size
        * BF16_BYTES
    )
    moe_mlp2_gather_bytes = (
        max_sequence_length
        * config.experts_per_token
        * config.hidden_size
        * config.intermediate_size
        * BF16_BYTES
    )
    moe_mlp2_bias_bytes = (
        max_sequence_length
        * config.experts_per_token
        * config.hidden_size
        * BF16_BYTES
    )

    rough_activation_bytes = (
        hidden_state_bytes
        + attention_qkv_bytes
        + attention_scores_bytes
        + moe_activation_bytes
    )
    moe_mlp1_temp_bytes = moe_mlp1_gather_bytes + moe_mlp1_bias_bytes
    moe_mlp2_temp_bytes = moe_mlp2_gather_bytes + moe_mlp2_bias_bytes

    peak_runtime_candidates = {
        "rough_activation_bytes": rough_activation_bytes,
        "logits_bytes": logits_bytes,
        "moe_mlp1_temp_bytes": moe_mlp1_temp_bytes,
        "moe_mlp2_temp_bytes": moe_mlp2_temp_bytes,
    }
    peak_runtime_component, peak_runtime_bytes = max(
        peak_runtime_candidates.items(), key=lambda item: item[1]
    )

    return {
        "hidden_state_bytes": hidden_state_bytes,
        "attention_qkv_bytes": attention_qkv_bytes,
        "rough_attention_working_set_bytes": attention_scores_bytes,
        "rough_moe_activation_bytes": moe_activation_bytes,
        "rough_activation_bytes": rough_activation_bytes,
        "logits_bytes": logits_bytes,
        "moe_mlp1_gather_bytes": moe_mlp1_gather_bytes,
        "moe_mlp1_bias_bytes": moe_mlp1_bias_bytes,
        "moe_mlp1_temp_bytes": moe_mlp1_temp_bytes,
        "moe_mlp2_gather_bytes": moe_mlp2_gather_bytes,
        "moe_mlp2_bias_bytes": moe_mlp2_bias_bytes,
        "moe_mlp2_temp_bytes": moe_mlp2_temp_bytes,
        "peak_runtime_component": peak_runtime_component,
        "peak_runtime_bytes": peak_runtime_bytes,
    }


def preflight_report(
    arch_cfg: ModelConfig | dict[str, Any],
    workload_cfg: WorkloadConfig | dict[str, Any],
    device: str | torch.device = "cuda",
) -> dict[str, Any]:
    config = _as_model_config(arch_cfg)
    workload = (
        workload_cfg
        if isinstance(workload_cfg, WorkloadConfig)
        else WorkloadConfig(**workload_cfg)
    )
    resolved_device = torch.device(device)

    max_sequence_length = workload.prefill_tokens + max(workload.generated_tokens - 1, 0)

    params = _parameter_breakdown(config)
    weights = _weight_memory_bytes(config)
    runtime = _runtime_memory_breakdown(config, max_sequence_length)
    estimated_peak_bytes = weights["dense_weight_bytes"] + runtime["peak_runtime_bytes"]

    report = {
        **params,
        **weights,
        **runtime,
        "prefill_tokens": workload.prefill_tokens,
        "generated_tokens": workload.generated_tokens,
        "max_sequence_length": max_sequence_length,
        "estimated_peak_bytes": estimated_peak_bytes,
        "estimated_required_bytes": estimated_peak_bytes,
        "device": str(resolved_device),
        "device_free_bytes": None,
        "device_total_bytes": None,
        "fits_free_memory": None,
    }

    if resolved_device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(resolved_device)
        report["device_free_bytes"] = int(free_bytes)
        report["device_total_bytes"] = int(total_bytes)
        report["fits_free_memory"] = estimated_peak_bytes <= free_bytes

    return report


def _raise_if_not_runnable(report: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    fits = report.get("fits_free_memory")
    if fits is None or fits:
        return
    required = format_bytes(report["estimated_peak_bytes"])
    free = format_bytes(report["device_free_bytes"])
    weights = format_bytes(report["dense_weight_bytes"])
    runtime_peak = format_bytes(report["peak_runtime_bytes"])
    runtime_component = report["peak_runtime_component"]
    raise MemoryError(
        "Selected configuration is unlikely to fit on the target device in the "
        "reference BF16 implementation. "
        f"Estimated peak: {required} (weights {weights}, peak runtime "
        f"{runtime_peak} from {runtime_component}); free device memory: {free}."
    )


def _raise_if_runtime_memory_does_not_fit(
    report: dict[str, Any], device: torch.device, runtime_required_bytes: int
) -> None:
    if device.type != "cuda":
        return
    free_bytes = report.get("device_free_bytes")
    if free_bytes is None or runtime_required_bytes <= free_bytes:
        return
    required = format_bytes(runtime_required_bytes)
    free = format_bytes(free_bytes)
    raise MemoryError(
        "Selected workload is unlikely to fit on the target device with the existing "
        "reference model instance. "
        f"Required peak runtime memory: {required}; free device memory: {free}."
    )


@contextlib.contextmanager
def _installed_profiler(profiler: Any) -> Iterator[None]:
    previous = get_active_profiler()
    set_active_profiler(profiler)
    try:
        yield
    finally:
        set_active_profiler(previous)


def build_reference_model(
    arch_cfg: ModelConfig | dict[str, Any],
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> Transformer:
    config = _as_model_config(arch_cfg)
    resolved_device = torch.device(device)

    torch.manual_seed(seed)
    if resolved_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = Transformer(config=config, device=resolved_device)
    model.eval()

    with torch.inference_mode():
        for name, param in model.named_parameters():
            if not param.is_floating_point():
                continue
            if "scale" in name:
                param.fill_(1.0)
            elif "bias" in name or "sinks" in name:
                param.zero_()
            else:
                param.normal_(mean=0.0, std=0.02)

    return model


class _TimingRecorder:
    def __init__(self, device: torch.device, base_metadata: dict[str, Any], enable_trace: bool):
        self.device = device
        self.base_metadata = dict(base_metadata)
        self.enable_trace = enable_trace
        self._context_stack: list[dict[str, Any]] = []
        self._rows: list[dict[str, Any]] = []
        self._pending_events: list[tuple[dict[str, Any], torch.cuda.Event, torch.cuda.Event]] = []

    def _merged_metadata(self, scope_name: str, metadata: dict[str, Any]) -> dict[str, Any]:
        merged = dict(self.base_metadata)
        for item in self._context_stack:
            merged.update(item)
        merged.update({key: value for key, value in metadata.items() if value is not None})
        merged["scope_name"] = scope_name
        return merged

    def _trace_label(self, scope_name: str, metadata: dict[str, Any]) -> str:
        label = [scope_name]
        if "phase" in metadata:
            label.append(f"phase={metadata['phase']}")
        if "level" in metadata:
            label.append(f"L{metadata['level']}")
        if "component" in metadata:
            label.append(f"component={metadata['component']}")
        if metadata.get("layer_idx") is not None:
            label.append(f"layer={metadata['layer_idx']}")
        if metadata.get("section"):
            label.append(f"section={metadata['section']}")
        return " | ".join(label)

    @contextlib.contextmanager
    def context(self, **metadata: Any) -> Iterator[None]:
        self._context_stack.append(metadata)
        try:
            yield
        finally:
            self._context_stack.pop()

    @contextlib.contextmanager
    def scope(self, scope_name: str, **metadata: Any) -> Iterator[None]:
        payload = self._merged_metadata(scope_name, metadata)
        trace_cm = (
            torch.profiler.record_function(self._trace_label(scope_name, payload))
            if self.enable_trace
            else contextlib.nullcontext()
        )

        if self.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with trace_cm:
                start_event.record()
                try:
                    yield
                finally:
                    end_event.record()
            self._pending_events.append((payload, start_event, end_event))
            return

        start_time = time.perf_counter()
        with trace_cm:
            try:
                yield
            finally:
                duration_ms = (time.perf_counter() - start_time) * 1000.0
        self._rows.append({**payload, "duration_ms": duration_ms})

    def add_row(self, scope_name: str, duration_ms: float, **metadata: Any) -> None:
        payload = self._merged_metadata(scope_name, metadata)
        self._rows.append({**payload, "duration_ms": float(duration_ms)})

    def finalize(self) -> list[dict[str, Any]]:
        if self.device.type == "cuda" and self._pending_events:
            torch.cuda.synchronize(self.device)
            for payload, start_event, end_event in self._pending_events:
                duration_ms = start_event.elapsed_time(end_event)
                self._rows.append({**payload, "duration_ms": duration_ms})
            self._pending_events.clear()
        return list(self._rows)


def _sample_next_token(
    logits: torch.Tensor,
    recorder: _TimingRecorder,
    *,
    phase: str,
    decode_step: int | None = None,
) -> torch.Tensor:
    with recorder.context(phase=phase, decode_step=decode_step):
        with recorder.scope(
            "embedding_plus_head",
            level=1,
            component="embedding_plus_head",
            section="sampling",
        ):
            with recorder.scope(
                "sampling", level=2, component="embedding_plus_head", section="sampling"
            ):
                return torch.argmax(logits[-1], dim=-1).to(dtype=torch.long)


def _make_prompt_tokens(
    config: ModelConfig, workload: WorkloadConfig, seed: int, device: torch.device
) -> torch.Tensor:
    generator_device = "cuda" if device.type == "cuda" else "cpu"
    generator = torch.Generator(device=generator_device)
    generator.manual_seed(seed)
    return torch.randint(
        low=0,
        high=config.vocab_size,
        size=(workload.prefill_tokens,),
        dtype=torch.long,
        generator=generator,
        device=device,
    )


@torch.inference_mode()
def _execute_workload(
    model: Transformer,
    config: ModelConfig,
    workload: WorkloadConfig,
    recorder: _TimingRecorder,
    *,
    iteration: int,
    iteration_kind: str,
    seed: int,
) -> list[dict[str, Any]]:
    device = next(model.parameters()).device
    prompt_tokens = _make_prompt_tokens(config, workload, seed, device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    with _installed_profiler(recorder):
        with recorder.context(iteration=iteration, iteration_kind=iteration_kind):
            with recorder.scope("workload_total", level=0, component="workload"):
                with recorder.context(phase="prefill", decode_step=None):
                    with recorder.scope("prefill_total", level=0, component="workload"):
                        logits = model(prompt_tokens)
                        generated_tokens = []
                        if workload.generated_tokens > 0:
                            next_token = _sample_next_token(logits, recorder, phase="prefill")
                            generated_tokens.append(next_token)
                            prompt_tokens = torch.cat((prompt_tokens, next_token.view(1)))

                remaining_decode_steps = max(workload.generated_tokens - 1, 0)
                if remaining_decode_steps > 0:
                    with recorder.context(phase="decode"):
                        with recorder.scope("decode_total", level=0, component="workload"):
                            for decode_step in range(remaining_decode_steps):
                                with recorder.context(decode_step=decode_step):
                                    logits = model(prompt_tokens)
                                    next_token = _sample_next_token(
                                        logits,
                                        recorder,
                                        phase="decode",
                                        decode_step=decode_step,
                                    )
                                    generated_tokens.append(next_token)
                                    prompt_tokens = torch.cat(
                                        (prompt_tokens, next_token.view(1))
                                    )
                else:
                    recorder.add_row(
                        "decode_total",
                        0.0,
                        level=0,
                        component="workload",
                        phase="decode",
                        decode_step=None,
                    )

    return recorder.finalize()


def _run_trace_capture(
    model: Transformer,
    config: ModelConfig,
    workload: WorkloadConfig,
    timing: TimingConfig,
    base_metadata: dict[str, Any],
) -> Path | None:
    if not timing.enable_torch_profiler:
        return None

    output_dir = Path(timing.trace_output_dir or "traces")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = (
        f"{timing.trace_file_stem}_p{workload.prefill_tokens}_g{workload.generated_tokens}.json"
    )
    trace_path = output_dir / file_stem

    device = next(model.parameters()).device
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    recorder = _TimingRecorder(device, base_metadata=base_metadata, enable_trace=True)
    with torch.profiler.profile(activities=activities) as profiler:
        _execute_workload(
            model,
            config,
            workload,
            recorder,
            iteration=0,
            iteration_kind="trace",
            seed=timing.seed,
        )
    profiler.export_chrome_trace(str(trace_path))
    return trace_path


def run_workload(
    arch_cfg: ModelConfig | dict[str, Any],
    workload_cfg: WorkloadConfig | dict[str, Any],
    timing_cfg: TimingConfig | dict[str, Any],
    *,
    device: str | torch.device = "cuda",
    preset_name: str = "custom",
    model: Transformer | None = None,
) -> pd.DataFrame:
    config = _as_model_config(arch_cfg)
    workload = (
        workload_cfg
        if isinstance(workload_cfg, WorkloadConfig)
        else WorkloadConfig(**workload_cfg)
    )
    timing = timing_cfg if isinstance(timing_cfg, TimingConfig) else TimingConfig(**timing_cfg)
    resolved_device = torch.device(device)

    report = preflight_report(config, workload, resolved_device)
    if model is None:
        _raise_if_not_runnable(report, resolved_device)
    else:
        report["weights_already_loaded"] = True
        report["runtime_required_bytes"] = report["peak_runtime_bytes"]
        if report.get("device_free_bytes") is not None:
            report["fits_free_memory"] = (
                report["runtime_required_bytes"] <= report["device_free_bytes"]
            )
        _raise_if_runtime_memory_does_not_fit(
            report, resolved_device, report["runtime_required_bytes"]
        )

    if model is None:
        model = build_reference_model(config, device=resolved_device, seed=timing.seed)

    base_metadata = {
        **_config_metadata(config, preset_name),
        "device": str(resolved_device),
        "prefill_tokens": workload.prefill_tokens,
        "generated_tokens": workload.generated_tokens,
        "seed": timing.seed,
    }

    rows: list[dict[str, Any]] = []
    for warmup_idx in range(timing.warmup_iters):
        recorder = _TimingRecorder(resolved_device, base_metadata, enable_trace=False)
        _execute_workload(
            model,
            config,
            workload,
            recorder,
            iteration=warmup_idx,
            iteration_kind="warmup",
            seed=timing.seed,
        )

    for measure_idx in range(timing.measure_iters):
        recorder = _TimingRecorder(resolved_device, base_metadata, enable_trace=False)
        rows.extend(
            _execute_workload(
                model,
                config,
                workload,
                recorder,
                iteration=measure_idx,
                iteration_kind="measure",
                seed=timing.seed,
            )
        )

    trace_path = _run_trace_capture(model, config, workload, timing, base_metadata)
    df = pd.DataFrame(rows)
    if trace_path is not None:
        df.attrs["trace_path"] = str(trace_path)
    df.attrs["preflight"] = report
    return df


def run_workload_sweep(
    arch_cfg: ModelConfig | dict[str, Any],
    *,
    prefill_tokens: list[int] | None = None,
    generated_tokens: list[int] | None = None,
    timing_cfg: TimingConfig | dict[str, Any] | None = None,
    device: str | torch.device = "cuda",
    preset_name: str = "custom",
) -> pd.DataFrame:
    config = _as_model_config(arch_cfg)
    timing = timing_cfg if isinstance(timing_cfg, TimingConfig) else TimingConfig(**(timing_cfg or {}))
    resolved_device = torch.device(device)
    prefill_values = prefill_tokens or list(DEFAULT_PREFILL_SWEEP)
    generated_values = generated_tokens or list(DEFAULT_GENERATED_SWEEP)

    max_workload = WorkloadConfig(
        prefill_tokens=max(prefill_values),
        generated_tokens=max(generated_values),
    )
    _raise_if_not_runnable(preflight_report(config, max_workload, resolved_device), resolved_device)

    model = build_reference_model(config, device=resolved_device, seed=timing.seed)
    frames = []
    preflight_by_workload: dict[str, dict[str, Any]] = {}
    trace_paths: list[str] = []

    for prefill in prefill_values:
        for generated in generated_values:
            workload = WorkloadConfig(prefill_tokens=prefill, generated_tokens=generated)
            frame = run_workload(
                config,
                workload,
                timing,
                device=resolved_device,
                preset_name=preset_name,
                model=model,
            )
            frames.append(frame)
            key = f"prefill={prefill}|generated={generated}"
            preflight_by_workload[key] = frame.attrs.get("preflight", {})
            trace_path = frame.attrs.get("trace_path")
            if trace_path:
                trace_paths.append(trace_path)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    result.attrs["preflight_by_workload"] = preflight_by_workload
    if trace_paths:
        result.attrs["trace_paths"] = trace_paths
    return result


def _group_keys(df: pd.DataFrame) -> list[str]:
    candidate_columns = [
        "preset_name",
        "device",
        "prefill_tokens",
        "generated_tokens",
        "seed",
    ]
    return [column for column in candidate_columns if column in df.columns]


def _aggregate_scope_rows(df: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    grouped = (
        df.groupby(group_columns, dropna=False)["duration_ms"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "mean_duration_ms",
                "std": "std_duration_ms",
                "min": "min_duration_ms",
                "max": "max_duration_ms",
                "count": "sample_count",
            }
        )
    )
    grouped["std_duration_ms"] = grouped["std_duration_ms"].fillna(0.0)
    return grouped


def _combined_phase_rows(df: pd.DataFrame, phase_columns: list[str]) -> pd.DataFrame:
    phase_df = df[df["phase"].isin(["prefill", "decode"])].copy()
    if phase_df.empty:
        return phase_df
    group_columns = [column for column in phase_columns if column != "phase"]
    combined = (
        phase_df.groupby(group_columns, dropna=False)["duration_ms"].sum().reset_index()
    )
    combined["phase"] = "combined"
    return combined


def summarize_results(rows: pd.DataFrame | list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    df = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        empty = pd.DataFrame()
        return {
            "raw": empty,
            "level0_scopes": empty,
            "level0_metrics": empty,
            "level1": empty,
            "level2": empty,
            "level2_full": empty,
        }

    workload_keys = _group_keys(df)

    raw = df.copy()

    level0_iteration = df[df["level"] == 0].copy()
    level0_scopes = _aggregate_scope_rows(
        level0_iteration,
        workload_keys + ["scope_name"],
    )
    level0_metrics = level0_scopes.pivot_table(
        index=workload_keys,
        columns="scope_name",
        values="mean_duration_ms",
        fill_value=0.0,
    ).reset_index()
    level0_metrics.columns.name = None

    for required_column in ("prefill_total", "decode_total", "workload_total"):
        if required_column not in level0_metrics.columns:
            level0_metrics[required_column] = 0.0

    decode_steps = (level0_metrics["generated_tokens"] - 1).clip(lower=0)
    level0_metrics["avg_ms_per_new_token"] = level0_metrics["workload_total"] / level0_metrics[
        "generated_tokens"
    ].where(level0_metrics["generated_tokens"] > 0)
    level0_metrics["prefill_tok_s"] = level0_metrics["prefill_tokens"] / (
        level0_metrics["prefill_total"] / 1000.0
    )
    level0_metrics["decode_steps"] = decode_steps
    level0_metrics["decode_tok_s"] = decode_steps / (
        level0_metrics["decode_total"] / 1000.0
    ).where(level0_metrics["decode_total"] > 0)
    ordered_level0_columns = [
        column
        for column in [
            "preset_name",
            "device",
            "prefill_tokens",
            "generated_tokens",
            "prefill_total",
            "decode_total",
            "workload_total",
            "avg_ms_per_new_token",
            "prefill_tok_s",
            "decode_tok_s",
            "decode_steps",
            "seed",
        ]
        if column in level0_metrics.columns
    ]
    level0_metrics = level0_metrics[ordered_level0_columns]

    level1_iteration = (
        df[df["level"] == 1]
        .groupby(
            workload_keys + ["iteration", "component", "scope_name", "phase"],
            dropna=False,
        )["duration_ms"]
        .sum()
        .reset_index()
    )
    level1_phase_keys = workload_keys + ["iteration", "component", "scope_name", "phase"]
    level1_combined = _combined_phase_rows(level1_iteration, level1_phase_keys)
    level1_summary = _aggregate_scope_rows(
        pd.concat((level1_iteration, level1_combined), ignore_index=True),
        workload_keys + ["phase", "component", "scope_name"],
    )
    ordered_level1_columns = [
        column
        for column in [
            "preset_name",
            "device",
            "prefill_tokens",
            "generated_tokens",
            "phase",
            "component",
            "scope_name",
            "mean_duration_ms",
            "std_duration_ms",
            "min_duration_ms",
            "max_duration_ms",
            "sample_count",
            "seed",
        ]
        if column in level1_summary.columns
    ]
    level1_summary = level1_summary[ordered_level1_columns]

    level2_iteration = df[df["level"] == 2].copy()
    level2_phase_keys = workload_keys + [
        "iteration",
        "component",
        "scope_name",
        "layer_idx",
        "phase",
    ]
    level2_combined = _combined_phase_rows(level2_iteration, level2_phase_keys)
    level2_full = _aggregate_scope_rows(
        pd.concat((level2_iteration, level2_combined), ignore_index=True),
        workload_keys + ["phase", "component", "layer_idx", "scope_name"],
    )
    ordered_level2_full_columns = [
        column
        for column in [
            "preset_name",
            "device",
            "prefill_tokens",
            "generated_tokens",
            "phase",
            "component",
            "layer_idx",
            "scope_name",
            "mean_duration_ms",
            "std_duration_ms",
            "min_duration_ms",
            "max_duration_ms",
            "sample_count",
            "seed",
        ]
        if column in level2_full.columns
    ]
    level2_full = level2_full[ordered_level2_full_columns]

    level2_compact_iteration = (
        pd.concat((level2_iteration, level2_combined), ignore_index=True)
        .groupby(
            workload_keys + ["iteration", "phase", "component", "scope_name"],
            dropna=False,
        )["duration_ms"]
        .sum()
        .reset_index()
    )
    level2_summary = _aggregate_scope_rows(
        level2_compact_iteration,
        workload_keys + ["phase", "component", "scope_name"],
    )
    ordered_level2_columns = [
        column
        for column in [
            "preset_name",
            "device",
            "prefill_tokens",
            "generated_tokens",
            "phase",
            "component",
            "scope_name",
            "mean_duration_ms",
            "std_duration_ms",
            "min_duration_ms",
            "max_duration_ms",
            "sample_count",
            "seed",
        ]
        if column in level2_summary.columns
    ]
    level2_summary = level2_summary[ordered_level2_columns]

    return {
        "raw": raw,
        "level0_scopes": level0_scopes,
        "level0_metrics": level0_metrics,
        "level1": level1_summary,
        "level2": level2_summary,
        "level2_full": level2_full,
    }


def plot_level0_breakdown(level0_metrics: pd.DataFrame):
    import matplotlib.pyplot as plt

    if level0_metrics.empty:
        raise ValueError("level0_metrics is empty.")

    fig, ax = plt.subplots(figsize=(10, 5))
    frame = level0_metrics.copy()
    frame["label"] = (
        "P="
        + frame["prefill_tokens"].astype(str)
        + " | G="
        + frame["generated_tokens"].astype(str)
    )

    ax.bar(frame["label"], frame["prefill_total"], label="prefill_total")
    ax.bar(
        frame["label"],
        frame["decode_total"],
        bottom=frame["prefill_total"],
        label="decode_total",
    )
    ax.set_ylabel("Time (ms)")
    ax.set_title("Level 0 Breakdown")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_level1_breakdown(level1_summary: pd.DataFrame, phase: str = "combined"):
    import matplotlib.pyplot as plt

    frame = level1_summary[level1_summary["phase"] == phase].copy()
    if frame.empty:
        raise ValueError(f"No level 1 rows for phase={phase!r}.")

    pivot = frame.pivot_table(
        index=["prefill_tokens", "generated_tokens"],
        columns="scope_name",
        values="mean_duration_ms",
        fill_value=0.0,
    )
    labels = [
        f"P={prefill} | G={generated}"
        for prefill, generated in pivot.index.to_list()
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = None
    for scope_name in pivot.columns:
        values = pivot[scope_name].to_numpy()
        ax.bar(labels, values, bottom=bottom, label=scope_name)
        bottom = values if bottom is None else bottom + values
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Level 1 Breakdown ({phase})")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_level2_heatmap(
    level2_summary: pd.DataFrame,
    *,
    phase: str = "prefill",
    component: str = "attention",
    prefill_tokens: int | None = None,
    generated_tokens: int | None = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    frame = level2_summary[
        (level2_summary["phase"] == phase)
        & (level2_summary["component"] == component)
    ].copy()
    if prefill_tokens is not None:
        frame = frame[frame["prefill_tokens"] == prefill_tokens]
    if generated_tokens is not None:
        frame = frame[frame["generated_tokens"] == generated_tokens]
    if frame.empty:
        raise ValueError("No level 2 rows matched the requested filters.")

    pivot = frame.pivot_table(
        index="layer_idx",
        columns="scope_name",
        values="mean_duration_ms",
        fill_value=0.0,
    ).sort_index()

    fig, ax = plt.subplots(figsize=(12, max(4, 0.35 * len(pivot.index))))
    sns.heatmap(pivot, cmap="magma", ax=ax)
    ax.set_title(f"Level 2 {component} heatmap ({phase})")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Operation")
    fig.tight_layout()
    return fig, ax


def plot_workload_grid(level0_metrics: pd.DataFrame, value: str = "workload_total"):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if level0_metrics.empty:
        raise ValueError("level0_metrics is empty.")
    if value not in level0_metrics.columns:
        raise KeyError(f"{value!r} is not present in level0_metrics.")

    pivot = level0_metrics.pivot_table(
        index="generated_tokens",
        columns="prefill_tokens",
        values=value,
        fill_value=0.0,
    ).sort_index().sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis", ax=ax)
    ax.set_title(f"Workload Grid: {value}")
    ax.set_ylabel("Generated Tokens")
    ax.set_xlabel("Prefill Tokens")
    fig.tight_layout()
    return fig, ax
