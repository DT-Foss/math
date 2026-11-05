# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.7.1",
#   "transformers==4.57.1",
#   "safetensors>=0.5.3",
#   "huggingface-hub>=0.35.0",
#   "numpy>=2.0,<3",
# ]
# ///
from __future__ import annotations

from collections import defaultdict
import gc
import json
import math
import os
import platform
import statistics
import time
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = os.environ.get("MODEL_ID", "HuggingFaceTB/SmolLM2-135M-Instruct")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "24"))
BITS = tuple(int(x) for x in os.environ.get("BITS", "4,5,8").split(",") if x)
REPEATS = int(os.environ.get("REPEATS", "3"))
THREADS = int(os.environ.get("TORCH_THREADS", str(min(8, os.cpu_count() or 1))))
PROMPT = os.environ.get(
    "PROMPT",
    "Explain in one concise paragraph why persistent mathematical state can replace repeated dense computation.",
)


def median_ms(fn: Callable[[], object], repeats: int = REPEATS) -> float:
    fn()
    values = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        fn()
        values.append((time.perf_counter_ns() - start) / 1e6)
    return float(statistics.median(values))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] * (hi - pos) + xs[hi] * (pos - lo))


def energy_rank(delta_matrix: torch.Tensor, fraction: float) -> int:
    if delta_matrix.numel() == 0 or float(torch.linalg.vector_norm(delta_matrix)) == 0.0:
        return 0
    singular = torch.linalg.svdvals(delta_matrix.to(torch.float64))
    energy = singular.square()
    target = fraction * float(energy.sum())
    cumulative = torch.cumsum(energy, dim=0)
    return int(torch.searchsorted(cumulative, torch.tensor(target, dtype=cumulative.dtype)).item() + 1)


def quantize_symmetric(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    qmax = (1 << (bits - 1)) - 1
    max_abs = float(x.abs().max())
    scale = max(max_abs / max(qmax, 1), torch.finfo(torch.float32).tiny)
    codes = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int16)
    decoded = codes.to(torch.float32) * scale
    return codes, decoded, scale


def source_metrics(x: torch.Tensor, codes: torch.Tensor, decoded: torch.Tensor) -> dict:
    deltas = x[1:] - x[:-1]
    code_deltas = codes[1:] != codes[:-1]
    counts = code_deltas.sum(dim=1).to(torch.float64)
    exact_counts = (deltas != 0).sum(dim=1).to(torch.float64)
    rel_error = float(torch.linalg.vector_norm(decoded - x) / torch.linalg.vector_norm(x).clamp_min(1e-30))
    return {
        "width": int(x.shape[1]),
        "transitions": int(len(x) - 1),
        "float_exact_nonzero_fraction_median": float(torch.median(exact_counts / x.shape[1])),
        "code_change_count_median": float(torch.median(counts)),
        "code_change_count_p90": percentile(counts.tolist(), 0.90),
        "code_change_count_max": float(counts.max()) if len(counts) else 0.0,
        "code_change_fraction_median": float(torch.median(counts / x.shape[1])),
        "code_change_fraction_p90": percentile((counts / x.shape[1]).tolist(), 0.90),
        "delta_rank_90": energy_rank(deltas, 0.90),
        "delta_rank_99": energy_rank(deltas, 0.99),
        "delta_rank_999": energy_rank(deltas, 0.999),
        "quantized_relative_l2_error": rel_error,
    }


def replay_linear(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    decoded: torch.Tensor,
    codes: torch.Tensor,
    scale: float,
) -> dict:
    weight = weight.detach().cpu().to(torch.float32).contiguous()
    bias = None if bias is None else bias.detach().cpu().to(torch.float32).contiguous()
    columns = weight.t().contiguous()
    transitions: list[tuple[torch.Tensor, torch.Tensor]] = []
    supports: list[int] = []
    for t in range(1, len(codes)):
        idx = torch.nonzero(codes[t] != codes[t - 1], as_tuple=False).flatten()
        delta = (codes[t].index_select(0, idx) - codes[t - 1].index_select(0, idx)).to(torch.float32) * scale
        transitions.append((idx, delta))
        supports.append(int(idx.numel()))

    y0 = torch.mv(weight, decoded[0])
    if bias is not None:
        y0 = y0 + bias

    def full_replay() -> torch.Tensor:
        y = y0
        for t in range(1, len(decoded)):
            y = torch.mv(weight, decoded[t])
            if bias is not None:
                y = y + bias
        return y

    def delta_apply_replay() -> torch.Tensor:
        y = y0.clone()
        for idx, delta in transitions:
            if idx.numel():
                y.add_(torch.matmul(delta, columns.index_select(0, idx)))
        return y

    def delta_detect_replay() -> torch.Tensor:
        y = y0.clone()
        previous = codes[0]
        for t in range(1, len(codes)):
            current = codes[t]
            idx = torch.nonzero(current != previous, as_tuple=False).flatten()
            if idx.numel():
                delta = (current.index_select(0, idx) - previous.index_select(0, idx)).to(torch.float32) * scale
                y.add_(torch.matmul(delta, columns.index_select(0, idx)))
            previous = current
        return y

    reference = full_replay()
    candidate = delta_detect_replay()
    max_abs = float((candidate - reference).abs().max())
    max_rel = float((candidate - reference).abs().max() / reference.abs().max().clamp_min(1e-30))

    full_ms = median_ms(full_replay)
    apply_ms = median_ms(delta_apply_replay)
    detect_ms = median_ms(delta_detect_replay)
    result = {
        "input_width": int(weight.shape[1]),
        "output_width": int(weight.shape[0]),
        "transitions": len(transitions),
        "support_median": float(statistics.median(supports)) if supports else 0.0,
        "support_p90": percentile([float(x) for x in supports], 0.90),
        "full_replay_ms": full_ms,
        "delta_apply_replay_ms": apply_ms,
        "delta_detect_replay_ms": detect_ms,
        "speedup_apply_only": full_ms / apply_ms,
        "speedup_detect_plus_apply": full_ms / detect_ms,
        "routed_replay_ms": min(full_ms, detect_ms),
        "route": "delta" if detect_ms < full_ms else "full",
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
    }
    del weight, columns, y0, reference, candidate
    gc.collect()
    return result


def summarize_modules(rows: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    by_layer: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_type[row["module_type"]].append(row)
        by_layer[row["layer"]].append(row)

    def summarize(group: list[dict]) -> dict:
        full = sum(x["full_replay_ms"] for x in group)
        delta = sum(x["delta_detect_replay_ms"] for x in group)
        routed = sum(x["routed_replay_ms"] for x in group)
        return {
            "modules": len(group),
            "full_replay_ms": full,
            "all_delta_replay_ms": delta,
            "routed_replay_ms": routed,
            "all_delta_speedup": full / delta,
            "routed_speedup": full / routed,
            "delta_route_modules": sum(x["route"] == "delta" for x in group),
            "support_fraction_median": statistics.median(
                x["support_median"] / max(1, x["input_width"]) for x in group
            ),
        }

    return {
        "all": summarize(rows),
        "by_module_type": {key: summarize(value) for key, value in sorted(by_type.items())},
        "by_layer": {str(key): summarize(value) for key, value in sorted(by_layer.items())},
        "fastest_modules": sorted(rows, key=lambda x: x["speedup_detect_plus_apply"], reverse=True)[:20],
        "slowest_modules": sorted(rows, key=lambda x: x["speedup_detect_plus_apply"])[:20],
    }


def main() -> None:
    torch.set_grad_enabled(False)
    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = PROMPT
    encoded = tokenizer(text, return_tensors="pt")
    prompt_length = int(encoded.input_ids.shape[1])

    generation_start = time.perf_counter_ns()
    generated = model.generate(
        **encoded,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    generation_ms = (time.perf_counter_ns() - generation_start) / 1e6
    generated_tokens = int(generated.shape[1] - prompt_length)

    captures: dict[str, torch.Tensor] = {}
    hooks = []
    layers = model.model.layers

    def capture(name: str):
        def hook(_module, args):
            captures[name] = args[0].detach().cpu().to(torch.float32).squeeze(0).contiguous()
        return hook

    for i, layer in enumerate(layers):
        hooks.append(layer.self_attn.q_proj.register_forward_pre_hook(capture(f"L{i}.attn_norm")))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(capture(f"L{i}.attn_out")))
        hooks.append(layer.mlp.gate_proj.register_forward_pre_hook(capture(f"L{i}.mlp_norm")))
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(capture(f"L{i}.mlp_hidden")))

    capture_start = time.perf_counter_ns()
    model(generated, attention_mask=torch.ones_like(generated), use_cache=False)
    capture_ms = (time.perf_counter_ns() - capture_start) / 1e6
    for handle in hooks:
        handle.remove()

    start_index = max(0, prompt_length - 1)
    source_to_modules: dict[str, list[tuple[str, torch.nn.Linear]]] = defaultdict(list)
    for i, layer in enumerate(layers):
        source_to_modules[f"L{i}.attn_norm"].extend([
            ("q_proj", layer.self_attn.q_proj),
            ("k_proj", layer.self_attn.k_proj),
            ("v_proj", layer.self_attn.v_proj),
        ])
        source_to_modules[f"L{i}.attn_out"].append(("o_proj", layer.self_attn.o_proj))
        source_to_modules[f"L{i}.mlp_norm"].extend([
            ("gate_proj", layer.mlp.gate_proj),
            ("up_proj", layer.mlp.up_proj),
        ])
        source_to_modules[f"L{i}.mlp_hidden"].append(("down_proj", layer.mlp.down_proj))

    result = {
        "schema": "smollm2-real-temporal-replay/v1",
        "model": MODEL_ID,
        "model_class": type(model).__name__,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "layers": len(layers),
        "prompt_tokens": prompt_length,
        "generated_tokens": generated_tokens,
        "total_sequence_tokens": int(generated.shape[1]),
        "replay_transitions": int(generated.shape[1] - start_index - 1),
        "generation_ms": generation_ms,
        "generation_tokens_per_second": generated_tokens / (generation_ms / 1000.0),
        "full_capture_forward_ms": capture_ms,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers_model": MODEL_ID,
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "platform": platform.platform(),
        },
        "decoded_tail": tokenizer.decode(generated[0, -min(64, generated.shape[1]):], skip_special_tokens=True),
        "bits": {},
    }

    for bits in BITS:
        module_rows: list[dict] = []
        source_rows: list[dict] = []
        total_quantize_ms = 0.0
        for source_name in sorted(source_to_modules):
            x = captures[source_name][start_index:].contiguous()
            if len(x) < 2:
                continue
            qmax = (1 << (bits - 1)) - 1
            max_abs = float(x.abs().max())
            scale = max(max_abs / max(qmax, 1), torch.finfo(torch.float32).tiny)

            def quantize_trace():
                return torch.round(x / scale).clamp(-qmax, qmax).to(torch.int16)

            quant_ms = median_ms(quantize_trace)
            codes = quantize_trace()
            decoded = codes.to(torch.float32) * scale
            metrics = source_metrics(x, codes, decoded)
            metrics.update({
                "source": source_name,
                "layer": int(source_name.split(".")[0][1:]),
                "source_type": source_name.split(".")[1],
                "bits": bits,
                "quantize_full_trace_ms": quant_ms,
            })
            source_rows.append(metrics)
            total_quantize_ms += quant_ms

            for module_type, module in source_to_modules[source_name]:
                row = replay_linear(module.weight, module.bias, decoded, codes, scale)
                row.update({
                    "source": source_name,
                    "layer": int(source_name.split(".")[0][1:]),
                    "module_type": module_type,
                    "bits": bits,
                })
                module_rows.append(row)

            del x, codes, decoded
            gc.collect()

        summary = summarize_modules(module_rows)
        transitions = max(1, result["replay_transitions"])
        summary["sources"] = len(source_rows)
        summary["quantize_all_sources_ms"] = total_quantize_ms
        summary["full_linear_ms_per_transition"] = summary["all"]["full_replay_ms"] / transitions
        summary["routed_linear_ms_per_transition"] = summary["all"]["routed_replay_ms"] / transitions
        summary["routed_plus_quantization_ms_per_transition"] = (
            summary["all"]["routed_replay_ms"] + total_quantize_ms
        ) / transitions
        summary["linear_speedup_routed"] = summary["all"]["routed_speedup"]
        summary["linear_speedup_routed_plus_quantization"] = (
            summary["all"]["full_replay_ms"] /
            (summary["all"]["routed_replay_ms"] + total_quantize_ms)
        )
        summary["source_geometry_by_type"] = {}
        grouped_sources: dict[str, list[dict]] = defaultdict(list)
        for row in source_rows:
            grouped_sources[row["source_type"]].append(row)
        for source_type, rows in sorted(grouped_sources.items()):
            summary["source_geometry_by_type"][source_type] = {
                "sources": len(rows),
                "width_median": statistics.median(x["width"] for x in rows),
                "code_change_fraction_median": statistics.median(x["code_change_fraction_median"] for x in rows),
                "code_change_fraction_p90_median": statistics.median(x["code_change_fraction_p90"] for x in rows),
                "delta_rank_99_median": statistics.median(x["delta_rank_99"] for x in rows),
                "delta_rank_999_median": statistics.median(x["delta_rank_999"] for x in rows),
                "quantized_relative_l2_error_median": statistics.median(x["quantized_relative_l2_error"] for x in rows),
            }
        summary["source_examples"] = source_rows[:12]
        result["bits"][str(bits)] = summary

    result["elapsed_seconds"] = time.time() - started
    print("RESULT_JSON=" + json.dumps(result, separators=(",", ":"), sort_keys=True), flush=True)
    print(json.dumps({
        "model": MODEL_ID,
        "generated_tokens_per_second": result["generation_tokens_per_second"],
        "bits": {
            bits: {
                "linear_speedup": data["linear_speedup_routed_plus_quantization"],
                "full_linear_ms_per_transition": data["full_linear_ms_per_transition"],
                "temporal_linear_ms_per_transition": data["routed_plus_quantization_ms_per_transition"],
            }
            for bits, data in result["bits"].items()
        },
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
