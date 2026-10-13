"""Repeatable experiments for the seven lecture phases."""
from __future__ import annotations

import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from .runtime import Runtime
from .toy import generate, speculative_generate


def parallelism_simulation(samples: int = 64, workers: int = 4, microbatches: int = 8,
                           stages: int = 4) -> dict:
    if min(samples, workers, microbatches, stages) < 1:
        raise ValueError("all dimensions must be positive")
    return {
        "data_parallel": {"samples_per_worker": math.ceil(samples / workers),
                          "ideal_speedup_upper_bound": min(samples, workers)},
        "tensor_parallel": {"weight_shard_fraction": 1 / workers,
                            "collectives_per_layer_forward_lower_bound": 1,
                            "note": "Communication and memory bandwidth are omitted."},
        "pipeline_parallel": {"stages": stages, "microbatches": microbatches,
                              "ideal_utilization": microbatches / (microbatches + stages - 1),
                              "bubble_fraction": (stages - 1) / (microbatches + stages - 1)},
        "kind": "analytical simulation; no distributed training occurred",
    }


def kv_benchmark(prompt: str = "enterprise systems", max_new_tokens: int = 32,
                 repeats: int = 3) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    result = {}
    outputs = []
    for cached in (False, True):
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            output, stats = generate(prompt, max_new_tokens, cached=cached)
            times.append(time.perf_counter() - start)
        outputs.append(output)
        result["cached" if cached else "uncached"] = {
            "median_seconds": statistics.median(times), "attention_pairs": stats.attention_pairs,
            "forward_calls": stats.forward_calls, "output": output,
        }
    result["outputs_equal"] = outputs[0] == outputs[1]
    result["kind"] = "synthetic causal-attention algorithm benchmark"
    return result


def speculative_benchmark(prompt: str = "enterprise systems", max_new_tokens: int = 32,
                          draft_width: int = 4) -> dict:
    start = time.perf_counter()
    reference, _ = generate(prompt, max_new_tokens)
    reference_time = time.perf_counter() - start
    start = time.perf_counter()
    output, stats = speculative_generate(prompt, max_new_tokens, draft_width)
    speculative_time = time.perf_counter() - start
    return {"matches_greedy_target": output == reference, "output": output,
            "draft_proposals": stats.draft_proposals,
            "accepted_draft_tokens": stats.accepted_draft_tokens,
            "acceptance_rate": stats.accepted_draft_tokens / stats.draft_proposals,
            "reference_seconds": reference_time, "speculative_seconds": speculative_time,
            "kind": "synthetic correctness experiment; target verification is sequential"}


def serving_benchmark(runtime: Runtime, prompt: str, max_new_tokens: int = 16,
                      requests: int = 8, concurrency: int = 2, use_cache: bool = True,
                      hourly_cost: float | None = None) -> dict:
    if requests < 1 or concurrency < 1 or (hourly_cost is not None and hourly_cost < 0):
        raise ValueError("requests and concurrency must be positive; hourly cost must be nonnegative")
    load_seconds = runtime.load()
    runtime.generate(prompt, 1, use_cache=use_cache)  # warmup
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        generations = list(pool.map(lambda _: runtime.generate(prompt, max_new_tokens, use_cache), range(requests)))
    wall = time.perf_counter() - start
    latencies = sorted(g.elapsed_seconds for g in generations)
    output_tokens = sum(g.output_tokens for g in generations)
    return {"backend": runtime.backend, "model_id": runtime.model_id, "device": runtime.device,
            "load_seconds": load_seconds, "requests": requests, "concurrency": concurrency,
            "wall_seconds": wall, "requests_per_second": requests / wall,
            "output_tokens_per_second": output_tokens / wall,
            "cost_per_million_output_tokens": (hourly_cost * wall / 3600 / output_tokens * 1_000_000
                                               if hourly_cost is not None and output_tokens else None),
            "median_latency_seconds": statistics.median(latencies),
            "p95_latency_seconds": latencies[math.ceil(.95 * len(latencies)) - 1],
            "use_cache": use_cache,
            "kind": "real model benchmark" if runtime.model_id else "synthetic toy benchmark"}


def memory_estimate(parameters: int, layers: int, kv_heads: int, head_dim: int,
                    context_tokens: int, concurrent_requests: int, weight_bits: int = 16,
                    kv_bits: int = 16) -> dict:
    if min(parameters, layers, kv_heads, head_dim, context_tokens, concurrent_requests, weight_bits, kv_bits) < 1:
        raise ValueError("all dimensions must be positive")
    weights = parameters * weight_bits / 8
    kv = 2 * layers * kv_heads * head_dim * context_tokens * concurrent_requests * kv_bits / 8
    return {"weights_gib": weights / 2**30, "kv_cache_gib": kv / 2**30,
            "minimum_gib": (weights + kv) / 2**30,
            "note": "Lower bound only; excludes activations, runtime overhead, fragmentation, and OS memory."}


def attention_benchmark(sequence_length: int = 256, repeats: int = 10,
                        device: str = "cpu") -> dict:
    """Compare PyTorch math SDPA with the default dispatcher on available hardware."""
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if sequence_length < 1 or repeats < 1:
        raise ValueError("sequence_length and repeats must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable")
    q = torch.randn((1, 4, sequence_length, 64), device=device)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
    def measure(math_only: bool) -> tuple[float, object]:
        times = []
        output = None
        for _ in range(repeats + 1):
            sync()
            start = time.perf_counter()
            if math_only:
                with sdpa_kernel(SDPBackend.MATH):
                    output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            else:
                output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            sync()
            times.append(time.perf_counter() - start)
        return statistics.median(times[1:]), output
    math_time, math_output = measure(True)
    default_time, default_output = measure(False)
    flash_time = None
    flash_matches = None
    if device == "cuda" and torch.backends.cuda.is_flash_attention_available():
        try:
            times = []
            for _ in range(repeats + 1):
                sync()
                start = time.perf_counter()
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    flash_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                sync()
                times.append(time.perf_counter() - start)
            flash_time = statistics.median(times[1:])
            flash_matches = bool(torch.allclose(math_output, flash_output, atol=1e-4, rtol=1e-4))
        except RuntimeError:
            pass  # This shape or dtype has no eligible Flash kernel.
    return {"device": device, "sequence_length": sequence_length,
            "math_median_seconds": math_time, "default_median_seconds": default_time,
            "outputs_close": bool(torch.allclose(math_output, default_output, atol=1e-4, rtol=1e-4)),
            "flash_median_seconds": flash_time, "flash_outputs_close": flash_matches,
            "note": "Flash timing is present only when an explicitly selected CUDA Flash kernel succeeds."}


def quantization_benchmark(width: int = 128, repeats: int = 20) -> dict:
    """Demonstrate per-row symmetric INT8 weight quantization on a matrix-vector product."""
    if width < 1 or repeats < 1:
        raise ValueError("width and repeats must be positive")
    weights = [[math.sin(row * .13 + col * .07) for col in range(width)] for row in range(width)]
    vector = [math.cos(col * .11) for col in range(width)]
    scales = [max(abs(value) for value in row) / 127 or 1.0 for row in weights]
    quantized = [[max(-127, min(127, round(value / scales[i]))) for value in row]
                 for i, row in enumerate(weights)]
    def fp32() -> list[float]:
        return [sum(value * x for value, x in zip(row, vector)) for row in weights]
    def int8() -> list[float]:
        return [scale * sum(value * x for value, x in zip(row, vector))
                for scale, row in zip(scales, quantized)]
    def measure(fn) -> tuple[float, list[float]]:
        times = []
        output = []
        for _ in range(repeats):
            start = time.perf_counter()
            output = fn()
            times.append(time.perf_counter() - start)
        return statistics.median(times), output
    fp_time, expected = measure(fp32)
    int_time, actual = measure(int8)
    return {"width": width, "fp32_weight_bytes_theoretical": width * width * 4,
            "int8_weight_bytes_theoretical": width * width + width * 4,
            "mean_absolute_error": sum(abs(a - b) for a, b in zip(expected, actual)) / width,
            "fp32_median_seconds": fp_time, "int8_median_seconds": int_time,
            "kind": "synthetic quantization experiment; Python integer arithmetic is not an optimized kernel"}
