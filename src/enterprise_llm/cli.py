"""Command-line entry points for serving and experiments."""
from __future__ import annotations

import argparse
import json

from .experiments import (attention_benchmark, kv_benchmark, memory_estimate,
                          parallelism_simulation, quantization_benchmark, serving_benchmark, speculative_benchmark)
from .runtime import Runtime


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise LLM inference lab")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("parallelism")
    quant = sub.add_parser("quantization")
    quant.add_argument("--width", type=int, default=128)
    quant.add_argument("--repeats", type=int, default=20)
    kv = sub.add_parser("kv")
    kv.add_argument("--prompt", default="enterprise systems")
    kv.add_argument("--tokens", type=int, default=32)
    spec = sub.add_parser("speculative")
    spec.add_argument("--prompt", default="enterprise systems")
    spec.add_argument("--tokens", type=int, default=32)
    spec.add_argument("--draft-width", type=int, default=4)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--model", default=None, help="local path or Hugging Face model ID")
    bench.add_argument("--allow-download", action="store_true")
    bench.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    bench.add_argument("--prompt", default="Explain inference optimization in one sentence.")
    bench.add_argument("--tokens", type=int, default=16)
    bench.add_argument("--requests", type=int, default=8)
    bench.add_argument("--concurrency", type=int, default=2)
    bench.add_argument("--no-cache", action="store_true")
    bench.add_argument("--hourly-cost", type=float, default=None,
                       help="your all-in hardware cost per hour for estimated output-token cost")
    attention = sub.add_parser("attention")
    attention.add_argument("--sequence-length", type=int, default=256)
    attention.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    attention.add_argument("--repeats", type=int, default=10)
    memory = sub.add_parser("memory")
    memory.add_argument("--parameters", type=int, required=True)
    memory.add_argument("--layers", type=int, required=True)
    memory.add_argument("--kv-heads", type=int, required=True)
    memory.add_argument("--head-dim", type=int, required=True)
    memory.add_argument("--context", type=int, required=True)
    memory.add_argument("--concurrency", type=int, default=1)
    memory.add_argument("--weight-bits", type=int, default=16)
    memory.add_argument("--kv-bits", type=int, default=16)
    args = parser.parse_args()
    if args.command == "parallelism":
        result = parallelism_simulation()
    elif args.command == "quantization":
        result = quantization_benchmark(args.width, args.repeats)
    elif args.command == "kv":
        result = kv_benchmark(args.prompt, args.tokens)
    elif args.command == "speculative":
        result = speculative_benchmark(args.prompt, args.tokens, args.draft_width)
    elif args.command == "benchmark":
        result = serving_benchmark(Runtime(args.model, args.device, not args.allow_download),
                                   args.prompt, args.tokens, args.requests, args.concurrency,
                                   not args.no_cache, args.hourly_cost)
    elif args.command == "attention":
        result = attention_benchmark(args.sequence_length, args.repeats, args.device)
    else:
        result = memory_estimate(args.parameters, args.layers, args.kv_heads, args.head_dim,
                                 args.context, args.concurrency, args.weight_bits, args.kv_bits)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
