# Enterprise_LLM_Systems
An enterprise-grade LLM systems engineering project focused on scalable model serving, distributed training concepts, inference optimization, KV caching, speculative decoding, Flash Attention, performance benchmarking, and hardware-aware deployment.
# Problem statement
Enterprise applications require LLMs that can handle increasing workloads while maintaining acceptable latency, throughput, resource utilization, and operational costs.
Deploying an LLM is not simply about loading a model and generating responses.
# Production environments must address:
  - High inference latency.
  - GPU and memory utilization.
  - Concurrent requests.
  - Long-context processing.
  - Expensive model serving.
  - Scalability and reliability.
  - Hardware limitations.
# Proposed solution
  Build an enterprise LLM inference and optimization platform.
  The project will demonstrate how LLMs can be served, optimized, benchmarked, and monitored.

# Enterprise LLM inference lab

A local-first implementation of the Lecture 5 roadmap. The default backend is a **synthetic, untrained toy model** for studying algorithms; it does not produce useful prose and its timings do not predict LLM serving performance. A local Hugging Face causal model can be selected for real inference and benchmarking.

## Quick start

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
PYTHONPATH=src .venv/bin/python -m pytest -q
PYTHONPATH=src .venv/bin/uvicorn enterprise_llm.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/generate -H 'content-type: application/json' -d '{"prompt":"hello","max_new_tokens":16}'
curl -s http://127.0.0.1:8000/metrics
```

The service starts with the toy model. Set `LLM_MODEL_ID` to a cached local model path or Hugging Face model ID to use Transformers. Install `.[ml]` first. Downloads are disabled in the API by default; populate the model cache beforehand or use a local path. The CLI permits a one-time download with `--allow-download`.

## Lecture 5 experiments

| Phase | Command | What it measures |
| --- | --- | --- |
| 5.1 Distributed training | `llm-lab parallelism` | Analytical data, tensor, and pipeline parallelism formulas; no training |
| 5.2 Inference optimization | `llm-lab benchmark --model /path/to/model` and `llm-lab quantization` | Real model load/latency/throughput; synthetic INT8 memory and error experiment |
| 5.3 KV caching | `llm-lab kv --tokens 32` | Cached and uncached outputs, time, and attention work |
| 5.4 Speculative decoding | `llm-lab speculative --draft-width 4` | Greedy correctness and draft acceptance; sequential target verification, no claimed speedup |
| 5.5 Attention | `llm-lab attention --device cpu` | PyTorch math SDPA against default dispatcher; requires `.[ml]` |
| 5.6 Hardware | `llm-lab memory --parameters 1000000000 --layers 24 --kv-heads 8 --head-dim 64 --context 4096 --concurrency 4` | Lower bound for weights and KV cache |
| 5.7 Integration | FastAPI routes `/v1/generate`, `/health`, `/metrics` | Bounded concurrency, queue timeout, and Prometheus text metrics |

For real model benchmarking:

```bash
.venv/bin/python -m pip install -e '.[ml]'
llm-lab benchmark --model /path/to/local-causal-model --device mps --requests 8 --concurrency 2
llm-lab benchmark --model /path/to/local-causal-model --device mps --requests 8 --concurrency 2 --no-cache
```

Compare runs with the same model, prompt, token count, device, process state, and concurrency. `requests_per_second` is completed requests divided by wall time; `output_tokens_per_second` sums generated tokens divided by wall time. Load time is reported separately. Supply `--hourly-cost` in your currency to estimate cost per million output tokens from measured throughput. The runtime serializes calls to one model instance to avoid unsafe concurrent use; queueing still affects end-to-end latency in the API. The memory estimator excludes activations, framework overhead, and operating-system memory.

The `attention` command reports an explicit CUDA Flash timing only when that kernel succeeds. Default SDPA dispatch is not necessarily Flash Attention. On Apple silicon, test CPU and MPS backends without treating either as a CUDA Flash Attention benchmark.

## Serving configuration

- `LLM_MODEL_ID`: local model path or already cached model ID; unset uses toy backend.
- `LLM_MAX_CONCURRENCY`: number of active inference jobs, default 2.
- `LLM_MAX_QUEUE`: waiting request capacity, default 16.
- `LLM_QUEUE_TIMEOUT`: seconds to wait for admission or an active slot, default 30.

The API caps prompts at 16,000 characters and generation at 512 tokens. `/metrics` exposes request, failure, rejection, token, active request, and latency histogram series. Logs and metrics do not include prompt or response contents.

## Docker

```bash
docker build -t enterprise-llm-lab .
docker run --rm -p 8000:8000 enterprise-llm-lab
```

The Docker image installs only the API and toy backend. For a real model image, install the ML extra, supply model files, and size the container for its weights and KV cache. A running Docker daemon is required.

## Verification on the reference Mac

On 13 October 2026, this project passed 25 pytest checks with Python 3.13.14 and the ML extra installed. One check creates a tiny random-weight local Transformers model and verifies cached and uncached generation agree. PyTorch 2.14.1 ran the attention comparison at sequence length 128 on both CPU and MPS; the two outputs matched within the configured tolerance. The Docker image built and its `/health`, `/v1/generate`, and `/metrics` routes passed a live smoke test through Colima. MPS may be unavailable to a sandboxed process even when it is available on the host.

No CUDA Flash benchmark or real model benchmark was run on this Mac. The small attention timings in the local run are diagnostic only; use longer runs and representative shapes for performance decisions.

## Scope

This is an instructional platform. The parallelism and quantization phases are small simulations, and the speculative decoder verifies target tokens sequentially. To demonstrate production speedups, use an optimized serving engine and accelerator-specific kernels in a compatible environment, then benchmark under realistic request mixes and service-level targets.
