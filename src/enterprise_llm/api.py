"""Bounded FastAPI service with basic Prometheus-compatible metrics."""
from __future__ import annotations

import asyncio
import os
import threading
import time

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from .runtime import Runtime


class GenerateRequest(BaseModel):
    prompt: str = Field(max_length=16000)
    max_new_tokens: int = Field(default=32, ge=1, le=512)
    use_cache: bool = True


class GenerateResponse(BaseModel):
    text: str
    output_tokens: int
    elapsed_seconds: float
    backend: str


class Metrics:
    BUCKETS = (0.1, 0.5, 1, 2, 5, 10, 30, 60)

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.requests = 0
        self.failures = 0
        self.rejected = 0
        self.output_tokens = 0
        self.latencies: list[float] = []
        self.active = 0

    def record(self, seconds: float, tokens: int, failed: bool = False) -> None:
        with self.lock:
            self.requests += 1
            self.failures += int(failed)
            self.output_tokens += tokens
            self.latencies.append(seconds)

    def render(self) -> str:
        with self.lock:
            lines = [
                "# TYPE llm_requests_total counter", f"llm_requests_total {self.requests}",
                "# TYPE llm_failures_total counter", f"llm_failures_total {self.failures}",
                "# TYPE llm_rejected_total counter", f"llm_rejected_total {self.rejected}",
                "# TYPE llm_output_tokens_total counter", f"llm_output_tokens_total {self.output_tokens}",
                "# TYPE llm_active_requests gauge", f"llm_active_requests {self.active}",
                "# TYPE llm_request_duration_seconds histogram",
            ]
            for bucket in self.BUCKETS:
                lines.append(f'llm_request_duration_seconds_bucket{{le="{bucket}"}} '
                             f'{sum(t <= bucket for t in self.latencies)}')
            lines.extend([f'llm_request_duration_seconds_bucket{{le="+Inf"}} {len(self.latencies)}',
                          f"llm_request_duration_seconds_sum {sum(self.latencies)}",
                          f"llm_request_duration_seconds_count {len(self.latencies)}"])
            return "\n".join(lines) + "\n"


def create_app(runtime: Runtime | None = None, max_concurrency: int | None = None,
               max_queue: int | None = None, queue_timeout: float | None = None) -> FastAPI:
    runtime = runtime or Runtime()
    max_concurrency = max_concurrency or int(os.getenv("LLM_MAX_CONCURRENCY", "2"))
    max_queue = max_queue if max_queue is not None else int(os.getenv("LLM_MAX_QUEUE", "16"))
    queue_timeout = queue_timeout if queue_timeout is not None else float(os.getenv("LLM_QUEUE_TIMEOUT", "30"))
    if max_concurrency < 1 or max_queue < 0 or queue_timeout <= 0:
        raise ValueError("invalid request manager limits")
    metrics = Metrics()
    active_slots = asyncio.Semaphore(max_concurrency)
    pending_lock = threading.Lock()
    pending = 0
    app = FastAPI(title="Enterprise LLM inference lab", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": runtime.backend, "model_loaded": not runtime.model_id or runtime.model is not None}

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    @app.post("/v1/generate", response_model=GenerateResponse)
    async def generate_endpoint(request: GenerateRequest) -> GenerateResponse:
        nonlocal pending
        with pending_lock:
            if pending >= max_concurrency + max_queue:
                with metrics.lock:
                    metrics.rejected += 1
                raise HTTPException(status_code=503, detail="request queue is full")
            pending += 1
        active_acquired = False
        start = time.perf_counter()
        try:
            await asyncio.wait_for(active_slots.acquire(), timeout=queue_timeout)
            active_acquired = True
            with metrics.lock:
                metrics.active += 1
            result = await asyncio.to_thread(runtime.generate, request.prompt,
                                             request.max_new_tokens, request.use_cache)
            metrics.record(time.perf_counter() - start, result.output_tokens)
            return GenerateResponse(**result.__dict__)
        except asyncio.TimeoutError:
            with metrics.lock:
                metrics.rejected += 1
            raise HTTPException(status_code=503, detail="request timed out in queue")
        except Exception as exc:
            metrics.record(time.perf_counter() - start, 0, failed=True)
            raise HTTPException(status_code=500, detail=f"inference failed: {type(exc).__name__}") from exc
        finally:
            if active_acquired:
                active_slots.release()
                with metrics.lock:
                    metrics.active -= 1
            with pending_lock:
                pending -= 1

    return app


app = create_app()
