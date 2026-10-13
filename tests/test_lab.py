import pytest

from enterprise_llm.experiments import (kv_benchmark, memory_estimate,
                                        parallelism_simulation, quantization_benchmark,
                                        speculative_benchmark)
from enterprise_llm.toy import KVCache, Stats, ToyModel, generate, speculative_generate


@pytest.mark.parametrize("prompt", ["", "enterprise", "A longer prompt, with punctuation!"])
@pytest.mark.parametrize("tokens", [1, 8, 24])
def test_cached_output_matches_uncached(prompt, tokens):
    cached, _ = generate(prompt, tokens, cached=True)
    uncached, _ = generate(prompt, tokens, cached=False)
    assert cached == uncached


@pytest.mark.parametrize("prompt", ["", "enterprise", "Another input!"])
@pytest.mark.parametrize("width", [1, 4, 8])
def test_speculative_matches_greedy(prompt, width):
    expected, _ = generate(prompt, 19)
    actual, stats = speculative_generate(prompt, 19, width)
    assert actual == expected
    assert stats.draft_proposals >= 19


def test_kv_cache_reduces_attention_work():
    result = kv_benchmark("enterprise systems", 16, 1)
    assert result["outputs_equal"]
    assert result["cached"]["attention_pairs"] < result["uncached"]["attention_pairs"]


def test_cache_rejects_wrong_prefix():
    model, cache = ToyModel(), KVCache()
    model.next_token([1, 2], Stats(), cache)
    with pytest.raises(ValueError):
        model.next_token([3, 2], Stats(), cache)


def test_quantization_reports_tradeoff():
    result = quantization_benchmark(16, 2)
    assert result["int8_weight_bytes_theoretical"] < result["fp32_weight_bytes_theoretical"]
    assert result["mean_absolute_error"] < 0.1


def test_parallelism_and_memory_formulas():
    assert parallelism_simulation(microbatches=8, stages=4)["pipeline_parallel"]["ideal_utilization"] == 8 / 11
    result = memory_estimate(1000, 2, 2, 4, 10, 1)
    assert result["kv_cache_gib"] > 0


def test_api_generate_and_metrics():
    from fastapi.testclient import TestClient
    from enterprise_llm.api import create_app
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/v1/generate", json={"prompt": "test", "max_new_tokens": 4})
        assert response.status_code == 200
        assert response.json()["output_tokens"] == 4
        assert "llm_requests_total 1" in client.get("/metrics").text
        assert client.post("/v1/generate", json={"prompt": "test", "max_new_tokens": 0}).status_code == 422


def test_queue_capacity_rejects_excess_requests():
    import asyncio
    import threading
    from httpx import ASGITransport, AsyncClient
    from enterprise_llm.api import create_app
    from enterprise_llm.runtime import Generation, Runtime

    gate = threading.Event()
    entered = threading.Event()

    class SlowRuntime(Runtime):
        def generate(self, prompt, max_new_tokens=32, use_cache=True):
            entered.set()
            gate.wait(timeout=3)
            return Generation("x", 1, 0.01, "toy")

    async def run():
        app = create_app(SlowRuntime(), max_concurrency=1, max_queue=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = asyncio.create_task(client.post("/v1/generate", json={"prompt": "a"}))
            for _ in range(100):
                if entered.is_set():
                    break
                await asyncio.sleep(.01)
            assert entered.is_set()
            second = await client.post("/v1/generate", json={"prompt": "b"})
            gate.set()
            assert second.status_code == 503
            assert (await first).status_code == 200
            assert "llm_rejected_total 1" in (await client.get("/metrics")).text

    asyncio.run(run())


def test_transformers_runtime_with_local_tiny_model(tmp_path):
    pytest.importorskip("torch")
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    from enterprise_llm.runtime import Runtime

    vocab = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3, "world": 4}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tokenizer,
        unk_token="[UNK]", bos_token="[BOS]", eos_token="[EOS]", pad_token="[EOS]")
    fast.save_pretrained(tmp_path)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(
        vocab_size=len(vocab), n_positions=32, n_ctx=32, n_embd=16, n_layer=1, n_head=2,
        bos_token_id=1, eos_token_id=2, pad_token_id=2))
    model.save_pretrained(tmp_path)
    runtime = Runtime(str(tmp_path), device="cpu")
    cached = runtime.generate("hello world", 3, use_cache=True)
    uncached = runtime.generate("hello world", 3, use_cache=False)
    assert runtime.model is not None
    assert cached.backend == "transformers"
    assert cached.text == uncached.text
    assert cached.output_tokens == uncached.output_tokens
