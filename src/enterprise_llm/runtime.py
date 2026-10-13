"""Optional real-model runtime. Models are loaded only when explicitly configured."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from time import perf_counter


@dataclass
class Generation:
    text: str
    output_tokens: int
    elapsed_seconds: float
    backend: str


class Runtime:
    def __init__(self, model_id: str | None = None, device: str = "auto", local_files_only: bool = True):
        self.model_id = model_id or os.getenv("LLM_MODEL_ID")
        self.device_preference = device
        self.local_files_only = local_files_only
        self._lock = threading.Lock()
        self.model = None
        self.tokenizer = None
        self.device = "cpu"

    @property
    def backend(self) -> str:
        return "transformers" if self.model_id else "toy"

    def load(self) -> float:
        if not self.model_id or self.model is not None:
            return 0.0
        with self._lock:
            if self.model is not None:
                return 0.0
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            started = perf_counter()
            if self.device_preference == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            else:
                self.device = self.device_preference
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, local_files_only=self.local_files_only,
                                                       trust_remote_code=False)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, local_files_only=self.local_files_only,
                                                         trust_remote_code=False)
            model.to(self.device)
            model.eval()
            self.tokenizer, self.model = tokenizer, model
            return perf_counter() - started

    def generate(self, prompt: str, max_new_tokens: int = 32, use_cache: bool = True) -> Generation:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        started = perf_counter()
        if not self.model_id:
            from .toy import generate
            output, _ = generate(prompt, max_new_tokens, cached=use_cache)
            return Generation(output, max_new_tokens, perf_counter() - started, "toy")
        self.load()
        import torch
        inputs = self.tokenizer(prompt or " ", return_tensors="pt").to(self.device)
        with self._lock, torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                         do_sample=False, use_cache=use_cache,
                                         pad_token_id=self.tokenizer.eos_token_id)
        new_ids = output[0, inputs["input_ids"].shape[1]:]
        return Generation(self.tokenizer.decode(new_ids, skip_special_tokens=True),
                          len(new_ids), perf_counter() - started, "transformers")
