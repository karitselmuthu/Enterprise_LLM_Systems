"""Small deterministic causal attention model for algorithm demonstrations.

This model has no trained weights and must not be used as a quality or production
performance benchmark. It makes cache and decoding mechanics observable on a CPU.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

ALPHABET = "abcdefghijklmnopqrstuvwxyz .,!?"
BOS = len(ALPHABET)
VOCAB_SIZE = BOS + 1


def encode(text: str) -> list[int]:
    return [ALPHABET.find(char.lower()) if char.lower() in ALPHABET else ALPHABET.index(" ") for char in text]


def decode(tokens: list[int]) -> str:
    return "".join(ALPHABET[token] for token in tokens if token != BOS)


@dataclass
class Stats:
    attention_pairs: int = 0
    forward_calls: int = 0
    draft_proposals: int = 0
    accepted_draft_tokens: int = 0


class ToyModel:
    def __init__(self, seed: int = 1):
        self.seed = seed

    def _vectors(self, token: int, position: int) -> tuple[list[float], list[float], list[float]]:
        phase = token * 0.29 + position * 0.07 + self.seed * 0.13
        q = [math.sin(phase + i * 0.53) for i in range(8)]
        k = [math.cos(phase * 0.71 + i * 0.43) for i in range(8)]
        v = [math.sin(phase * 1.17 + i * 0.31) for i in range(8)]
        return q, k, v

    def next_token(self, tokens: list[int], stats: Stats, cache: "KVCache | None" = None) -> int:
        if not tokens:
            tokens = [BOS]
        if cache is None:
            vectors = [self._vectors(token, i) for i, token in enumerate(tokens)]
            # A full causal forward pass calculates attention for each position.
            stats.attention_pairs += len(tokens) * (len(tokens) + 1) // 2
            query = vectors[-1][0]
            keys = [item[1] for item in vectors]
            values = [item[2] for item in vectors]
        else:
            if cache.length > len(tokens) or cache.tokens != tokens[:cache.length]:
                raise ValueError("KV cache does not match the token prefix")
            for position in range(cache.length, len(tokens)):
                _, key, value = self._vectors(tokens[position], position)
                cache.keys.append(key)
                cache.values.append(value)
                cache.tokens.append(tokens[position])
                stats.attention_pairs += position + 1
            query = self._vectors(tokens[-1], len(tokens) - 1)[0]
            keys, values = cache.keys, cache.values
        stats.forward_calls += 1
        scores = [sum(a * b for a, b in zip(query, key)) / math.sqrt(8) for key in keys]
        peak = max(scores)
        weights = [math.exp(score - peak) for score in scores]
        total = sum(weights)
        context = [sum(weight * value[i] for weight, value in zip(weights, values)) / total for i in range(8)]
        last = tokens[-1]
        # A fixed transition term keeps the output deterministic and draft/target distinct.
        logits = [sum(context[i] * math.sin(candidate * 0.17 + i * 0.49 + self.seed * 0.2)
                      for i in range(8)) + 0.15 * math.cos(last * 0.4 + candidate * 0.3 + self.seed)
                  for candidate in range(BOS)]
        return max(range(BOS), key=lambda candidate: logits[candidate])


@dataclass
class KVCache:
    tokens: list[int]
    keys: list[list[float]]
    values: list[list[float]]

    def __init__(self) -> None:
        self.tokens, self.keys, self.values = [], [], []

    @property
    def length(self) -> int:
        return len(self.tokens)


def generate(prompt: str, max_new_tokens: int = 32, *, cached: bool = True,
             model: ToyModel | None = None) -> tuple[str, Stats]:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    model = model or ToyModel()
    tokens = encode(prompt) or [BOS]
    stats = Stats()
    cache = KVCache() if cached else None
    output: list[int] = []
    for _ in range(max_new_tokens):
        token = model.next_token(tokens, stats, cache)
        tokens.append(token)
        output.append(token)
    return decode(output), stats


def speculative_generate(prompt: str, max_new_tokens: int = 32, draft_width: int = 4,
                         *, target: ToyModel | None = None, draft: ToyModel | None = None) -> tuple[str, Stats]:
    """Exact greedy output through draft proposals and target verification.

    Target verification is sequential here; this demonstrates correctness and
    acceptance, not a speculative decoding speedup. A batched target forward pass
    is needed to gain throughput on an actual accelerator.
    """
    if max_new_tokens < 1 or draft_width < 1:
        raise ValueError("max_new_tokens and draft_width must be positive")
    target, draft = target or ToyModel(), draft or ToyModel(seed=2)
    prefix = encode(prompt) or [BOS]
    result: list[int] = []
    stats = Stats()
    target_cache = KVCache()
    while len(result) < max_new_tokens:
        proposal_prefix = prefix.copy()
        proposals = []
        draft_cache = KVCache()
        for _ in range(min(draft_width, max_new_tokens - len(result))):
            proposed = draft.next_token(proposal_prefix, stats, draft_cache)
            proposals.append(proposed)
            proposal_prefix.append(proposed)
            stats.draft_proposals += 1
        for proposed in proposals:
            actual = target.next_token(prefix, stats, target_cache)
            result.append(actual)
            prefix.append(actual)
            if actual != proposed:
                break
            stats.accepted_draft_tokens += 1
    return decode(result), stats
