"""Synthetic prompts with the recorded lengths and the recorded prefix sharing.

A prompt is sent as token ids, not text, so its length in tokens is exact by
construction and no chat template can change it. The ids are ordinary tokens
of the model's own tokenizer, in random order: prefill and decode cost depend
on how many tokens there are, not on what they say.

Shared prefixes are rebuilt from the gateway's prefix hashes. Each hash stands
for a block of text; here it becomes a fixed run of tokens, so two requests
whose first n hashes match get the same first n runs, and the engine's prefix
cache sees the same sharing it saw in production.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from ico.replay.workload import Recorded

# Tokenized once to learn which token ids are ordinary words in this model.
POOL_TEXT = (
    "the quick brown fox jumps over a lazy dog while seven wizards quietly judge twelve boxing matches "
    "people often say that time and tide wait for no one yet every morning brings another chance to begin "
    "a small river runs through the old town past the market the school and the long stone bridge "
    "numbers such as one two three four five six seven eight nine and ten are easy to count and hard to forget "
    "she opened the window looked at the garden and thought about what to plant when the rain finally stopped"
)


@dataclass(frozen=True)
class Prompt:
    tokens: tuple[int, ...]
    max_tokens: int


class Synthesizer:
    def __init__(self, pool: Sequence[int], requests: Sequence[Recorded], block_chars: int) -> None:
        if len(set(pool)) < 16:
            raise ValueError("the tokenizer returned too few distinct tokens to build prompts from")
        self._pool = sorted(set(pool))
        self._requests = requests
        # The token length of each shared block, fixed by the first request
        # that has it: block_chars characters at that request's tokens per character.
        self._block_tokens: dict[int, int] = {}
        for r in requests:
            per_char = r.prompt_tokens / r.prompt_chars if r.prompt_chars else 0.0
            for h in r.prefix_hashes:
                self._block_tokens.setdefault(h, max(1, round(block_chars * per_char)))

    def __len__(self) -> int:
        return len(self._requests)

    def prompt(self, index: int, use: int) -> Prompt:
        """The prompt for request ``index`` on its ``use``-th replay.

        The shared part is the same on every use. The rest is drawn afresh
        each time, so replaying the sample twice does not turn every prompt
        into a prefix-cache hit of its earlier self.
        """
        r = self._requests[index % len(self._requests)]
        tokens: list[int] = []
        for h in r.prefix_hashes:
            block = random.Random(h).choices(self._pool, k=self._block_tokens[h])
            if len(tokens) + len(block) > r.prompt_tokens:
                break
            tokens += block
        tail = random.Random(f"{index}:{use}")
        tokens += tail.choices(self._pool, k=r.prompt_tokens - len(tokens))
        return Prompt(tuple(tokens), r.completion_tokens)

    def shared_tokens(self) -> float:
        """Share of prompt tokens that repeat an earlier prompt's start, replaying the sample once in order."""
        seen: set[int] = set()
        shared = total = 0
        for r in self._requests:
            run = 0
            still_shared = True
            for h in r.prefix_hashes:
                if run + self._block_tokens[h] > r.prompt_tokens:
                    break
                run += self._block_tokens[h]
                still_shared = still_shared and h in seen
                if still_shared:
                    shared += self._block_tokens[h]
                seen.add(h)
            total += r.prompt_tokens
        return shared / total if total else 0.0
