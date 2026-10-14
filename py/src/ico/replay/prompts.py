"""Synthetic prompts with the recorded lengths and the recorded prefix sharing.

A prompt is sent as token ids, not text, so its length in tokens is exact by
construction and no chat template can change it. The ids are ordinary tokens
of the model's own tokenizer, in random order: prefill and decode cost depend
on how many tokens there are, not on what they say.

Shared prefixes are rebuilt from the gateway's prefix hashes. Each hash stands
for a block of text; here it becomes a run of tokens, so two requests whose
first n hashes match get the same first n runs, and the engine's prefix cache
sees the same sharing it saw in production.

The sample is replayed many times over, and each pass stands for new traffic
of the same shape. Text that many conversations start with, such as a system
prompt, would be in the cache for new traffic too, so its runs are the same
on every pass. A conversation's own history would not be: its runs are drawn
afresh on every pass, and shared only between that conversation's turns within
it. Otherwise each pass would find the whole of every prompt cached from the
pass before, and the engine would look as if it never had to prefill.
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
        self._common = common_blocks(requests)

    def __len__(self) -> int:
        return len(self._requests)

    def prompt(self, index: int, use: int) -> Prompt:
        """The prompt for request ``index`` on its ``use``-th replay.

        ``index`` may run past the sample: request ``index`` is then the
        sample's ``index % len`` on a further pass. Blocks common to several
        conversations are the same on every pass; a conversation's own blocks
        are the same within a pass and new on the next; the rest of the prompt
        is new every time.
        """
        n = len(self._requests)
        r = self._requests[index % n]
        this_pass = f"{use}:{index // n}"
        tokens: list[int] = []
        for h in r.prefix_hashes:
            seed = h if h in self._common else f"{this_pass}:{h}"
            block = random.Random(seed).choices(self._pool, k=self._block_tokens[h])
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


def common_blocks(requests: Sequence[Recorded]) -> frozenset[int]:
    """The prefix hashes several conversations have in common.

    The hashes are chained: each one stands for its block and everything
    before it, so the requests' hash lists form a tree of prefixes. One
    conversation is a single line through it, each turn repeating the last
    and adding to it. Where the tree branches, different conversations went
    on from the same text, so a branch point and everything above it is
    common; below the last branch point, a block belongs to one line only.
    """
    children: dict[int, set[int]] = {}
    parent: dict[int, int | None] = {}
    for r in requests:
        above = None
        for h in r.prefix_hashes:
            parent.setdefault(h, above)
            if above is not None:
                children.setdefault(above, set()).add(h)
            above = h
    common: set[int] = set()
    for h, below in children.items():
        if len(below) < 2:
            continue
        node: int | None = h
        while node is not None and node not in common:
            common.add(node)
            node = parent[node]
    return frozenset(common)
