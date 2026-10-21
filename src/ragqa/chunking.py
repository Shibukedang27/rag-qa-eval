"""Split passages into overlapping word windows and label which chunks hold each answer."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .data import Passage, Question

_WORD = re.compile(r"\S+")


@dataclass(frozen=True)
class Chunk:
    cid: int
    pid: str
    title: str
    text: str
    start: int      # character offsets of the chunk inside its passage
    end: int


def chunk_passage(passage: Passage, size: int, overlap: int, first_id: int = 0) -> list[Chunk]:
    """Windows of `size` words with `overlap` words shared between neighbours.

    Offsets are kept so a gold answer span can be mapped to the chunk(s) that contain it.
    A passage of `size` words or fewer gives a single chunk. The last window may be shorter
    than `size`; it still starts `size - overlap` words after the previous one.
    """
    if not 0 <= overlap < size:
        raise ValueError("need 0 <= overlap < size")
    spans = [(m.start(), m.end()) for m in _WORD.finditer(passage.text)]
    if not spans:
        return []
    stride = size - overlap
    starts = [0]
    while starts[-1] + size < len(spans):
        starts.append(starts[-1] + stride)
    chunks = []
    for w0 in starts:
        w1 = min(w0 + size, len(spans))
        c0, c1 = spans[w0][0], spans[w1 - 1][1]
        chunks.append(
            Chunk(cid=first_id + len(chunks), pid=passage.pid, title=passage.title,
                  text=passage.text[c0:c1], start=c0, end=c1)
        )
    return chunks


def chunk_corpus(passages: list[Passage], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for p in passages:
        chunks.extend(chunk_passage(p, size, overlap, first_id=len(chunks)))
    return chunks


def gold_chunks(question: Question, chunks_by_pid: dict[str, list[Chunk]]) -> set[int]:
    """Chunks of the question's own paragraph that fully contain at least one gold answer.

    This is a strict label: the same answer string may also appear in other paragraphs,
    but only the annotated span counts as evidence.
    """
    gold = set()
    for c in chunks_by_pid.get(question.pid, []):
        if any(c.start <= a.start and a.end <= c.end for a in question.answers):
            gold.add(c.cid)
    return gold


def index_by_pid(chunks: list[Chunk]) -> dict[str, list[Chunk]]:
    out: dict[str, list[Chunk]] = {}
    for c in chunks:
        out.setdefault(c.pid, []).append(c)
    return out
