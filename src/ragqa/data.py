"""Download, verify and flatten the SQuAD 2.0 dev set into passages and questions."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .config import DATA_FILE, DATA_SHA256, DATA_URL


@dataclass(frozen=True)
class Passage:
    pid: str          # "<article index>-<paragraph index>"
    title: str
    text: str


@dataclass(frozen=True)
class Answer:
    text: str
    start: int        # character offset in the passage

    @property
    def end(self) -> int:
        return self.start + len(self.text)


@dataclass(frozen=True)
class Question:
    qid: str
    text: str
    pid: str                          # the paragraph the question was written about
    answerable: bool
    answers: tuple[Answer, ...] = field(default_factory=tuple)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(path: Path = DATA_FILE, url: str = DATA_URL, force: bool = False) -> Path:
    """Fetch the JSON once and check it is the exact file the results were produced with."""
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    digest = sha256(tmp)
    if digest != DATA_SHA256:
        tmp.unlink()
        raise ValueError(f"checksum mismatch for {url}: got {digest}")
    tmp.rename(path)
    return path


def parse(raw: dict) -> tuple[list[Passage], list[Question]]:
    """Flatten the nested SQuAD structure (article -> paragraph -> question)."""
    passages: list[Passage] = []
    questions: list[Question] = []
    for a_idx, article in enumerate(raw["data"]):
        title = article["title"].replace("_", " ")
        for p_idx, para in enumerate(article["paragraphs"]):
            pid = f"{a_idx}-{p_idx}"
            context = para["context"]
            passages.append(Passage(pid=pid, title=title, text=context))
            for qa in para["qas"]:
                answers = []
                seen = set()
                for ans in qa.get("answers", []):
                    key = (ans["text"], ans["answer_start"])
                    if key in seen:
                        continue
                    seen.add(key)
                    if context[ans["answer_start"]: ans["answer_start"] + len(ans["text"])] != ans["text"]:
                        raise ValueError(f"answer offset does not match text in question {qa['id']}")
                    answers.append(Answer(text=ans["text"], start=ans["answer_start"]))
                questions.append(
                    Question(
                        qid=qa["id"],
                        text=qa["question"].strip(),
                        pid=pid,
                        answerable=not qa.get("is_impossible", False),
                        answers=tuple(answers),
                    )
                )
    return passages, questions


def load(path: Path | None = None) -> tuple[list[Passage], list[Question]]:
    path = path or download()
    with open(path, encoding="utf-8") as f:
        return parse(json.load(f))


if __name__ == "__main__":
    passages, questions = load()
    answerable = sum(q.answerable for q in questions)
    print(f"{len(passages)} passages, {len(questions)} questions "
          f"({answerable} answerable, {len(questions) - answerable} unanswerable)")
