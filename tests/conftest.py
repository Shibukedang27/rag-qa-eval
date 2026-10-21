from pathlib import Path

import pytest

from ragqa.chunking import chunk_corpus
from ragqa.data import load

FIXTURE = Path(__file__).parent / "fixtures" / "mini_squad.json"


@pytest.fixture(scope="session")
def mini():
    passages, questions = load(FIXTURE)
    return passages, questions


@pytest.fixture(scope="session")
def mini_chunks(mini):
    passages, _ = mini
    return chunk_corpus(passages, size=40, overlap=10)
