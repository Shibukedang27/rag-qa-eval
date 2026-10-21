import pytest

from ragqa.chunking import chunk_passage, gold_chunks, index_by_pid
from ragqa.data import Answer, Passage, Question


def make_passage(n_words: int) -> Passage:
    return Passage(pid="0-0", title="T", text=" ".join(f"w{i}" for i in range(n_words)))


@pytest.mark.parametrize("n", [1, 39, 40, 41, 70, 71, 100, 157])
def test_windows_cover_every_word(n):
    p = make_passage(n)
    chunks = chunk_passage(p, size=40, overlap=10)
    covered = set()
    for c in chunks:
        assert p.text[c.start:c.end] == c.text
        assert len(c.text.split()) <= 40
        covered.update(c.text.split())
    assert covered == set(p.text.split())


def test_neighbours_share_overlap_words():
    chunks = chunk_passage(make_passage(100), size=40, overlap=10)
    for a, b in zip(chunks, chunks[1:]):
        assert a.text.split()[-10:] == b.text.split()[:10]


def test_short_passage_is_one_chunk():
    assert len(chunk_passage(make_passage(25), size=40, overlap=10)) == 1


def test_bad_overlap_rejected():
    with pytest.raises(ValueError):
        chunk_passage(make_passage(10), size=10, overlap=10)


def test_gold_chunk_must_contain_whole_answer():
    p = make_passage(100)
    chunks = chunk_passage(p, size=40, overlap=10)
    start = p.text.index("w35")
    ans = Answer(text="w35 w36", start=start)            # inside chunk 0 and chunk 1
    q = Question(qid="q", text="?", pid=p.pid, answerable=True, answers=(ans,))
    assert gold_chunks(q, index_by_pid(chunks)) == {0, 1}

    start = p.text.index("w39")
    ans = Answer(text="w39 w40", start=start)            # straddles the end of chunk 0
    q = Question(qid="q", text="?", pid=p.pid, answerable=True, answers=(ans,))
    assert gold_chunks(q, index_by_pid(chunks)) == {1}


def test_every_fixture_answer_has_gold_chunk(mini, mini_chunks):
    _, questions = mini
    by_pid = index_by_pid(mini_chunks)
    for q in questions:
        if q.answerable:
            assert gold_chunks(q, by_pid)
