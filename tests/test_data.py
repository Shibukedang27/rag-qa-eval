from ragqa.data import parse


def test_fixture_parses(mini):
    passages, questions = mini
    assert len(passages) == 6
    assert {p.title for p in passages} == {"Normans", "Oxygen", "Rhine"}
    assert any(q.answerable for q in questions)
    assert any(not q.answerable for q in questions)


def test_answer_offsets_match_text(mini):
    passages, questions = mini
    by_pid = {p.pid: p for p in passages}
    for q in questions:
        for a in q.answers:
            assert by_pid[q.pid].text[a.start:a.end] == a.text


def test_unanswerable_have_no_answers(mini):
    _, questions = mini
    for q in questions:
        if not q.answerable:
            assert q.answers == ()


def test_duplicate_annotations_are_merged():
    raw = {"data": [{"title": "T", "paragraphs": [{"context": "Paris is in France.", "qas": [
        {"id": "q1", "question": "Where is Paris?", "is_impossible": False,
         "answers": [{"text": "France", "answer_start": 12}] * 3}]}]}]}
    _, questions = parse(raw)
    assert len(questions[0].answers) == 1


def test_bad_offset_raises():
    raw = {"data": [{"title": "T", "paragraphs": [{"context": "Paris is in France.", "qas": [
        {"id": "q1", "question": "Where is Paris?", "is_impossible": False,
         "answers": [{"text": "France", "answer_start": 0}]}]}]}]}
    try:
        parse(raw)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
