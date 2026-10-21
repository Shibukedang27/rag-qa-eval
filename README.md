# rag-qa-eval

Retrieval-augmented question answering on the SQuAD 2.0 dev set, built around measurement:
which retriever actually finds the evidence, how often, and how fast.

Work in progress. Data loading and chunking are in place; retrieval comparison comes next.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python -m ragqa.data      # downloads the SQuAD 2.0 dev set and checks its SHA-256
pytest -q
```
