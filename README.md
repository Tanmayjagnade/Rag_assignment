# Government Gender-Budget RAG

A small, reproducible retrieval system over the supplied government budget corpus. It extracts each PDF page, CSV row, and XLS row into source-attributed chunks, then ranks them with TF-IDF lexical retrieval. Query-time year boosting reduces confusion between near-identical Delhi statements. A few Hindi scheme/jurisdiction terms are translated to English because the PDFs' Hindi text is encoded in legacy fonts.

## Run

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python ingest.py
python query.py "What was Delhi's outlay for Ladli Yojna in 2024-25?"
python run_eval.py
```

`query.py` returns a verified answer and its source where it recognises a supported table or narrative pattern. It reports `answered: false` rather than guessing when it cannot safely extract an answer. Use `--evidence` to inspect the top source chunks directly.

```powershell
python query.py "What was Delhi's outlay for Ladli Yojna in 2024-25?" --evidence
```

## Layout

- `ingest.py` creates `build/chunks.jsonl` (not committed; reproducible).
- `query.py` prints top source chunks with file/page or sheet/row metadata.
- `run_eval.py` regenerates `answers.jsonl` for all evaluation questions.
- `answers.jsonl` contains one object for every evaluation question.
