"""Extract the supplied PDFs, CSV and XLS into page/row-level JSONL chunks."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import pandas as pd
from pypdf import PdfReader


def emit(out, text, file, page=None, location=None):
    text = " ".join(str(text).split())
    if len(text) > 20:
        out.write(json.dumps({"text": text, "file": file, "page": page,
                              "location": location}, ensure_ascii=False) + "\n")


def ingest(corpus: Path, output: Path) -> None:
    with output.open("w", encoding="utf-8") as out:
        for path in sorted(corpus.iterdir()):
            if path.suffix.lower() == ".pdf":
                for n, page in enumerate(PdfReader(path).pages, start=1):
                    emit(out, page.extract_text() or "", path.name, n)
            elif path.suffix.lower() == ".csv":
                # The supplied Union CSV contains a few Windows-1252 bytes (not valid UTF-8).
                with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
                    for n, row in enumerate(csv.DictReader(f), start=2):
                        emit(out, " | ".join(f"{k}: {v}" for k, v in row.items()),
                             path.name, location=f"CSV row {n}")
            elif path.suffix.lower() == ".xls":
                for sheet in pd.ExcelFile(path).sheet_names:
                    frame = pd.read_excel(path, sheet_name=sheet, header=None)
                    for n, row in frame.iterrows():
                        emit(out, " | ".join(str(x) for x in row.dropna().tolist()),
                             path.name, location=f"{sheet} row {n + 1}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--output", default="build/chunks.jsonl")
    a = p.parse_args(); Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    ingest(Path(a.corpus), Path(a.output))
