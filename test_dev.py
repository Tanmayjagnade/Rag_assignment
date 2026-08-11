"""Run retrieval checks against the labelled development questions."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from query import search


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    chunks = [json.loads(line) for line in Path("build/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(chunk["text"] for chunk in chunks)
    questions = [json.loads(line) for line in Path("questions_dev.jsonl").read_text(encoding="utf-8").splitlines()]

    hits = 0
    for item in questions:
        results = search(item["question"], chunks, vectorizer, matrix)
        rank = next((n for n, result in enumerate(results, start=1)
                     if result["file"] in item["source_files"]), None)
        hits += rank is not None
        status = f"PASS (source rank {rank})" if rank else "CHECK (source not in top 5)"
        print(f"\n{item['question_id']}: {status}")
        print(f"Question: {item['question']}")
        print(f"Expected answer: {item['answer']}")
        print(f"Top result: {results[0]['file']} | page/row: {results[0]['page'] or results[0]['location']}")

    print(f"\nSource retrieval: {hits}/{len(questions)} development questions found in the top 5.")


if __name__ == "__main__":
    main()
