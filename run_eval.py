"""Run the RAG system on every unlabelled evaluation question."""
from __future__ import annotations
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from query import answer_from_evidence, search


def main():
    chunks = [json.loads(line) for line in Path("build/chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    matrix = vectorizer.fit_transform(chunk["text"] for chunk in chunks)
    questions = [json.loads(line) for line in Path("questions_eval.jsonl").read_text(encoding="utf-8").splitlines()]
    outputs = []
    for item in questions:
        results = search(item["question"], chunks, vectorizer, matrix)
        answer, evidence = answer_from_evidence(item["question"], results, chunks)
        source = {"file": evidence["file"]}
        if evidence.get("page") is not None:
            source["page"] = evidence["page"]
        if evidence.get("location"):
            source["location"] = evidence["location"]
        outputs.append({"question_id": item["question_id"], "answered": answer is not None,
                        "answer": answer or "", "sources": [source] if answer else []})
    Path("answers.jsonl").write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in outputs), encoding="utf-8")
    print(f"Wrote {len(outputs)} answers to answers.jsonl ({sum(x['answered'] for x in outputs)} answered, {sum(not x['answered'] for x in outputs)} abstained).")


if __name__ == "__main__":
    main()
