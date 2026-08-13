"""Runs SciHDC's hallucination-correction step (paper's Section 3.2.4) over
the true-positive catches on a set of held-out test papers, and writes one
corrected sentence per row to results/predictions_correction.csv.

Correcting requires an actual verdict as input, so this script re-derives
the true-positive flagged sentences from run_our_model.py's output for a
given --temperature: run that script first (e.g. `python run_our_model.py
--retrain --include-synthetic --temperature 0.8`) so
results/predictions_our-model.csv exists.

For each flagged sentence, retrieves the same top-k evidence comparisons
used by the detector's features (nli_backend.get_candidate_comparisons)
and picks the correction evidence per the paper's protocol: the
highest-confidence contradicting comparison if one clears
config.MIN_CONTRADICTION_CONFIDENCE, else the single most similar
comparison (the "neutral" case). Calls GPT-4o once per flagged sentence.

Usage:
    OPENAI_API_KEY=$(cat api_key.txt) python run_correction.py
    OPENAI_API_KEY=$(cat api_key.txt) python run_correction.py --papers "Paper A" "Paper B"
"""
import argparse
import csv
import json
from pathlib import Path

import config
from evidence import get_full_paper_text
from manifest import build_manifest, normalize_title
from nli_backend import get_candidate_comparisons
from openai_client import correct_sentence, get_client

FIELDNAMES = [
    "Paper Name", "Aspect", "Sentence", "Mode", "Evidence Used", "Corrected Sentence",
]


def load_flagged_true_positives(predictions_csv, papers=None):
    with open(predictions_csv, newline="") as f:
        rows = [
            r for r in csv.DictReader(f)
            if r["Predicted_Hallucination"] == "1" and r["Gold_Hallucination"] == "1"
        ]
    if papers is not None:
        rows = [r for r in rows if r["Paper Name"] in papers]
    return rows


def pick_correction_evidence(comparisons):
    """Mirrors nli_backend.classify_sentence_nli's contradiction-first
    rule: use the highest-confidence contradiction among comparisons that
    clear MIN_SIMILARITY, if any clears MIN_CONTRADICTION_CONFIDENCE; else
    fall back to the single most similar comparison (mode="neutral")."""
    eligible = [c for c in comparisons if c["similarity"] >= config.MIN_SIMILARITY]
    contradictions = [
        c for c in eligible
        if max(c["probs"], key=c["probs"].get) == "CONTRADICTION"
        and c["probs"]["CONTRADICTION"] >= config.MIN_CONTRADICTION_CONFIDENCE
    ]
    if contradictions:
        best = max(contradictions, key=lambda c: c["probs"]["CONTRADICTION"])
        return "contradiction", best["text"]
    if comparisons:
        best = max(comparisons, key=lambda c: c["similarity"])
        return "neutral", best["text"]
    return "neutral", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", default="predictions_our-model.csv",
                         help="Detector output to read flagged sentences from (relative to results/).")
    parser.add_argument("--papers", nargs="+", default=None,
                         help="Restrict correction to these papers (default: all papers in --predictions).")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--out", default="predictions_correction.csv")
    args = parser.parse_args()

    manifest = build_manifest(config.PAPERS_DIR)
    rows = load_flagged_true_positives(config.RESULTS_DIR / args.predictions, args.papers)
    print(f"{len(rows)} true-positive flagged sentences to correct")

    client = get_client()
    paper_json_cache = {}
    out_rows = []
    for i, row in enumerate(rows, 1):
        paper_name, aspect, sentence = row["Paper Name"], row["Aspect"], row["Sentence"]
        _title, json_path = manifest[normalize_title(paper_name)]
        if json_path not in paper_json_cache:
            paper_json_cache[json_path] = json.loads(Path(json_path).read_text())
        paper = paper_json_cache[json_path]

        full_text, _truncated = get_full_paper_text(paper)
        comparisons = get_candidate_comparisons(full_text, sentence, args.top_k)
        mode, evidence = pick_correction_evidence(comparisons)

        corrected = correct_sentence(client, args.model, sentence, evidence, mode)
        out_rows.append({
            "Paper Name": paper_name, "Aspect": aspect, "Sentence": sentence,
            "Mode": mode, "Evidence Used": evidence, "Corrected Sentence": corrected,
        })
        print(f"[{i}/{len(rows)}] {paper_name} / {aspect} ({mode})")
        print(f"  Original:  {sentence}")
        print(f"  Corrected: {corrected}")

    out_path = config.RESULTS_DIR / args.out
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {len(out_rows)} corrections to {out_path}")


if __name__ == "__main__":
    main()
