"""LLM-as-judge evaluation of the corrections produced by run_correction.py,
scoring each on 4 aspects (1-5): faithfulness, fluency, relevance, and
adequacy (see openai_client.evaluate_correction for the exact rubric).

Usage:
    OPENAI_API_KEY=$(cat api_key.txt) python evaluate_correction.py
"""
import argparse
import csv

import config
from openai_client import evaluate_correction, get_client

ASPECTS = ["faithfulness", "fluency", "relevance", "adequacy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", default="predictions_correction.csv")
    parser.add_argument("--out", default="correction_evaluation.csv")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--pass-threshold", type=int, default=4)
    args = parser.parse_args()

    in_path = config.RESULTS_DIR / args.in_file
    with open(in_path, newline="") as f:
        rows = list(csv.DictReader(f))

    client = get_client()
    out_rows = []
    for i, row in enumerate(rows, 1):
        scores = evaluate_correction(
            client, args.model, row["Sentence"], row["Evidence Used"], row["Corrected Sentence"]
        )
        if scores is None:
            scores = {a: None for a in ASPECTS}
        out_rows.append({**row, **{f"LLM_{a}": scores[a] for a in ASPECTS}})
        print(f"[{i}/{len(rows)}] {scores}")

    out_path = config.RESULTS_DIR / args.out
    fieldnames = list(rows[0].keys()) + [f"LLM_{a}" for a in ASPECTS]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"\nWrote {len(out_rows)} evaluations to {out_path}")
    for a in ASPECTS:
        vals = [r[f"LLM_{a}"] for r in out_rows if r[f"LLM_{a}"] is not None]
        avg = sum(vals) / len(vals)
        pct_pass = 100 * sum(1 for v in vals if v >= args.pass_threshold) / len(vals)
        print(f"{a:14s} avg={avg:.2f}/5 ({100*avg/5:.1f}%)  pass(>={args.pass_threshold})={pct_pass:.1f}%")


if __name__ == "__main__":
    main()
