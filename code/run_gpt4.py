"""Runs GPT-4-turbo / GPT-4o hallucination detection on the held-out test
set, with ONE API call PER PAPER: the full paper text and every one of that
paper's summary sentences are sent together, and the model returns one
classification per sentence in a single response.

Requires an OpenAI API key -- see README.md.

Usage:
    python run_gpt4.py                       # both models, held-out test set (25 papers)
    python run_gpt4.py --models gpt-4o
    python run_gpt4.py --dry-run
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import config
from dataset_utils import filter_to_papers, load_dataset_rows, print_coverage_report, resolve_matched_rows
from evidence import get_full_paper_text
from manifest import build_manifest, normalize_title
from openai_client import classify_paper_sentences, get_client

FIELDNAMES = ["Paper Name", "Aspect", "Sentence", "Gold_Hallucination", "Predicted_Hallucination", "Raw_Model_Response"]


def group_by_paper(rows):
    """Preserves each paper's row order (needed since sentence numbers in
    the batched prompt/response are positional)."""
    by_paper = defaultdict(list)
    for row in rows:
        by_paper[row["Paper Name"]].append(row)
    return by_paper


def load_done_papers(out_path, by_paper):
    """A paper counts as done only if ALL its rows are already present AND
    successfully predicted -- a paper that failed mid-run must be
    reprocessed, not treated as done just because rows exist for it."""
    if not out_path.exists():
        return set(), []
    with open(out_path, newline="") as f:
        existing_rows = list(csv.DictReader(f))
    counts = defaultdict(int)
    for r in existing_rows:
        if r["Predicted_Hallucination"] != "":
            counts[r["Paper Name"]] += 1
    done_papers = {p for p, n in counts.items() if n == len(by_paper.get(p, []))}
    kept_rows = [r for r in existing_rows if r["Paper Name"] in done_papers]
    return done_papers, kept_rows


def write_all(out_path, rows):
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def dry_run_report(by_paper, manifest):
    paper_json_cache = {}
    for paper_name, paper_rows in by_paper.items():
        title, json_path = manifest[normalize_title(paper_name)]
        if json_path not in paper_json_cache:
            paper_json_cache[json_path] = json.loads(Path(json_path).read_text())
        text, truncated = get_full_paper_text(paper_json_cache[json_path])
        print(f"\n[{title}]")
        print(f"  full-paper chars: {len(text)}{' (TRUNCATED)' if truncated else ''}")
        print(f"  sentences to classify: {len(paper_rows)} -> 1 API call for this paper")
    print(f"\n{len(by_paper)} papers -> {len(by_paper)} API calls per model.")


def run_model(client, model, by_paper, manifest):
    out_path = config.RESULTS_DIR / f"predictions_{model}.csv"
    done_papers, existing_rows = load_done_papers(out_path, by_paper)
    all_rows = list(existing_rows)

    n_papers = len(by_paper)
    print(f"\n=== {model}: {n_papers} papers total, {len(done_papers)} already fully done ===")

    paper_json_cache = {}
    n_calls = 0
    for i, (paper_name, paper_rows) in enumerate(by_paper.items(), 1):
        if paper_name in done_papers:
            continue

        title, json_path = manifest[normalize_title(paper_name)]
        if json_path not in paper_json_cache:
            paper_json_cache[json_path] = json.loads(Path(json_path).read_text())
        paper_text, _truncated = get_full_paper_text(paper_json_cache[json_path])

        sentences = [r["Sentence"] for r in paper_rows]
        classifications = classify_paper_sentences(client, model, paper_text, sentences)
        n_calls += 1

        for row, (verdict, raw_line) in zip(paper_rows, classifications):
            all_rows.append({
                "Paper Name": paper_name,
                "Aspect": row["Aspect"],
                "Sentence": row["Sentence"],
                "Gold_Hallucination": row["Hallucination"],
                "Predicted_Hallucination": "" if verdict is None else verdict,
                "Raw_Model_Response": raw_line,
            })

        write_all(out_path, all_rows)
        print(f"  {i}/{n_papers} papers processed ({n_calls} API calls this run)")

    print(f"Saved predictions to {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temperature", default="0", choices=["0", "0.8"], help="Which decoding setting's rows to evaluate (0 = greedy, 0.8 = sampled).")
    parser.add_argument("--models", nargs="+", default=config.MODELS, choices=config.MODELS)
    parser.add_argument("--all-papers", action="store_true", help="Evaluate on all 50 papers instead of the 25-paper held-out test set.")
    parser.add_argument("--dry-run", action="store_true", help="Print per-paper call info and exit, without calling the API.")
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(config.PAPERS_DIR)
    rows = load_dataset_rows(config.DATASET_CSV, temperature=args.temperature)
    matched_rows, skipped_papers = resolve_matched_rows(rows, manifest)
    print_coverage_report(rows, matched_rows, skipped_papers)

    if not args.all_papers:
        matched_rows = filter_to_papers(matched_rows, config.test_papers_for(args.temperature))
        print(f"Restricted to the {len(config.test_papers_for(args.temperature))}-paper held-out test set: {len(matched_rows)} rows.")

    by_paper = group_by_paper(matched_rows)

    if args.dry_run:
        dry_run_report(by_paper, manifest)
        return

    client = get_client()
    for model in args.models:
        run_model(client, model, by_paper, manifest)


if __name__ == "__main__":
    main()
