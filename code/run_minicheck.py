"""Runs MiniCheck (Liyan06/MiniCheck, EMNLP 2024) as a document-grounded
fact-checking baseline on the held-out test set.

Uses the aspect-restricted evidence text (evidence.get_evidence) rather than
the full paper text used by the other baselines -- a deliberate, documented
exception to config.USE_ASPECT_MAPPING (see evidence.py). MiniCheck's
CPU-friendly Inferencer backend (flan-t5-large) re-chunks and re-encodes the
ENTIRE document from scratch for every single (doc, claim) call, with no
caching across calls that share the same document. Aspect-restricted
evidence is far shorter than a full paper, which keeps this tractable.

Runs per (paper, aspect) group with incremental (resumable) CSV writes, so a
kill part way through doesn't lose all progress.

Uses MiniCheck-Flan-T5-Large (770M params), which runs entirely on CPU.
Model weights are cached under ./ckpts on first run (not committed to git).

Usage:
    python run_minicheck.py
"""
import argparse
import csv
import gc
import json
import time
from pathlib import Path

import config
from dataset_utils import filter_to_papers, load_dataset_rows, print_coverage_report, resolve_matched_rows
from evidence import get_evidence
from manifest import build_manifest, normalize_title

FIELDNAMES = ["Paper Name", "Aspect", "Sentence", "Gold_Hallucination", "Predicted_Hallucination", "Raw_Model_Response"]


def group_by_paper_aspect(matched_rows):
    groups = {}
    for row in matched_rows:
        groups.setdefault((row["Paper Name"], row["Aspect"]), []).append(row)
    return groups


def load_existing_predictions(out_path):
    done = set()
    if out_path.exists():
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                done.add((r["Paper Name"], r["Aspect"], r["Sentence"]))
    return done


def append_rows(out_path, rows, write_header):
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", default="0", choices=["0", "0.8"], help="Which decoding setting's rows to evaluate (0 = greedy, 0.8 = sampled).")
    parser.add_argument("--all-papers", action="store_true", help="Evaluate on all 50 papers instead of the 25-paper held-out test set.")
    parser.add_argument("--cache-dir", default="./ckpts")
    args = parser.parse_args()

    from minicheck.minicheck import MiniCheck

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "predictions_minicheck.csv"

    manifest = build_manifest(config.PAPERS_DIR)
    rows = load_dataset_rows(config.DATASET_CSV, temperature=args.temperature)
    matched_rows, skipped_papers = resolve_matched_rows(rows, manifest)
    print_coverage_report(rows, matched_rows, skipped_papers)
    if not args.all_papers:
        matched_rows = filter_to_papers(matched_rows, config.test_papers_for(args.temperature))
        print(f"Restricted to the {len(config.test_papers_for(args.temperature))}-paper held-out test set: {len(matched_rows)} rows.")

    already_done = load_existing_predictions(out_path)
    write_header = not out_path.exists()

    print("Loading MiniCheck-Flan-T5-Large...")
    scorer = MiniCheck(model_name="flan-t5-large", cache_dir=args.cache_dir)

    paper_json_cache = {}
    groups = group_by_paper_aspect(matched_rows)
    for i, ((paper_name, aspect), group_rows) in enumerate(groups.items(), 1):
        pending = [r for r in group_rows if (r["Paper Name"], r["Aspect"], r["Sentence"]) not in already_done]
        if not pending:
            print(f"[{i}/{len(groups)}] {paper_name[:40]} / {aspect} -> already done, skipping")
            continue

        _title, json_path = manifest[normalize_title(paper_name)]
        if json_path not in paper_json_cache:
            paper_json_cache[json_path] = json.loads(Path(json_path).read_text())
        paper = paper_json_cache[json_path]

        evidence_text, _truncated, _names = get_evidence(paper, aspect)

        t0 = time.time()
        docs = [evidence_text] * len(pending)
        claims = [r["Sentence"] for r in pending]
        pred_labels, raw_probs, _, _ = scorer.score(docs=docs, claims=claims)

        out_rows = []
        for row, pred, prob in zip(pending, pred_labels, raw_probs):
            # MiniCheck: pred=1 means the claim IS supported by the doc
            # (faithful); pred=0 means NOT supported -> hallucinated.
            verdict = 0 if pred == 1 else 1
            out_rows.append({
                "Paper Name": row["Paper Name"],
                "Aspect": row["Aspect"],
                "Sentence": row["Sentence"],
                "Gold_Hallucination": row["Hallucination"],
                "Predicted_Hallucination": verdict,
                "Raw_Model_Response": f"minicheck_support_prob={prob:.4f}",
            })

        append_rows(out_path, out_rows, write_header)
        write_header = False
        gc.collect()

        elapsed = time.time() - t0
        print(f"[{i}/{len(groups)}] {paper_name[:40]} / {aspect} ({len(pending)} sentences, {elapsed:.1f}s) processed")

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
