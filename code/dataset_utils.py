"""Shared dataset-loading helpers used by every run_*.py script."""
import csv

from manifest import normalize_title


def load_dataset_rows(csv_path, temperature=None):
    """Loads dataset rows, optionally restricted to one generation setting
    via the Temperature column ("0" for greedy decoding, "0.8" for sampled
    decoding). temperature=None loads every row in the file."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if temperature is None:
        return rows
    return [r for r in rows if r["Temperature"] == str(temperature)]


def resolve_matched_rows(rows, manifest):
    """Splits dataset rows into those whose paper has a matching source
    JSON (and can therefore be evidence-checked) and those that don't."""
    matched, skipped_papers = [], set()
    for row in rows:
        if normalize_title(row["Paper Name"]) in manifest:
            matched.append(row)
        else:
            skipped_papers.add(row["Paper Name"])
    return matched, skipped_papers


def filter_to_papers(rows, paper_names):
    """Keeps only rows whose Paper Name is in `paper_names`."""
    return [r for r in rows if r["Paper Name"] in paper_names]


def print_coverage_report(rows, matched_rows, skipped_papers):
    print(f"Dataset rows: {len(rows)}")
    print(f"Matched to a source JSON: {len(matched_rows)}")
    if skipped_papers:
        print(
            f"Skipped {len(rows) - len(matched_rows)} rows from "
            f"{len(skipped_papers)} papers with no source JSON available:"
        )
        for p in sorted(skipped_papers):
            print(f"  - {p}")
