"""Computes accuracy/precision/recall/F1 for every results/predictions_*.csv
file, both overall and broken down by aspect, and writes a comparison report.

Usage:
    python evaluate.py
"""
from collections import defaultdict
import csv

import config


def load_predictions(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(rows):
    tp = fp = tn = fn = skipped = 0
    for r in rows:
        if r["Predicted_Hallucination"] == "":
            skipped += 1
            continue
        gold = int(r["Gold_Hallucination"])
        pred = int(r["Predicted_Hallucination"])
        if gold == 1 and pred == 1:
            tp += 1
        elif gold == 0 and pred == 1:
            fp += 1
        elif gold == 0 and pred == 0:
            tn += 1
        elif gold == 1 and pred == 0:
            fn += 1

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "n": n, "skipped": skipped, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
    }


def metrics_by_aspect(rows):
    by_aspect = defaultdict(list)
    for r in rows:
        by_aspect[r["Aspect"]].append(r)
    return {aspect: compute_metrics(rs) for aspect, rs in sorted(by_aspect.items())}


def print_table(label, rows_by_key):
    header = f"{label:<14} {'N':>6} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    print(header)
    print("-" * len(header))
    for key, m in rows_by_key.items():
        print(f"{key:<14} {m['n']:>6} {m['accuracy']:>7.3f} {m['precision']:>7.3f} {m['recall']:>7.3f} {m['f1']:>7.3f}")


def markdown_table(header_label, rows_by_key):
    lines = [f"| {header_label} | N | Accuracy | Precision | Recall | F1 |", "|---|---|---|---|---|---|"]
    for key, m in rows_by_key.items():
        lines.append(f"| {key} | {m['n']} | {m['accuracy']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |")
    return lines


def discover_prediction_files():
    """Finds every results/predictions_<model>.csv file, so any detector is
    picked up automatically."""
    if not config.RESULTS_DIR.exists():
        return {}
    files = {}
    for path in sorted(config.RESULTS_DIR.glob("predictions_*.csv")):
        model = path.stem[len("predictions_"):]
        files[model] = path
    return files


def main():
    all_metrics, all_rows = {}, {}
    for model, path in discover_prediction_files().items():
        rows = load_predictions(path)
        all_rows[model] = rows
        all_metrics[model] = compute_metrics(rows)

    if not all_metrics:
        print("No predictions found. Run one of the run_*.py scripts first.")
        return

    print("=== Overall detection performance ===")
    print_table("Model", all_metrics)

    report_lines = ["# Hallucination Detection Results", "", "## Overall", ""]
    report_lines += markdown_table("Model", all_metrics)

    for model, rows in all_rows.items():
        by_aspect = metrics_by_aspect(rows)
        print(f"\n=== {model}: by aspect ===")
        print_table("Aspect", by_aspect)
        report_lines += ["", f"## {model} by aspect", ""]
        report_lines += markdown_table("Aspect", by_aspect)

    report_path = config.RESULTS_DIR / "comparison_report.md"
    report_path.write_text("\n".join(report_lines) + "\n")
    print(f"\nSaved comparison report to {report_path}")


if __name__ == "__main__":
    main()
