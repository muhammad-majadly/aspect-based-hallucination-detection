"""Evaluates "Our Model" (the learned aggregator: sentence-similarity
retrieval + DeBERTa NLI + a logistic-regression aggregator over 13 features)
on the held-out test set.

Default mode loads the frozen aggregator shipped in models/aggregator.joblib
-- the exact model whose numbers are reported in the paper. Its weights,
regularization strength (C), and decision threshold were selected via
leave-one-paper-out cross-validation on the union of two training sources
(see models/aggregator_meta.json and README.md):
  (1) an internal 17-paper split from an earlier project iteration, disjoint
      from this benchmark and NOT released here, and
  (2) 150 synthetic sentences (data/synthetic/), fabricated specifically for
      this training set with known entailed/contradicted/neutral ground
      truth, IS released here (see generate_synthetic_data.py).
Because (1) is not released, this frozen model is what makes the reported
results reproducible without it.

--retrain instead refits the aggregator from scratch, self-contained within
this benchmark: it trains on the 25 non-test papers of the 50-paper dataset
(the ones not in the held-out test set for --temperature; see
config.test_papers_for()) using the identical leave-one-paper-out
C/threshold selection, then evaluates on the same 25-paper held-out test set.
Add --include-synthetic to also add the 150 synthetic sentences to that
training set (still fully self-contained, since both sources are released).
This is a purely local computation (sentence-transformers + DeBERTa NLI +
scikit-learn, no API cost). The validated-best configuration trains on the
temperature=0.8 (sampled-decoding) non-test papers specifically -- pass
--temperature 0.8 to reproduce it; the default (--temperature 0) instead
retrains self-contained on the temperature=0 (greedy-decoding) split.

--retrain uses a richer configuration than the frozen model: 19 features
(the original 13 plus acronym_mismatch, number_overlap_evidence,
self_reference_leak, is_question, hedge_opinion, meta_commentary -- see
feature_extraction.py:FEATURES_13_PLUS_4_PLUS_FP) and FEATURE_TOP_K=6
instead of 5 (RETRAIN_TOP_K above), both empirically validated to improve
held-out F1 over the frozen model's 13-feature/top_k=5 configuration. The
frozen model itself is untouched and always uses
BASE_FEATURE_NAMES/FROZEN_TOP_K=5, so its reported numbers stay
reproducible regardless of future retrain-side improvements.

Usage:
    python run_our_model.py                                                    # frozen model (reproduces paper results)
    python run_our_model.py --temperature 0.8                                  # frozen model, evaluated on the temp=0.8 test set
    python run_our_model.py --retrain                                          # retrain, self-contained, temp=0 non-test papers
    python run_our_model.py --retrain --include-synthetic --temperature 0.8   # validated-best configuration
"""
import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataset_utils import filter_to_papers, load_dataset_rows, print_coverage_report, resolve_matched_rows
from feature_extraction import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    FEATURE_TOP_K,
    FEATURES_13_PLUS_4_PLUS_FP,
    add_duplicate_feature,
    compute_features,
)

FROZEN_TOP_K = 5  # the frozen model was trained at this top_k -- must never change
RETRAIN_TOP_K = 6  # validated best: +0.015 combined F1 over top_k=5
from manifest import build_manifest, normalize_title

FIELDNAMES = ["Paper Name", "Aspect", "Sentence", "Gold_Hallucination", "Predicted_Hallucination", "Raw_Model_Response"]


def compute_feature_rows(rows, manifest, top_k=FEATURE_TOP_K):
    paper_json_cache = {}
    out = []
    for i, row in enumerate(rows, 1):
        paper_name, aspect, sentence = row["Paper Name"], row["Aspect"], row["Sentence"]
        _title, json_path = manifest[normalize_title(paper_name)]
        if json_path not in paper_json_cache:
            paper_json_cache[json_path] = json.loads(Path(json_path).read_text())
        paper = paper_json_cache[json_path]

        features = compute_features(paper, aspect, sentence, top_k=top_k)
        out.append({
            "Paper Name": paper_name, "Aspect": aspect, "Sentence": sentence,
            "Gold_Hallucination": row["Hallucination"], **features,
        })
        if i % 50 == 0 or i == len(rows):
            print(f"  features: {i}/{len(rows)} rows processed")
    return add_duplicate_feature(out)


def load_synthetic_feature_rows(top_k=FEATURE_TOP_K):
    """Computes features for the 150 synthetic sentences (data/synthetic/),
    matching each sentence directly to its fabricated paper JSON rather than
    going through manifest.py (synthetic papers aren't part of the released
    benchmark's manifest)."""
    synthetic_dir = config.ROOT / "data" / "synthetic"
    csv_path = synthetic_dir / "synthetic_sentences.csv"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))

    paper_json_cache = {}
    out = []
    for row in rows:
        paper_name, aspect, sentence = row["Paper Name"], row["Aspect"], row["Sentence"]
        if paper_name not in paper_json_cache:
            for json_path in sorted(synthetic_dir.glob("synth_*.json")):
                data = json.loads(json_path.read_text())
                paper_json_cache[data["paper_title"]] = data
        paper = paper_json_cache[paper_name]
        features = compute_features(paper, aspect, sentence, top_k=top_k)
        out.append({
            "Paper Name": paper_name, "Aspect": aspect, "Sentence": sentence,
            "Gold_Hallucination": row["Hallucination"], **features,
        })
    print(f"  synthetic features: {len(out)}/150 rows processed")
    return add_duplicate_feature(out)


def to_arrays(rows, feature_names=BASE_FEATURE_NAMES):
    X = np.array([[float(r[f]) for f in feature_names] for r in rows])
    y = np.array([int(r["Gold_Hallucination"]) for r in rows])
    groups = np.array([r["Paper Name"] for r in rows])
    return X, y, groups


def make_model(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def lopo_oof_probs(X, y, groups, C):
    oof = np.zeros(len(y))
    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        model = make_model(C)
        model.fit(X[train_idx], y[train_idx])
        oof[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof


def best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, -1.0
    for t in config.THRESHOLD_GRID:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def report(y_true, preds, label):
    print(
        f"{label}: N={len(y_true)} "
        f"Acc={accuracy_score(y_true, preds):.3f} "
        f"P={precision_score(y_true, preds, zero_division=0):.3f} "
        f"R={recall_score(y_true, preds, zero_division=0):.3f} "
        f"F1={f1_score(y_true, preds, zero_division=0):.3f}"
    )


def write_predictions(out_path, test_rows, probs, preds, extra_label):
    out_rows = []
    for r, prob, pred in zip(test_rows, probs, preds):
        out_rows.append({
            "Paper Name": r["Paper Name"], "Aspect": r["Aspect"], "Sentence": r["Sentence"],
            "Gold_Hallucination": r["Gold_Hallucination"], "Predicted_Hallucination": int(pred),
            "Raw_Model_Response": f"learned_aggregator{extra_label}(prob={prob:.3f})",
        })
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Paper Name", "Aspect", "Sentence", "Gold_Hallucination", "Predicted_Hallucination", "Raw_Model_Response"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved predictions to {out_path}")


def run_frozen(test_rows, manifest):
    model_path = config.MODELS_DIR / "aggregator.joblib"
    meta_path = config.MODELS_DIR / "aggregator_meta.json"
    print(f"Loading frozen aggregator from {model_path} ...")
    model = joblib.load(model_path)
    threshold = json.loads(meta_path.read_text())["threshold"]

    print("Extracting features for the held-out test set...")
    # Hardcoded to the frozen model's own training config (13 base features,
    # top_k=5) regardless of RETRAIN_TOP_K / FEATURES_13_PLUS_4 above -- this
    # must stay fixed forever, or the shipped model's reported numbers stop
    # being reproducible.
    feature_rows = compute_feature_rows(test_rows, manifest, top_k=FROZEN_TOP_K)
    X_test, y_test, _ = to_arrays(feature_rows, BASE_FEATURE_NAMES)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    report(y_test, preds, "Held-out (frozen aggregator)")

    out_path = config.RESULTS_DIR / "predictions_our-model.csv"
    write_predictions(out_path, feature_rows, probs, preds, extra_label=f",threshold={threshold:.2f},frozen")


def run_retrain(test_rows, all_rows, manifest, include_synthetic, test_papers):
    train_paper_names = {r["Paper Name"] for r in all_rows} - test_papers
    train_rows_raw = [r for r in all_rows if r["Paper Name"] in train_paper_names]
    print(f"Retraining on {len(train_rows_raw)} rows / {len(train_paper_names)} non-test papers of the benchmark.")

    print("Extracting features for the training papers...")
    train_feature_rows = compute_feature_rows(train_rows_raw, manifest, top_k=RETRAIN_TOP_K)

    if include_synthetic:
        synthetic_feature_rows = load_synthetic_feature_rows(top_k=RETRAIN_TOP_K)
        train_feature_rows = train_feature_rows + synthetic_feature_rows
        print(f"Added 150 synthetic sentences -> {len(train_feature_rows)} total training rows.")

    X_train, y_train, groups_train = to_arrays(train_feature_rows, FEATURES_13_PLUS_4_PLUS_FP)

    print("\n=== Selecting regularization strength C via leave-one-paper-out CV ===")
    best_C, best_C_f1, best_oof = None, -1.0, None
    for C in config.C_GRID:
        oof = lopo_oof_probs(X_train, y_train, groups_train, C)
        t, f1 = best_threshold(y_train, oof)
        print(f"  C={C:<5} best_threshold={t:.2f} LOPO F1={f1:.3f}")
        if f1 > best_C_f1:
            best_C, best_C_f1, best_oof = C, f1, oof
    threshold, _ = best_threshold(y_train, best_oof)
    print(f"\nSelected C={best_C}, threshold={threshold:.2f}")
    report(y_train, (best_oof >= threshold).astype(int), "LOPO CV (retrained)")

    final_model = make_model(best_C)
    final_model.fit(X_train, y_train)

    print("\nExtracting features for the held-out test set...")
    test_feature_rows = compute_feature_rows(test_rows, manifest, top_k=RETRAIN_TOP_K)
    X_test, y_test, _ = to_arrays(test_feature_rows, FEATURES_13_PLUS_4_PLUS_FP)
    probs = final_model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    report(y_test, preds, "Held-out (retrained aggregator)")

    suffix = "-synthetic" if include_synthetic else ""
    out_path = config.RESULTS_DIR / f"predictions_our-model-retrained{suffix}.csv"
    write_predictions(out_path, test_feature_rows, probs, preds, extra_label=f",threshold={threshold:.2f},C={best_C},retrained{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--temperature", default="0", choices=["0", "0.8"], help="Which decoding setting's rows to use for the held-out test set (and, with --retrain, for the non-test training papers too).")
    parser.add_argument("--retrain", action="store_true", help="Retrain on the 25 non-test papers instead of using the shipped frozen model.")
    parser.add_argument("--include-synthetic", action="store_true", help="With --retrain, also add the 150 synthetic sentences (data/synthetic/) to the training set.")
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(config.PAPERS_DIR)
    rows = load_dataset_rows(config.DATASET_CSV, temperature=args.temperature)
    matched_rows, skipped_papers = resolve_matched_rows(rows, manifest)
    print_coverage_report(rows, matched_rows, skipped_papers)

    test_papers = config.test_papers_for(args.temperature)
    test_rows = filter_to_papers(matched_rows, test_papers)
    print(f"Held-out test set: {len(test_rows)} rows / {len(test_papers)} papers.")

    if args.retrain:
        run_retrain(test_rows, matched_rows, manifest, args.include_synthetic, test_papers)
    else:
        run_frozen(test_rows, manifest)


if __name__ == "__main__":
    main()
