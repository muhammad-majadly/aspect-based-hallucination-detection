"""Produces the single, consistent final "Our Model" prediction files: ONE
19-feature logistic regression fit (top_k=7, trained on the 25 temp=0.8
non-test papers + 150 synthetic sentences), evaluated on BOTH the temp=0
test set (config.TEST_PAPERS, 465 rows) and the temp=0.8 test set
(config.TEST_PAPERS_TEMP08, 408 rows) -- both come from the same trained
model.

Every row is computed fresh at TOP_K -- retrieval- and NLI-derived features
are computed over the top-k evidence candidates, so a cached feature file
from a different top_k value cannot be reused here even for unaffected
papers; only feature values computed at this exact TOP_K are valid inputs
to this model.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataset_utils import load_dataset_rows, resolve_matched_rows
from feature_extraction import FEATURES_13_PLUS_4_PLUS_FP
from manifest import build_manifest
from run_our_model import compute_feature_rows, load_synthetic_feature_rows, write_predictions

TOP_K = 7


def compute_for_papers(temperature, papers, manifest):
    rows = load_dataset_rows(config.DATASET_CSV, temperature=temperature)
    matched, _ = resolve_matched_rows(rows, manifest)
    if papers is not None:
        matched = [r for r in matched if r["Paper Name"] in papers]
    return compute_feature_rows(matched, manifest, top_k=TOP_K)


def to_arrays(rows, feature_names):
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
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
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


def main():
    manifest = build_manifest(config.PAPERS_DIR)

    train_rows = compute_for_papers("0.8", None, manifest)
    train_rows = [r for r in train_rows if r["Paper Name"] not in config.TEST_PAPERS_TEMP08]
    synthetic_rows = load_synthetic_feature_rows(top_k=TOP_K)
    train_rows = train_rows + synthetic_rows
    print(f"Training set: {len(train_rows)} rows.")

    test08_rows = compute_for_papers("0.8", config.TEST_PAPERS_TEMP08, manifest)
    test0_rows = compute_for_papers("0", config.TEST_PAPERS, manifest)
    print(f"Test08={len(test08_rows)}, Test0={len(test0_rows)}")

    X_train, y_train, groups_train = to_arrays(train_rows, FEATURES_13_PLUS_4_PLUS_FP)

    best_C, best_C_f1, best_oof = None, -1.0, None
    for C in config.C_GRID:
        oof = lopo_oof_probs(X_train, y_train, groups_train, C)
        t, f1 = best_threshold(y_train, oof)
        if f1 > best_C_f1:
            best_C, best_C_f1, best_oof = C, f1, oof
    threshold, lopo_f1 = best_threshold(y_train, best_oof)
    print(f"Selected C={best_C}, threshold={threshold:.2f}, LOPO F1={lopo_f1:.3f}")

    final_model = make_model(best_C)
    final_model.fit(X_train, y_train)

    X_test08, y_test08, _ = to_arrays(test08_rows, FEATURES_13_PLUS_4_PLUS_FP)
    probs08 = final_model.predict_proba(X_test08)[:, 1]
    preds08 = (probs08 >= threshold).astype(int)
    report(y_test08, preds08, "Held-out temp=0.8")

    X_test0, y_test0, _ = to_arrays(test0_rows, FEATURES_13_PLUS_4_PLUS_FP)
    probs0 = final_model.predict_proba(X_test0)[:, 1]
    preds0 = (probs0 >= threshold).astype(int)
    report(y_test0, preds0, "Held-out temp=0")

    y_comb = np.concatenate([y_test0, y_test08])
    preds_comb = np.concatenate([preds0, preds08])
    report(y_comb, preds_comb, "Held-out COMBINED (single consistent model)")

    write_predictions(config.RESULTS_DIR / "predictions_our-model-final_temp08.csv", test08_rows, probs08, preds08,
                       extra_label=f",threshold={threshold:.2f},C={best_C},features=19,top_k={TOP_K}")
    write_predictions(config.RESULTS_DIR / "predictions_our-model-final_temp0.csv", test0_rows, probs0, preds0,
                       extra_label=f",threshold={threshold:.2f},C={best_C},features=19,top_k={TOP_K}")


if __name__ == "__main__":
    main()
