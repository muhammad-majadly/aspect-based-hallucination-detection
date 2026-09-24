"""One-off for the paper's ablation study: evaluates the 13/17/19-feature
logistic regression aggregators on the FINAL test split (config.TEST_PAPERS
/ TEST_PAPERS_TEMP08), all from a single feature-extraction pass (top_k=7,
matching the final model in paper_final_ourmodel.py), so the three rows are
directly comparable. Also emits the temp=0 test set (unaffected by the
temp=0.8-specific split) and the combined 873-row number for each feature
set.

Every row is computed fresh at TOP_K -- see paper_final_ourmodel.py's
docstring for why cached feature files from a different top_k cannot be
reused here.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataset_utils import load_dataset_rows, resolve_matched_rows
from feature_extraction import BASE_FEATURE_NAMES, FEATURES_13_PLUS_4, FEATURES_13_PLUS_4_PLUS_FP
from manifest import build_manifest
from run_our_model import compute_feature_rows, load_synthetic_feature_rows

TOP_K = 7
FEATURE_SETS = {13: BASE_FEATURE_NAMES, 17: FEATURES_13_PLUS_4, 19: FEATURES_13_PLUS_4_PLUS_FP}


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


def metrics(y_true, preds):
    return (
        accuracy_score(y_true, preds),
        precision_score(y_true, preds, zero_division=0),
        recall_score(y_true, preds, zero_division=0),
        f1_score(y_true, preds, zero_division=0),
    )


def main():
    manifest = build_manifest(config.PAPERS_DIR)

    print("=== Building the (feature-complete) training/test rows once ===")
    train_rows = compute_for_papers("0.8", None, manifest)
    train_rows = [r for r in train_rows if r["Paper Name"] not in config.TEST_PAPERS_TEMP08]
    synthetic_rows = load_synthetic_feature_rows(top_k=TOP_K)
    train_rows = train_rows + synthetic_rows

    test08_rows = compute_for_papers("0.8", config.TEST_PAPERS_TEMP08, manifest)
    test0_rows = compute_for_papers("0", config.TEST_PAPERS, manifest)

    print(f"Train={len(train_rows)}, Test08={len(test08_rows)}, Test0={len(test0_rows)}\n")

    print(f"{'Features':<10}{'LOPO F1':>9}{'Acc':>8}{'P':>8}{'R':>8}{'F1(combined)':>14}")
    for n_feats, feat_names in FEATURE_SETS.items():
        X_train, y_train, groups_train = to_arrays(train_rows, feat_names)
        best_C, best_C_f1, best_oof = None, -1.0, None
        for C in config.C_GRID:
            oof = lopo_oof_probs(X_train, y_train, groups_train, C)
            _, f1 = best_threshold(y_train, oof)
            if f1 > best_C_f1:
                best_C, best_C_f1, best_oof = C, f1, oof
        threshold, lopo_f1 = best_threshold(y_train, best_oof)

        final_model = make_model(best_C)
        final_model.fit(X_train, y_train)

        X_test08, y_test08, _ = to_arrays(test08_rows, feat_names)
        X_test0, y_test0, _ = to_arrays(test0_rows, feat_names)
        preds08 = (final_model.predict_proba(X_test08)[:, 1] >= threshold).astype(int)
        preds0 = (final_model.predict_proba(X_test0)[:, 1] >= threshold).astype(int)

        y_comb = np.concatenate([y_test0, y_test08])
        preds_comb = np.concatenate([preds0, preds08])
        acc, p, r, f1 = metrics(y_comb, preds_comb)
        print(f"{n_feats:<10}{lopo_f1:>9.3f}{acc:>8.3f}{p:>8.3f}{r:>8.3f}{f1:>14.3f}  (C={best_C}, threshold={threshold:.2f})")


if __name__ == "__main__":
    main()
