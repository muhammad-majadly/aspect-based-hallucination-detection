"""Two more ablations for the paper, sharing one from-scratch top_k=7
feature-extraction pass (see paper_final_ourmodel.py's docstring for why):

1. Mutual information + ANOVA F-test between each of the 19 features and
   the gold label, on the training set -- ranks features independently of
   the classifier, and checks whether dropping the 3 lowest-MI features
   (self_reference_leak, hedge_opinion, is_question) changes held-out F1.
2. Classifier-choice ablation: logistic regression (this benchmark's
   choice, grid-searched C, matching paper_final_ourmodel.py exactly) vs.
   5 alternatives (random forest, gradient boosting, RBF SVM, kNN, MLP)
   at standard default hyperparameters -- the point is classifier family,
   not an equally exhaustive per-classifier grid search.
"""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from dataset_utils import load_dataset_rows, resolve_matched_rows
from feature_extraction import FEATURES_13_PLUS_4_PLUS_FP
from manifest import build_manifest
from run_our_model import compute_feature_rows, load_synthetic_feature_rows

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
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        m = make_model(C)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, -1.0
    for t in config.THRESHOLD_GRID:
        f1 = f1_score(y_true, (probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


def run_logreg(train_rows, test_rows, features, label):
    X_train, y_train, groups_train = to_arrays(train_rows, features)
    best_C, best_C_f1, best_oof = None, -1.0, None
    for C in config.C_GRID:
        oof = lopo_oof_probs(X_train, y_train, groups_train, C)
        _, f1 = best_threshold(y_train, oof)
        if f1 > best_C_f1:
            best_C, best_C_f1, best_oof = C, f1, oof
    threshold, lopo_f1 = best_threshold(y_train, best_oof)

    final_model = make_model(best_C)
    final_model.fit(X_train, y_train)
    X_test, y_test, _ = to_arrays(test_rows, features)
    probs = final_model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    p = precision_score(y_test, preds, zero_division=0)
    r = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    acc = accuracy_score(y_test, preds)
    print(f"{label} (n={len(features)}): C={best_C} thr={threshold:.2f} LOPO_F1={lopo_f1:.3f} | "
          f"Held-out P={p:.3f} R={r:.3f} F1={f1:.3f} Acc={acc:.3f}")


def main():
    manifest = build_manifest(config.PAPERS_DIR)

    train_rows = compute_for_papers("0.8", None, manifest)
    train_rows = [r for r in train_rows if r["Paper Name"] not in config.TEST_PAPERS_TEMP08]
    train_rows = train_rows + load_synthetic_feature_rows(top_k=TOP_K)
    print(f"Training set: {len(train_rows)} rows.")

    test08_rows = compute_for_papers("0.8", config.TEST_PAPERS_TEMP08, manifest)
    test0_rows = compute_for_papers("0", config.TEST_PAPERS, manifest)
    test_rows = test0_rows + test08_rows
    print(f"Test08={len(test08_rows)}, Test0={len(test0_rows)}")

    print("\n=== Mutual information + ANOVA F-test (19 features, training set) ===")
    X_train, y_train, _ = to_arrays(train_rows, FEATURES_13_PLUS_4_PLUS_FP)
    discrete_mask = [len(set(X_train[:, i].tolist())) <= 3 for i in range(X_train.shape[1])]
    mi = mutual_info_classif(X_train, y_train, discrete_features=discrete_mask, random_state=0)
    fstat, pval = f_classif(X_train, y_train)
    ranking = sorted(zip(FEATURES_13_PLUS_4_PLUS_FP, mi, fstat, pval), key=lambda t: -t[1])
    for name, m, f, p in ranking:
        print(f"  {name:28s} MI={m:.4f}  F={f:8.2f}  p={p:.2e}")
    top4_sum = sum(m for _, m, _, _ in ranking[:4])
    rest_sum = sum(m for _, m, _, _ in ranking[4:])
    print(f"  Top-4 MI sum={top4_sum:.4f}, remaining-15 sum={rest_sum:.4f}, ratio={top4_sum/rest_sum:.2f}x")
    lowest3 = [n for n, _, _, _ in ranking[-3:]]
    print(f"  3 lowest-MI features: {lowest3}")

    print("\n=== 19 vs 16 (drop the 3 lowest-MI features) ===")
    run_logreg(train_rows, test_rows, FEATURES_13_PLUS_4_PLUS_FP, "19 FEATURES (final)")
    features_16 = [f for f in FEATURES_13_PLUS_4_PLUS_FP if f not in lowest3]
    run_logreg(train_rows, test_rows, features_16, "16 FEATURES (dropped 3 lowest MI)")

    print("\n=== Classifier choice (19 features) ===")
    X_train19, y_train19, groups_train19 = to_arrays(train_rows, FEATURES_13_PLUS_4_PLUS_FP)
    X_test19, y_test19, _ = to_arrays(test_rows, FEATURES_13_PLUS_4_PLUS_FP)
    classifiers = {
        "Logistic regression": lambda: LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000),
        "Random forest": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=0),
        "Gradient boosting": lambda: GradientBoostingClassifier(random_state=0),
        "Support vector machine (RBF)": lambda: SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=0),
        "k-nearest neighbors": lambda: KNeighborsClassifier(n_neighbors=5),
        "Multi-layer perceptron": lambda: MLPClassifier(hidden_layer_sizes=(16,), max_iter=2000, random_state=0),
    }
    for name, make_clf in classifiers.items():
        oof = np.zeros(len(y_train19))
        for tr, te in LeaveOneGroupOut().split(X_train19, y_train19, groups_train19):
            m = make_pipeline(StandardScaler(), make_clf())
            m.fit(X_train19[tr], y_train19[tr])
            oof[te] = m.predict_proba(X_train19[te])[:, 1]
        threshold, lopo_f1 = best_threshold(y_train19, oof)
        final_model = make_pipeline(StandardScaler(), make_clf())
        final_model.fit(X_train19, y_train19)
        probs = final_model.predict_proba(X_test19)[:, 1]
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(y_test19, preds, zero_division=0)
        p = precision_score(y_test19, preds, zero_division=0)
        r = recall_score(y_test19, preds, zero_division=0)
        acc = accuracy_score(y_test19, preds)
        print(f"{name:32s} LOPO_F1={lopo_f1:.3f}  Held-out: P={p:.3f} R={r:.3f} F1={f1:.3f} Acc={acc:.3f}")

    print("\n=== Note: logistic regression above uses a fixed C=1.0 for a like-for-like ===")
    print("=== comparison across classifier families; the paper's reported LR row reuses ===")
    print("=== the grid-searched fit from paper_final_ourmodel.py (C=0.3) instead. ===")


if __name__ == "__main__":
    main()
