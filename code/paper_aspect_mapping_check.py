"""Diagnostic for the paper's aspect-to-section mapping (evidence.py's
get_evidence): compares the current mapping (Motivation/Paper Goal ->
Abstract+Intro; Contribution -> +Conclusion; Main Results -> +Intro;
Methodology -> everything after Related Work) against a narrower
alternative rule (Motivation/Contribution/Paper Goal -> Abstract+Intro;
Methodology -> Abstract+Method-like section; Main Results ->
Abstract+Results/Experiments), and measures whether swapping the
aspect_section_hit feature to the alternative rule changes held-out F1.

Two independent checks:
  1. Section-name diff across all 50 papers (cheap, no model inference).
  2. Refit logistic regression with aspect_section_hit recomputed under
     each rule, holding all other 18 features fixed at their top_k=7
     values -- isolates the effect of this one feature.
"""
import json as jsonlib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from dataset_utils import load_dataset_rows, resolve_matched_rows
from evidence import (
    _ABSTRACT,
    _INTRO,
    _METHOD_LIKE,
    _RESULTS_EXPERIMENTS,
    _select_by_predicate,
    get_evidence,
    get_full_paper_text,
)
from feature_extraction import FEATURES_13_PLUS_4_PLUS_FP, _aspect_section_hit, _normalize
from manifest import build_manifest, normalize_title
from nli_backend import split_into_windows, top_n_similar
from run_our_model import compute_feature_rows, load_synthetic_feature_rows

TOP_K = 7
ASPECTS = ["Motivation", "Contribution", "Methodology", "Main Results", "Paper Goal"]


def get_evidence_alt_rule_text(paper, aspect):
    sections = paper["sections"]
    if aspect in ("Motivation", "Contribution", "Paper Goal", "Paper-Goal"):
        picked = _select_by_predicate(sections, lambda n: _ABSTRACT.match(n) or _INTRO.match(n))
    elif aspect == "Methodology":
        picked = _select_by_predicate(sections, lambda n: _ABSTRACT.match(n) or _METHOD_LIKE.match(n))
    elif aspect == "Main Results":
        picked = _select_by_predicate(sections, lambda n: _ABSTRACT.match(n) or _RESULTS_EXPERIMENTS.search(n))
    else:
        raise ValueError(aspect)
    return "\n\n".join(f"[{s['section']}]\n{' '.join(s.get('chunks', []))}" for s in picked)


def get_evidence_alt_rule_names(paper, aspect):
    sections = paper["sections"]
    if aspect in ("Motivation", "Contribution", "Paper Goal", "Paper-Goal"):
        picked = _select_by_predicate(sections, lambda n: _ABSTRACT.match(n) or _INTRO.match(n))
    elif aspect == "Methodology":
        picked = _select_by_predicate(sections, lambda n: _ABSTRACT.match(n) or _METHOD_LIKE.match(n))
    elif aspect == "Main Results":
        picked = _select_by_predicate(sections, lambda n: _ABSTRACT.match(n) or _RESULTS_EXPERIMENTS.search(n))
    else:
        raise ValueError(aspect)
    return [s["section"] for s in picked]


def check_section_diffs(manifest):
    print("=== Section-name differences, current mapping vs. alternative rule ===")
    seen = set()
    diffs = {a: 0 for a in ASPECTS}
    total = {a: 0 for a in ASPECTS}
    for key, (title, json_path) in sorted(manifest.items()):
        if title in seen:
            continue
        seen.add(title)
        paper = jsonlib.load(open(json_path))
        for aspect in ASPECTS:
            total[aspect] += 1
            try:
                _text, _trunc, old_names = get_evidence(paper, aspect)
            except Exception as e:
                old_names = [f"ERROR: {e}"]
            new_names = get_evidence_alt_rule_names(paper, aspect)
            if set(old_names) != set(new_names):
                diffs[aspect] += 1
    for a in ASPECTS:
        print(f"  {a:15s} differs on {diffs[a]}/{total[a]} papers")


def compute_for_papers(temperature, papers, manifest):
    rows = load_dataset_rows(config.DATASET_CSV, temperature=temperature)
    matched, _ = resolve_matched_rows(rows, manifest)
    if papers is not None:
        matched = [r for r in matched if r["Paper Name"] in papers]
    return compute_feature_rows(matched, manifest, top_k=TOP_K)


def check_model_impact(manifest):
    print("\n=== Effect on the 19-feature model of swapping aspect_section_hit ===")
    synthetic_lookup = {}
    synthetic_dir = config.ROOT / "data" / "synthetic"
    for p in sorted(synthetic_dir.glob("synth_*.json")):
        data = jsonlib.loads(p.read_text())
        synthetic_lookup[normalize_title(data["paper_title"])] = data

    paper_cache = {}

    def get_paper(paper_name):
        if paper_name not in paper_cache:
            norm = normalize_title(paper_name)
            if norm in synthetic_lookup:
                paper_cache[paper_name] = synthetic_lookup[norm]
            elif norm in manifest:
                _title, json_path = manifest[norm]
                paper_cache[paper_name] = jsonlib.load(open(json_path))
            else:
                raise KeyError(f"paper not found: {paper_name!r}")
        return paper_cache[paper_name]

    train_rows = compute_for_papers("0.8", None, manifest)
    train_rows = [r for r in train_rows if r["Paper Name"] not in config.TEST_PAPERS_TEMP08]
    train_rows = train_rows + load_synthetic_feature_rows(top_k=TOP_K)
    test08_rows = compute_for_papers("0.8", config.TEST_PAPERS_TEMP08, manifest)
    test0_rows = compute_for_papers("0", config.TEST_PAPERS, manifest)

    all_rows = {"train": train_rows, "test08": test08_rows, "test0": test0_rows}
    changed = 0
    total = 0
    for split, rows in all_rows.items():
        full_text_cache = {}
        for r in rows:
            paper = get_paper(r["Paper Name"])
            if r["Paper Name"] not in full_text_cache:
                full_text_cache[r["Paper Name"]] = get_full_paper_text(paper)[0]
            full_text = full_text_cache[r["Paper Name"]]
            windows = split_into_windows(full_text, config.EVIDENCE_WINDOW_SIZE)
            top1 = top_n_similar(r["Sentence"], windows, 1)
            top1_text = top1[0][0] if top1 else ""

            new_expected = get_evidence_alt_rule_text(paper, r["Aspect"])
            new_hit = _aspect_section_hit(top1_text, _normalize(new_expected))
            r["_alt_aspect_section_hit"] = new_hit
            total += 1
            if new_hit != float(r["aspect_section_hit"]):
                changed += 1
    print(f"  {changed}/{total} rows have a different aspect_section_hit under the alternative rule")

    def to_arrays(rows, hit_key):
        X = np.array([
            [float(r[f]) if f != "aspect_section_hit" else float(r[hit_key]) for f in FEATURES_13_PLUS_4_PLUS_FP]
            for r in rows
        ])
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

    def run(hit_key, label):
        test_rows = test0_rows + test08_rows
        X_train, y_train, groups_train = to_arrays(train_rows, hit_key)
        best_C, best_C_f1, best_oof = None, -1.0, None
        for C in config.C_GRID:
            oof = lopo_oof_probs(X_train, y_train, groups_train, C)
            _, f1 = best_threshold(y_train, oof)
            if f1 > best_C_f1:
                best_C, best_C_f1, best_oof = C, f1, oof
        threshold, lopo_f1 = best_threshold(y_train, best_oof)
        final_model = make_model(best_C)
        final_model.fit(X_train, y_train)
        X_test, y_test, _ = to_arrays(test_rows, hit_key)
        probs = final_model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f1 = f1_score(y_test, preds, zero_division=0)
        acc = accuracy_score(y_test, preds)
        print(f"  {label}: C={best_C} thr={threshold:.2f} LOPO_F1={lopo_f1:.3f} | "
              f"Held-out P={p:.3f} R={r:.3f} F1={f1:.3f} Acc={acc:.3f}")

    run("aspect_section_hit", "current mapping")
    run("_alt_aspect_section_hit", "alternative rule")


def main():
    manifest = build_manifest(config.PAPERS_DIR)
    check_section_diffs(manifest)
    check_model_impact(manifest)


if __name__ == "__main__":
    main()
