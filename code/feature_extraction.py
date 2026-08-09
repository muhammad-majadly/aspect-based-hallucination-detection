"""Builds a small, fixed feature vector for a single (paper, aspect,
sentence) triple, to be fed to a learned aggregator (see
train_aggregator.py) instead of the hand-tuned threshold rule in
nli_backend.classify_sentence_nli.

Every feature here is derived either from the existing retrieval+NLI
pipeline (similarity scores, entailment/neutral/contradiction
probabilities) at no extra model-inference cost, or from cheap
regex/lexical checks against the full paper text -- no new models, no
LLMs, consistent with the project's small-model constraint.

Deliberately kept to a small, curated set (13 features) rather than a
large hand-engineered bank: with ~150-190 positive examples across ~40
papers, a wide feature set risks overfitting long before it risks
underfitting.
"""
import math
import re

from evidence import get_evidence, get_full_paper_text
from nli_backend import get_candidate_comparisons

FEATURE_TOP_K = 6  # top_k=6 validated as best (see run_lr_13plus4_topk.py: +0.015
# combined F1 over top_k=5 for the 17-feature retrain config). NOTE: the
# frozen shipped aggregator was trained at top_k=5 and run_our_model.py's
# run_frozen() hardcodes FROZEN_TOP_K=5 independent of this module default
# -- only --retrain (RETRAIN_TOP_K) and ad-hoc scripts pick up this value.

# Original 13-feature set (shipped in the frozen aggregator).
BASE_FEATURE_NAMES = [
    "sim_max",
    "sim_mean_top3",
    "sim_gap_top1_top2",
    "ent_max",
    "contra_max",
    "neutral_mean",
    "margin_ent_contra",
    "entropy_top1",
    "n_contradiction_soft",
    "n_entailment_soft",
    "number_overlap",
    "novel_capitalized_entity",
    "aspect_section_hit",
]

# Additional features, all derived from the same `comparisons` list the base
# features already use (no extra model-inference cost) -- they preserve
# per-rank and cross-candidate NLI information that the base set collapses
# into a handful of aggregate stats (e.g. ent_max/contra_max are the max
# over ALL top-k candidates, discarding which rank they came from and how
# much the runner-up candidates agreed or disagreed).
EXTRA_FEATURE_NAMES = [
    "ent_top1",
    "contra_top1",
    "neutral_top1",
    "ent_2nd_max",
    "contra_2nd_max",
    "ent_mean_top3",
    "contra_mean_top3",
    "ent_std",
    "contra_std",
    "sim_weighted_ent",
    "sim_weighted_contra",
    "ent_gap_top1_top2",
    "sentence_length",
    "lexical_overlap_top1",
]

# Features motivated by manually inspecting the 13-feature logistic
# regression's false negatives on the temp=0.8 held-out test: recurring
# fabrication patterns the original 13 features structurally can't see.
ERROR_ANALYSIS_FEATURE_NAMES = [
    "acronym_mismatch",
    "number_overlap_evidence",
    "self_reference_leak",
    "is_question",
]

# Features motivated by the temp=0 false negatives: a large share of them
# were exact/near-exact sentences repeated within the same paper's summary
# (greedy-decoding repetition loops), plus mid-clause truncations.
# `is_duplicate_in_summary` is NOT computable inside compute_features() --
# it needs the
# other sentences generated for the same paper, i.e. sibling dataset rows,
# not just this (paper, aspect, sentence) triple -- so it's filled in by
# add_duplicate_feature() below as a post-processing pass over a full list
# of feature rows, not returned by compute_features() itself.
TEMP0_ERROR_ANALYSIS_FEATURE_NAMES = [
    "is_duplicate_in_summary",
    "missing_terminal_punctuation",
]

# NLI-probability features filling two gaps discussed directly: (1) the
# "min across the top-5" statistic didn't exist for ANY label -- ent_max/
# contra_max are dominated by whichever single candidate looks best, even
# if the other 4 disagree; ent_min/contra_min instead ask "does EVEN THE
# WEAKEST of the 5 still agree" -- a much stronger unanimity signal than a
# raw count like n_entailment_soft. (2) neutral only had neutral_mean/
# neutral_top1, versus a full max/std/similarity-weighted family for
# entailment and contradiction -- neutral_max/neutral_std/
# sim_weighted_neutral bring it to the same richness.
NLI_STATS_FEATURE_NAMES = [
    "ent_min",
    "contra_min",
    "neutral_min",
    "neutral_max",
    "neutral_std",
    "sim_weighted_neutral",
]

# Features motivated by a false-positive error analysis: faithful-but-vague
# sentences the pipeline structurally can't distinguish from fabrications,
# because there's no concrete claim for NLI to confirm.
FP_ERROR_ANALYSIS_FEATURE_NAMES = [
    "hedge_opinion",
    "meta_commentary",
]

FEATURE_NAMES = (
    BASE_FEATURE_NAMES + EXTRA_FEATURE_NAMES + ERROR_ANALYSIS_FEATURE_NAMES
    + TEMP0_ERROR_ANALYSIS_FEATURE_NAMES + NLI_STATS_FEATURE_NAMES + FP_ERROR_ANALYSIS_FEATURE_NAMES
)

# The 13 original features plus the 4 error-analysis features above --
# deliberately excludes EXTRA_FEATURE_NAMES (shown earlier not to help).
FEATURES_13_PLUS_4 = BASE_FEATURE_NAMES + ERROR_ANALYSIS_FEATURE_NAMES

# The above plus the 2 duplicate/truncation features from the temp=0 pass.
FEATURES_13_PLUS_6 = FEATURES_13_PLUS_4 + TEMP0_ERROR_ANALYSIS_FEATURE_NAMES

# The 17-feature best-so-far set (13 base + 4 error-analysis) plus the 2
# false-positive-targeting features -- 19 features total.
FEATURES_13_PLUS_4_PLUS_FP = FEATURES_13_PLUS_4 + FP_ERROR_ANALYSIS_FEATURE_NAMES

# The original 13 plus the 6 NLI-probability features above.
FEATURES_13_PLUS_NLISTATS = BASE_FEATURE_NAMES + NLI_STATS_FEATURE_NAMES


def _normalize_for_dedup(sentence):
    return re.sub(r"[^a-z0-9 ]", "", sentence.lower()).strip()


def add_duplicate_feature(rows):
    """Post-processing pass: sets is_duplicate_in_summary=1.0 on every row
    whose (near-)normalized sentence text recurs more than once among the
    rows sharing the same Paper Name, else 0.0. Mutates and returns `rows`
    (each must be a dict with "Paper Name" and "Sentence" keys, e.g. the
    output of run_our_model.compute_feature_rows)."""
    counts = {}
    for r in rows:
        key = (r["Paper Name"], _normalize_for_dedup(r["Sentence"]))
        counts[key] = counts.get(key, 0) + 1
    for r in rows:
        key = (r["Paper Name"], _normalize_for_dedup(r["Sentence"]))
        r["is_duplicate_in_summary"] = 1.0 if counts[key] > 1 else 0.0
    return rows

_STOPWORDS = {
    "the", "a", "an", "of", "to", "and", "in", "on", "for", "with", "as",
    "is", "are", "was", "were", "be", "been", "being", "by", "at", "from",
    "that", "this", "these", "those", "it", "its", "we", "our", "which",
    "or", "but", "not", "can", "will", "than", "into", "such", "also",
    "have", "has", "had", "their", "they", "then", "so", "over", "more",
    "other", "each", "using", "used", "use", "based", "both",
}

_NUMBER_RE = re.compile(r"\d+\.?\d*")
# Proper-noun-like (capitalized word, len>=4) or acronym-like (all-caps, len 2-6) tokens.
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]{3,}[A-Za-z0-9]*|[A-Z]{2,6}[0-9]*)\b")
_COMMON_CAPITALIZED = {
    "The", "This", "That", "These", "Those", "Our", "We", "It", "Its", "In",
    "For", "As", "However", "Therefore", "Moreover", "Furthermore", "Additionally",
    "Given", "Based", "Using", "With", "While", "Since", "First", "Second",
    "Third", "Finally", "Overall", "Specifically", "Notably", "Table", "Figure",
    "Section", "Fig",
}


def _entropy(probs):
    return -sum(p * math.log(p) for p in probs.values() if p > 1e-12)


def _normalize(text):
    return " ".join(text.split())


def _number_overlap(sentence, paper_text_norm):
    numbers = _NUMBER_RE.findall(sentence)
    if not numbers:
        return 1.0  # no numeric claims to verify -- don't penalize
    hits = sum(1 for n in numbers if n in paper_text_norm)
    return hits / len(numbers)


def _novel_capitalized_entity(sentence, paper_text_norm):
    sentence = sentence.strip()
    candidates = [
        m.group()
        for m in _ENTITY_RE.finditer(sentence)
        if m.start() != 0 and m.group() not in _COMMON_CAPITALIZED  # skip sentence-initial capitalization
    ]
    if not candidates:
        return 0.0
    return 1.0 if any(tok not in paper_text_norm for tok in candidates) else 0.0


def _full_reference_text(paper):
    """All text available for the paper, including the title and
    sections normally excluded from NLI evidence (FrontMatter,
    References, Acknowledgments) -- used only for the surface-level
    fabrication checks below (number/entity overlap), where we want the
    most permissive "does this string appear ANYWHERE in the source
    material" check to avoid false positives on things like a sentence
    quoting the paper's own title (which lives in FrontMatter, excluded
    from NLI evidence but obviously not a hallucination)."""
    parts = [paper.get("paper_title", "")]
    parts += [" ".join(s.get("chunks", [])) for s in paper["sections"]]
    return _normalize(" ".join(parts))


def _aspect_section_hit(top_candidate_text, expected_text_norm):
    if not top_candidate_text or not expected_text_norm:
        return 0.0
    return 1.0 if _normalize(top_candidate_text) in expected_text_norm else 0.0


def _content_words(text):
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _lexical_overlap(sentence, evidence_text):
    sent_words = _content_words(sentence)
    if not sent_words:
        return 0.0
    evid_words = _content_words(evidence_text)
    return len(sent_words & evid_words) / len(sent_words)


def _std(values):
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _weighted_mean(values, weights):
    total_w = sum(weights)
    if total_w <= 0:
        return 0.0
    return sum(v * w for v, w in zip(values, weights)) / total_w


# Matches "ACRONYM (Spelled Out Words)", e.g. "FATE (Full-Head Avatar
# Reconstruction and Completion)" -- deliberately requires an all-uppercase,
# all-letter acronym (no digits) to avoid firing on things like "3DGS (3D
# Gaussian Splatting)", where digit-containing acronyms make an initials
# comparison meaningless.
_ACRONYM_DEF_RE = re.compile(r"\b([A-Z]{3,8})\s*\(([^)]{4,80})\)")


def _acronym_mismatch(sentence):
    """1.0 if the sentence spells out an acronym whose expansion's initials
    don't match the acronym's own letters (a fabricated/garbled acronym
    expansion), else 0.0. Connector words (and/of/the/...) are dropped
    before comparing initials, since real acronyms often skip them (e.g.
    NASA)."""
    for m in _ACRONYM_DEF_RE.finditer(sentence):
        acronym = m.group(1)
        words = [w for w in re.findall(r"[A-Za-z]+", m.group(2)) if w.lower() not in _STOPWORDS]
        if not words:
            continue
        initials = "".join(w[0] for w in words).upper()
        if initials != acronym.upper():
            return 1.0
    return 0.0


def _number_overlap_evidence(sentence, evidence_windows_text_norm):
    """Like _number_overlap, but restricted to the retrieved top-k evidence
    windows rather than the whole paper -- a number that only appears
    elsewhere in the paper (different table, different benchmark) shouldn't
    count as supporting THIS claim."""
    numbers = _NUMBER_RE.findall(sentence)
    if not numbers:
        return 1.0
    hits = sum(1 for n in numbers if n in evidence_windows_text_norm)
    return hits / len(numbers)


# Terms that would never legitimately appear in a paper describing its own
# contribution/method -- their presence means the summarizer leaked a
# reference to itself rather than describing the paper.
_SELF_REFERENCE_RE = re.compile(
    r"\b(qwen|chatgpt|gpt-?\d\w*|claude|gemini|as an ai|the assistant)\b", re.IGNORECASE
)


def _self_reference_leak(sentence):
    return 1.0 if _SELF_REFERENCE_RE.search(sentence) else 0.0


def _is_question(sentence):
    return 1.0 if sentence.strip().endswith("?") else 0.0


def _missing_terminal_punctuation(sentence):
    """1.0 if the sentence doesn't end with sentence-final punctuation --
    a cheap truncation/mid-clause-cutoff detector."""
    return 0.0 if sentence.strip().endswith((".", "!", "?", '"', ")")) else 1.0


# "The authors believe/think/hope X" attributes an OPINION/EXPECTATION to
# the authors, not a factual claim from the paper -- NLI can't confirm or
# deny a sentiment, so it tends to read as unsupported/neutral even when
# it's a faithful paraphrase of the authors' own stated hopes.
_HEDGE_OPINION_RE = re.compile(
    r"\b(the authors|we)\s+(believe|think|hope|expect|suggest)s?\s+(that\s+)?", re.IGNORECASE
)


def _hedge_opinion(sentence):
    return 1.0 if _HEDGE_OPINION_RE.search(sentence) else 0.0


# Sentences describing the PAPER'S OWN EXPOSITION ("the paper presents an
# overview...", "we summarize the experimental results...") rather than
# making a technical claim -- vague-but-faithful restatements with nothing
# concrete to verify, which the pipeline structurally can't distinguish
# from a vague fabrication. Deliberately narrow -- does NOT match "we
# present/introduce a novel method for X", the most common legitimate
# contribution-sentence template, which names a specific method/technique
# rather than just referring back to "overview"/"summary".
_META_COMMENTARY_RE = re.compile(
    r"\b(the paper|this paper|the authors)\s+(presents?|provides?|introduces?|discusses?|outlines?|summarizes?)\b"
    r"|\bwe\s+(provide|present|summarize|describe|outline)\s+(a\s+)?(comprehensive|detailed|brief)?\s*(overview|summary)\b"
    r"|\bwe\s+summarize\s+the\s+(experimental\s+)?(results|findings|conclusions)\b",
    re.IGNORECASE,
)


def _meta_commentary(sentence):
    return 1.0 if _META_COMMENTARY_RE.search(sentence) else 0.0


def compute_features(paper, aspect, sentence, top_k=FEATURE_TOP_K):
    """Returns a dict of FEATURE_NAMES -> float for one summary sentence."""
    full_text, _truncated = get_full_paper_text(paper)
    comparisons = get_candidate_comparisons(full_text, sentence, top_k)

    if not comparisons:
        # No evidence sentences at all (e.g. empty paper sections) -- return
        # a neutral/uninformative feature vector rather than crashing.
        return {
            "sim_max": 0.0, "sim_mean_top3": 0.0, "sim_gap_top1_top2": 0.0,
            "ent_max": 0.0, "contra_max": 0.0, "neutral_mean": 1.0,
            "margin_ent_contra": 0.0, "entropy_top1": math.log(3),
            "n_contradiction_soft": 0.0, "n_entailment_soft": 0.0,
            "number_overlap": 1.0, "novel_capitalized_entity": 0.0,
            "aspect_section_hit": 0.0,
            "ent_top1": 0.0, "contra_top1": 0.0, "neutral_top1": 1.0,
            "ent_2nd_max": 0.0, "contra_2nd_max": 0.0,
            "ent_mean_top3": 0.0, "contra_mean_top3": 0.0,
            "ent_std": 0.0, "contra_std": 0.0,
            "sim_weighted_ent": 0.0, "sim_weighted_contra": 0.0,
            "ent_gap_top1_top2": 0.0,
            "sentence_length": float(len(sentence.split())),
            "lexical_overlap_top1": 0.0,
            "acronym_mismatch": _acronym_mismatch(sentence),
            "number_overlap_evidence": 1.0 if not _NUMBER_RE.findall(sentence) else 0.0,
            "self_reference_leak": _self_reference_leak(sentence),
            "is_question": _is_question(sentence),
            "is_duplicate_in_summary": 0.0,  # filled in by add_duplicate_feature()
            "missing_terminal_punctuation": _missing_terminal_punctuation(sentence),
            "ent_min": 0.0, "contra_min": 0.0, "neutral_min": 1.0,
            "neutral_max": 1.0, "neutral_std": 0.0, "sim_weighted_neutral": 1.0,
            "hedge_opinion": _hedge_opinion(sentence),
            "meta_commentary": _meta_commentary(sentence),
        }

    similarities = [c["similarity"] for c in comparisons]
    ent_probs = [c["probs"].get("ENTAILMENT", 0.0) for c in comparisons]
    contra_probs = [c["probs"].get("CONTRADICTION", 0.0) for c in comparisons]
    neutral_probs = [c["probs"].get("NEUTRAL", 0.0) for c in comparisons]

    top3_sims = similarities[:3]
    ent_max = max(ent_probs)
    contra_max = max(contra_probs)
    ent_sorted = sorted(ent_probs, reverse=True)
    contra_sorted = sorted(contra_probs, reverse=True)

    reference_text_norm = _full_reference_text(paper)
    expected_text, _t, _names = get_evidence(paper, aspect)
    expected_text_norm = _normalize(expected_text)

    return {
        "sim_max": similarities[0],
        "sim_mean_top3": sum(top3_sims) / len(top3_sims),
        "sim_gap_top1_top2": (similarities[0] - similarities[1]) if len(similarities) > 1 else 0.0,
        "ent_max": ent_max,
        "contra_max": contra_max,
        "neutral_mean": sum(neutral_probs) / len(neutral_probs),
        "margin_ent_contra": ent_max - contra_max,
        "entropy_top1": _entropy(comparisons[0]["probs"]),
        "n_contradiction_soft": float(sum(1 for p in contra_probs if p > 0.5)),
        "n_entailment_soft": float(sum(1 for p in ent_probs if p > 0.5)),
        "number_overlap": _number_overlap(sentence, reference_text_norm),
        "novel_capitalized_entity": _novel_capitalized_entity(sentence, reference_text_norm),
        "aspect_section_hit": _aspect_section_hit(comparisons[0]["text"], expected_text_norm),
        "ent_top1": ent_probs[0],
        "contra_top1": contra_probs[0],
        "neutral_top1": neutral_probs[0],
        "ent_2nd_max": ent_sorted[1] if len(ent_sorted) > 1 else 0.0,
        "contra_2nd_max": contra_sorted[1] if len(contra_sorted) > 1 else 0.0,
        "ent_mean_top3": sum(ent_probs[:3]) / len(ent_probs[:3]),
        "contra_mean_top3": sum(contra_probs[:3]) / len(contra_probs[:3]),
        "ent_std": _std(ent_probs),
        "contra_std": _std(contra_probs),
        "sim_weighted_ent": _weighted_mean(ent_probs, similarities),
        "sim_weighted_contra": _weighted_mean(contra_probs, similarities),
        "ent_gap_top1_top2": (ent_probs[0] - ent_probs[1]) if len(ent_probs) > 1 else 0.0,
        "sentence_length": float(len(sentence.split())),
        "lexical_overlap_top1": _lexical_overlap(sentence, comparisons[0]["text"]),
        "acronym_mismatch": _acronym_mismatch(sentence),
        "number_overlap_evidence": _number_overlap_evidence(
            sentence, _normalize(" ".join(c["text"] for c in comparisons[:3]))
        ),
        "self_reference_leak": _self_reference_leak(sentence),
        "is_question": _is_question(sentence),
        "is_duplicate_in_summary": 0.0,  # filled in by add_duplicate_feature()
        "missing_terminal_punctuation": _missing_terminal_punctuation(sentence),
        "ent_min": ent_sorted[-1],
        "contra_min": contra_sorted[-1],
        "neutral_min": min(neutral_probs),
        "neutral_max": max(neutral_probs),
        "neutral_std": _std(neutral_probs),
        "sim_weighted_neutral": _weighted_mean(neutral_probs, similarities),
        "hedge_opinion": _hedge_opinion(sentence),
        "meta_commentary": _meta_commentary(sentence),
    }
