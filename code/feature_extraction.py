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

FEATURE_TOP_K = 5  # richer than config.TOP_N (3) -- more candidates to compute stats over

FEATURE_NAMES = [
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
        }

    similarities = [c["similarity"] for c in comparisons]
    ent_probs = [c["probs"].get("ENTAILMENT", 0.0) for c in comparisons]
    contra_probs = [c["probs"].get("CONTRADICTION", 0.0) for c in comparisons]
    neutral_probs = [c["probs"].get("NEUTRAL", 0.0) for c in comparisons]

    top3_sims = similarities[:3]
    ent_max = max(ent_probs)
    contra_max = max(contra_probs)

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
    }
