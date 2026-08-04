"""Retrieves the evidence text used to check a summary sentence.

get_full_paper_text() returns the (nearly) full paper text and is the
DEFAULT evidence source for GPT-4-turbo, GPT-4o, Gemma-2-2b-it, Llama-3.2-1B,
and "Our Model"'s retrieval step, per the paper's Experiments section.

get_evidence() implements an OPTIONAL, fixed aspect -> section mapping:

    Motivation    -> Abstract, Introduction
    Contribution  -> Abstract, Introduction, Conclusion
    Methodology   -> every section after Related Work
    Main Results  -> Abstract, Introduction, Experiments/Results
    Paper Goal    -> Abstract, Introduction

It is disabled by default (config.USE_ASPECT_MAPPING = False) because the
final experiments use the full paper text. SummaC and MiniCheck (run_summac.py,
run_minicheck.py) call get_evidence() directly regardless of this flag, since
their NLI/fact-checking backends are intractably slow against a full paper --
see those scripts' docstrings.

Section names vary across papers (e.g. "Method" vs "Methodology" vs
"Approach", "Related Work" vs "Related work"), so sections are matched
case-insensitively by pattern rather than exact string.
"""
import re

from config import MAX_EVIDENCE_CHARS, MAX_FULL_PAPER_CHARS

_ABSTRACT = re.compile(r"^abstract$", re.I)
_INTRO = re.compile(r"^introduction$", re.I)
_CONCLUSION = re.compile(r"^conclusion", re.I)
_RELATED_WORK = re.compile(r"related\s*work", re.I)
_RESULTS_EXPERIMENTS = re.compile(r"result|experiment", re.I)
_METHOD_LIKE = re.compile(r"^method|approach", re.I)
# Never useful as evidence, regardless of aspect.
_ALWAYS_EXCLUDE = re.compile(r"^reference|^acknowledg|^frontmatter", re.I)


def _section_text(section):
    return " ".join(section.get("chunks", []))


def _filter_excluded(sections):
    return [s for s in sections if not _ALWAYS_EXCLUDE.match(s["section"])]


def _select_by_predicate(sections, predicate):
    return _filter_excluded([s for s in sections if predicate(s["section"])])


def _methodology_sections(sections):
    start = next(
        (i for i, s in enumerate(sections) if _RELATED_WORK.search(s["section"])),
        None,
    )
    if start is not None:
        picked = sections[start + 1 :]
    else:
        # Fallback 1: no "Related Work" section found -- use Method-like sections.
        picked = [s for s in sections if _METHOD_LIKE.match(s["section"])]
        if not picked:
            # Fallback 2: everything except abstract/intro/conclusion.
            picked = [
                s
                for s in sections
                if not (
                    _ABSTRACT.match(s["section"])
                    or _INTRO.match(s["section"])
                    or _CONCLUSION.match(s["section"])
                )
            ]
    return _filter_excluded(picked)


def get_evidence(paper, aspect):
    """Returns (evidence_text, truncated, matched_section_names) restricted
    to the aspect's mapped sections (see module docstring)."""
    sections = paper["sections"]

    if aspect in ("Motivation", "Paper Goal", "Paper-Goal"):
        picked = _select_by_predicate(
            sections, lambda n: _ABSTRACT.match(n) or _INTRO.match(n)
        )
    elif aspect == "Contribution":
        picked = _select_by_predicate(
            sections,
            lambda n: _ABSTRACT.match(n) or _INTRO.match(n) or _CONCLUSION.match(n),
        )
    elif aspect == "Main Results":
        picked = _select_by_predicate(
            sections,
            lambda n: _ABSTRACT.match(n)
            or _INTRO.match(n)
            or _RESULTS_EXPERIMENTS.search(n),
        )
    elif aspect == "Methodology":
        picked = _methodology_sections(sections)
    else:
        raise ValueError(f"Unknown aspect: {aspect!r}")

    matched_names = [s["section"] for s in picked]
    text = "\n\n".join(f"[{s['section']}]\n{_section_text(s)}" for s in picked)

    truncated = len(text) > MAX_EVIDENCE_CHARS
    if truncated:
        text = text[:MAX_EVIDENCE_CHARS]

    return text, truncated, matched_names


def get_full_paper_text(paper, max_chars=MAX_FULL_PAPER_CHARS):
    """Returns the (nearly) full paper text -- every section except
    References/Acknowledgments/FrontMatter."""
    sections = _filter_excluded(paper["sections"])
    text = "\n\n".join(f"[{s['section']}]\n{_section_text(s)}" for s in sections)
    truncated = len(text) > max_chars
    if truncated:
        text = text[:max_chars]
    return text, truncated
