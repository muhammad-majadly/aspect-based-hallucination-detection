"""Thin wrapper around the OpenAI chat completion API for the
hallucination-detection prompt."""
import os
import re
import time
from pathlib import Path

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a careful fact-checker for scientific paper summaries. Given "
    "the full text of a scientific paper and a provided summary, classify "
    "each sentence in the summary as either 'Hallucinated' or 'Not "
    "Hallucinated'. Output only the classification for each sentence "
    "without any justification or explanation."
)

USER_TEMPLATE = """Full text of the scientific paper:
\"\"\"
{paper_text}
\"\"\"

Summary sentence to classify:
\"\"\"
{sentence}
\"\"\"

Classify the summary sentence as either 'Hallucinated' or 'Not \
Hallucinated'. Output only the classification, without any justification \
or explanation."""

BATCH_SYSTEM_PROMPT = (
    "You are a careful fact-checker for scientific paper summaries. Given "
    "the full text of a scientific paper and a numbered list of summary "
    "sentences about it, classify EACH sentence as either 'Hallucinated' "
    "or 'Not Hallucinated'. Output exactly one line per sentence, in the "
    "format 'N: Hallucinated' or 'N: Not Hallucinated' (N = the sentence "
    "number), in order, with no other text, explanation, or blank lines."
)

BATCH_USER_TEMPLATE = """Full text of the scientific paper:
\"\"\"
{paper_text}
\"\"\"

Summary sentences to classify:
{numbered_sentences}

Classify each of the {n} sentences above as either 'Hallucinated' or 'Not \
Hallucinated'. Output exactly {n} lines, one per sentence, in the format \
'N: Hallucinated' or 'N: Not Hallucinated'. No other text."""

_BATCH_LINE_RE = re.compile(r"^\s*(\d+)\s*[:.\)]\s*(.+?)\s*$")


def _load_api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key
    key_file = Path(__file__).resolve().parent / "api_key.txt"
    if key_file.exists():
        key = key_file.read_text().strip()
        if key:
            return key
    raise RuntimeError(
        "No OpenAI API key found. Either set the OPENAI_API_KEY "
        "environment variable, or create a file named 'api_key.txt' in "
        "code/detection/ containing just the key. See README.md."
    )


def get_client():
    return OpenAI(api_key=_load_api_key())


def _parse_verdict(raw):
    text = raw.lower()
    if "not" in text:
        return 0
    if "halluc" in text:
        return 1
    return None  # model gave an unparseable answer


def classify_sentence(client, model, paper_text, sentence, max_retries=3):
    """Returns (verdict, raw_response). verdict is 1 (hallucinated),
    0 (not hallucinated), or None if the call failed or the response
    could not be parsed."""
    prompt = USER_TEMPLATE.format(paper_text=paper_text, sentence=sentence)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=10,
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_verdict(raw), raw
        except Exception as e:  # network/rate-limit/API errors
            last_err = e
            time.sleep(2**attempt)
    return None, f"ERROR: {last_err}"


def classify_paper_sentences(client, model, paper_text, sentences, max_retries=3):
    """Classifies every sentence in `sentences` with a single API call:
    the full paper text plus the whole numbered list of summary
    sentences is sent once, and the model returns one classification per
    line. Returns a list of (verdict, raw_line) aligned with `sentences`
    (verdict is 1/0/None per the same convention as classify_sentence).

    This is both the paper's literal baseline setup ("given the full
    text... and a provided summary, classify each sentence...") and far
    cheaper than one call per sentence, since the paper text is sent once
    per paper instead of once per sentence.
    """
    n = len(sentences)
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sentences))
    prompt = BATCH_USER_TEMPLATE.format(paper_text=paper_text, numbered_sentences=numbered, n=n)

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=max(50, n * 15),
            )
            raw = resp.choices[0].message.content.strip()
            return _parse_batch_response(raw, n)
        except Exception as e:  # network/rate-limit/API errors
            last_err = e
            time.sleep(2**attempt)
    return [(None, f"ERROR: {last_err}")] * n


def _parse_batch_response(raw, n):
    """Maps a 'N: Hallucinated' / 'N: Not Hallucinated' per-line response
    back to a (verdict, raw_line) list of length n, by sentence number --
    robust to the model skipping/misnumbering a line, since each line is
    matched by its own number rather than by position."""
    by_number = {}
    for line in raw.splitlines():
        m = _BATCH_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        by_number[idx] = m.group(2)

    results = []
    for i in range(1, n + 1):
        line = by_number.get(i)
        if line is None:
            results.append((None, "MISSING (no line returned for this sentence)"))
        else:
            results.append((_parse_verdict(line), line))
    return results
