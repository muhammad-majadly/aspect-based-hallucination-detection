"""Local implementation of "Our Model"'s detection steps 3-5: sentence-
similarity retrieval, RoBERTa/DeBERTa NLI classification, and the paper's
contradiction > entailment > neutral-threshold aggregation rule.
Step 2 (Aspect-Text Mapping) is replaced by the fixed rule in evidence.py,
per the user's instruction; everything downstream runs exactly as
described in the paper, using off-the-shelf models in place of the
paper's own fine-tuned checkpoints (see config.py and README.md).

Retrieval uses a proper sentence-embedding model (sentence-transformers),
not raw SciBERT mean-pooling: raw BERT-family embeddings are known to be
"anisotropic" -- nearly every sentence pair scores 0.7+ cosine similarity
regardless of actual relatedness -- which made top-n retrieval close to
random and fed the NLI model weakly-relevant pairs. This was the main
driver of poor precision in the first version of this pipeline.

No API key needed -- everything runs locally via `transformers` /
`sentence-transformers`.
"""
import re

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
if torch.cuda.is_available():
    _DEVICE = "cuda"
elif torch.backends.mps.is_available():
    _DEVICE = "mps"
else:
    _DEVICE = "cpu"

_embed_model = None
_nli_tokenizer = None
_nli_model = None


def _lazy_load():
    global _embed_model, _nli_tokenizer, _nli_model
    if _embed_model is None:
        print(f"Loading {config.EMBEDDING_MODEL} for sentence similarity (device={_DEVICE})...")
        _embed_model = SentenceTransformer(config.EMBEDDING_MODEL, device=_DEVICE)
    if _nli_model is None:
        print(f"Loading {config.NLI_MODEL} for entailment/contradiction/neutral (device={_DEVICE})...")
        _nli_tokenizer = AutoTokenizer.from_pretrained(config.NLI_MODEL)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(config.NLI_MODEL).to(_DEVICE).eval()


def split_sentences(text):
    text = " ".join(text.split())
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def split_into_windows(text, window_size):
    """Splits text into overlapping windows of `window_size` consecutive
    sentences (stride 1). A synthesized/paraphrased summary sentence often
    combines facts spread across a few consecutive source sentences, so a
    single isolated source sentence frequently doesn't textually match
    closely enough for NLI to call entailment, even when the claim is
    fully supported by its surrounding context. window_size=1 recovers
    the original single-sentence behavior."""
    sentences = split_sentences(text)
    if window_size <= 1:
        return sentences
    return [
        " ".join(sentences[i : i + window_size])
        for i in range(len(sentences) - window_size + 1)
    ] or sentences


def embed(sentences):
    """Normalized sentence-transformers embeddings for a list of sentences."""
    _lazy_load()
    if not sentences:
        return torch.empty(0, _embed_model.get_sentence_embedding_dimension())
    return _embed_model.encode(sentences, convert_to_tensor=True, normalize_embeddings=True).cpu()


_evidence_embed_cache = {}


def top_n_similar(sentence, evidence_sentences, n):
    """Returns the n evidence sentences most similar to `sentence`, as
    (sentence, cosine_similarity) pairs, highest similarity first.

    Evidence embeddings are cached by their exact sentence list: the same
    evidence windows recur across every summary sentence for a given
    paper/aspect, and re-embedding them from scratch on every call (as
    opposed to embedding once and reusing) was the dominant cost in this
    pipeline."""
    if not evidence_sentences:
        return []
    key = tuple(evidence_sentences)
    evid_emb = _evidence_embed_cache.get(key)
    if evid_emb is None:
        evid_emb = embed(evidence_sentences)
        _evidence_embed_cache[key] = evid_emb
    cand_emb = embed([sentence])[0]
    sims = torch.nn.functional.cosine_similarity(cand_emb.unsqueeze(0), evid_emb)
    ranked = sorted(zip(evidence_sentences, sims.tolist()), key=lambda x: -x[1])
    return ranked[:n]


@torch.no_grad()
def nli_predict(premise, hypothesis):
    """Returns (top_label, {label: probability}) for a single
    (premise, hypothesis) pair."""
    _lazy_load()
    inputs = _nli_tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=256
    ).to(_DEVICE)
    logits = _nli_model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1)
    id2label = _nli_model.config.id2label
    label_probs = {id2label[i].upper(): probs[i].item() for i in range(len(probs))}
    top_label = max(label_probs, key=label_probs.get)
    return top_label, label_probs


def get_candidate_comparisons(evidence_text, sentence, top_k):
    """Returns the raw top-k (evidence_window, similarity, label_probs)
    comparisons for `sentence` against `evidence_text`, without applying
    any aggregation/verdict rule -- used by feature_extraction.py to
    build features for the learned aggregator, independent of the
    hand-tuned rule in classify_sentence_nli below."""
    evidence_sentences = split_into_windows(evidence_text, config.EVIDENCE_WINDOW_SIZE)
    top = top_n_similar(sentence, evidence_sentences, top_k)
    return [
        {"text": evid, "similarity": sim, "probs": nli_predict(evid, sentence)[1]}
        for evid, sim in top
    ]


def classify_sentence_nli(evidence_text, sentence):
    """Implements the paper's NLI aggregation rule:
      - contradiction in any of the top-n comparisons -> hallucinated
      - else entailment in any of them                -> not hallucinated
      - else (all neutral) -> hallucinated iff mean neutral-label
        confidence across the n comparisons is >= NEUTRAL_CONFIDENCE_THRESHOLD

    Comparisons whose retrieval similarity is below MIN_SIMILARITY are
    dropped before aggregation -- a weakly-related "evidence" sentence
    shouldn't be allowed to trigger a contradiction/entailment verdict.

    Returns (verdict, detail) where verdict is 1 (hallucinated) or 0 (not
    hallucinated -- including the case where no retrieved sentence was
    similar enough to trust), and detail is a short human-readable trace
    of the per-sentence NLI calls (with retrieval similarity shown), for
    debugging.
    """
    evidence_sentences = split_into_windows(evidence_text, config.EVIDENCE_WINDOW_SIZE)
    top = top_n_similar(sentence, evidence_sentences, config.TOP_N)

    all_results = [(evid, sim, *nli_predict(evid, sentence)) for evid, sim in top]
    results = [r for r in all_results if r[1] >= config.MIN_SIMILARITY]

    n_contradictions = sum(
        1
        for _, _, label, probs in results
        if label == "CONTRADICTION" and probs[label] >= config.MIN_CONTRADICTION_CONFIDENCE
    )
    neutral_confidences = [probs[label] for _, _, label, probs in results if label == "NEUTRAL"]
    if not results:
        verdict, reason = 0, "no_sufficiently_similar_evidence"
    elif n_contradictions >= config.MIN_CONTRADICTIONS:
        verdict, reason = 1, "contradiction"
    elif any(label == "ENTAILMENT" for _, _, label, _ in results):
        verdict, reason = 0, "entailment"
    elif neutral_confidences:
        mean_conf = sum(neutral_confidences) / len(neutral_confidences)
        verdict = 1 if mean_conf >= config.NEUTRAL_CONFIDENCE_THRESHOLD else 0
        reason = f"all_neutral(mean_conf={mean_conf:.2f})"
    else:
        # Only sub-threshold contradictions (and no entailment/neutral) --
        # not enough agreement to call it hallucinated.
        verdict, reason = 0, "insufficient_agreement"

    trace = "; ".join(
        f"{label}(sim={sim:.2f},conf={probs[label]:.2f})" for _, sim, label, probs in all_results
    ) or "(no evidence sentences)"
    return verdict, f"{reason} | {trace}"
