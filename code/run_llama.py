"""Runs Llama-3.2-1B-Instruct (mlx-community/Llama-3.2-1B-Instruct-4bit) as a
small-LLM hallucination-detection baseline on the held-out test set, using
the same batched-per-paper prompt as the GPT-4-turbo/GPT-4o baselines.

This is the SAME model family used to generate the benchmark's summaries
(Qwen2.5) is NOT this model -- Llama-3.2 is run here as a *same-family-style*
control alongside the cross-family Gemma-2-2b detector (run_gemma.py), to
test whether sharing a model family with the generator biases detection.
Note: this is an MLX-native download, not the checkpoint used for any
generation step -- different toolchain, same published weights.

Llama-3.2-1B has a 131k-token context window (via RoPE scaling), so no
aggressive truncation is needed -- uses the same generous cap as the GPT
baselines (config.MAX_FULL_PAPER_CHARS).

Runs entirely locally via MLX (Apple Silicon), no API key needed.

Usage:
    python run_llama.py
"""
import argparse
import csv
import json
from pathlib import Path

import config
from dataset_utils import filter_to_papers, load_dataset_rows, print_coverage_report, resolve_matched_rows
from evidence import get_full_paper_text
from manifest import build_manifest, normalize_title
from openai_client import BATCH_USER_TEMPLATE, _parse_batch_response

MODEL_NAME = "mlx-community/Llama-3.2-1B-Instruct-4bit"
FIELDNAMES = ["Paper Name", "Aspect", "Sentence", "Gold_Hallucination", "Predicted_Hallucination", "Raw_Model_Response"]

# openai_client.BATCH_SYSTEM_PROMPT's literal "'N: Hallucinated'" format
# example was echoed verbatim by small local models (Gemma-2-2b, this model)
# instead of being filled in with real sentence numbers. Adds a concrete
# worked example with real numbers so the model has something to
# pattern-match against.
LOCAL_BATCH_SYSTEM_PROMPT = (
    "You are a careful fact-checker for scientific paper summaries. Given "
    "the full text of a scientific paper and a numbered list of summary "
    "sentences about it, classify EACH sentence as either 'Hallucinated' "
    "or 'Not Hallucinated'. Output exactly one line per sentence, in the "
    "format '<number>: Hallucinated' or '<number>: Not Hallucinated', "
    "replacing <number> with the actual sentence number (1, 2, 3, ...), "
    "in order, with no other text, explanation, or blank lines.\n\n"
    "Example: for 2 input sentences, a correct response looks EXACTLY like "
    "this (with real sentence numbers, not the literal word 'number'):\n"
    "1: Not Hallucinated\n"
    "2: Hallucinated"
)


def group_by_paper(matched_rows):
    by_paper = {}
    for row in matched_rows:
        by_paper.setdefault(row["Paper Name"], []).append(row)
    return by_paper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", default="0", choices=["0", "0.8"], help="Which decoding setting's rows to evaluate (0 = greedy, 0.8 = sampled).")
    parser.add_argument("--all-papers", action="store_true", help="Evaluate on all 50 papers instead of the 25-paper held-out test set.")
    args = parser.parse_args()

    from mlx_lm import generate, load

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / "predictions_llama1b.csv"

    manifest = build_manifest(config.PAPERS_DIR)
    rows = load_dataset_rows(config.DATASET_CSV, temperature=args.temperature)
    matched_rows, skipped_papers = resolve_matched_rows(rows, manifest)
    print_coverage_report(rows, matched_rows, skipped_papers)
    if not args.all_papers:
        matched_rows = filter_to_papers(matched_rows, config.test_papers_for(args.temperature))
        print(f"Restricted to the {len(config.test_papers_for(args.temperature))}-paper held-out test set: {len(matched_rows)} rows.")

    print(f"Loading {MODEL_NAME}...")
    model, tokenizer = load(MODEL_NAME)

    paper_json_cache = {}
    out_rows = []
    by_paper = group_by_paper(matched_rows)
    for i, (paper_name, paper_rows) in enumerate(by_paper.items(), 1):
        _title, json_path = manifest[normalize_title(paper_name)]
        if json_path not in paper_json_cache:
            paper_json_cache[json_path] = json.loads(Path(json_path).read_text())
        paper_text, _truncated = get_full_paper_text(paper_json_cache[json_path])

        sentences = [r["Sentence"] for r in paper_rows]
        n = len(sentences)
        numbered = "\n".join(f"{j + 1}. {s}" for j, s in enumerate(sentences))
        prompt = BATCH_USER_TEMPLATE.format(paper_text=paper_text, numbered_sentences=numbered, n=n)
        messages = [{"role": "user", "content": LOCAL_BATCH_SYSTEM_PROMPT + "\n\n" + prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        raw = generate(model, tokenizer, prompt=formatted_prompt, max_tokens=max(80, n * 25), verbose=False)
        results = _parse_batch_response(raw.strip(), n)

        for row, (verdict, raw_line) in zip(paper_rows, results):
            out_rows.append({
                "Paper Name": row["Paper Name"],
                "Aspect": row["Aspect"],
                "Sentence": row["Sentence"],
                "Gold_Hallucination": row["Hallucination"],
                "Predicted_Hallucination": "" if verdict is None else verdict,
                "Raw_Model_Response": raw_line,
            })
        print(f"[{i}/{len(by_paper)}] {paper_name[:50]} ({n} sentences) processed")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)
    n_halluc = sum(1 for r in out_rows if r["Predicted_Hallucination"] == 1)
    print(f"Saved {len(out_rows)} rows ({n_halluc} predicted hallucinated) to {out_path}")


if __name__ == "__main__":
    main()
