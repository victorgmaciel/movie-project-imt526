"""
extract_semantic_features.py

Step 1 of the pipeline: Run Qwen-2.5-7B-Instruct over YouTube comment batches
and extract structured semantic features per batch.

Input:  data/raw/youtube_batches.jsonl
Output: data/processed/semantic_features_raw.jsonl

Each output record contains:
  - movie_id, batch_id, video_id
  - chain_of_thought reasoning trace (string)
  - semantic_features (dict with 10 float scores)
  - is_valid_json (bool)
  - token_counts (dict)

Usage:
    python scripts/extract_semantic_features.py \
        --input  data/raw/youtube_batches.jsonl \
        --output data/processed/semantic_features_raw.jsonl \
        --model  Qwen/Qwen2.5-7B-Instruct \
        --batch_size 8 \
        --max_comments_per_batch 50 \
        --device auto

For testing on the dummy sample:
    python scripts/extract_semantic_features.py \
        --input  data/sample/dummy_youtube_batches.jsonl \
        --output data/processed/semantic_features_raw_dummy.jsonl \
        --sample_limit 20
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/extract_semantic_features.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a film market analyst assistant. You will be given a batch of YouTube comments
posted on a movie trailer before the film's release date. Your job is to analyze the sentiment
and audience demand signals contained in these comments.

You must respond in two clearly separated parts:

PART 1 — REASONING:
Write a short paragraph (3–5 sentences) covering:
- Overall sentiment balance (positive vs. negative vs. mixed)
- Any strong hype signals or excitement markers
- Red flags such as skepticism, sequel fatigue, or weak marketing reception
- Any purchase-intent language (mentions of buying tickets, opening weekend, etc.)

PART 2 — JSON:
Output a single JSON object with exactly these keys and float values from 0.0 to 1.0:
{
  "hype": <float>,
  "skepticism": <float>,
  "purchase_intent": <float>,
  "sequel_fatigue": <float>,
  "nostalgia": <float>,
  "controversy": <float>,
  "cast_interest": <float>,
  "vfx_excitement": <float>,
  "story_confusion": <float>,
  "positive_sentiment": <float>
}

Scoring guide:
- hype: fraction of comments expressing excitement or anticipation
- skepticism: fraction expressing doubt or wait-and-see attitude
- purchase_intent: fraction mentioning buying tickets, opening weekend, or definite plans to watch
- sequel_fatigue: fraction expressing exhaustion with franchise or sequels
- nostalgia: fraction referencing affection for prior installments or source material
- controversy: fraction containing divisive or polarizing language
- cast_interest: fraction expressing interest in or opinions about specific cast members
- vfx_excitement: fraction praising visuals, effects, or cinematography
- story_confusion: fraction expressing confusion about the plot or story
- positive_sentiment: overall fraction with a net positive tone

Do NOT output anything after the JSON object."""

USER_TEMPLATE = """Movie: {title}

Pre-release YouTube comments ({n_comments} comments):
{comments_block}

Analyze these comments and respond as instructed."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# my python is 3.8, if 3.9 and above dont need this import, and Tuple -> tuple
from typing import Tuple, Optional

def build_prompt(record: dict, max_comments: int) -> Tuple[str, int]:
    """Build the (system, user) prompt strings for one batch record."""
    raw_comments = record.get("texts")

    if raw_comments is None:
        raw_comments = record.get("comments", [])

    comments = []
    for c in raw_comments[:max_comments]:
        if isinstance(c, dict):
            comment_text = c.get("text", "")
        else:
            comment_text = str(c)
        if comment_text:
            comments.append(comment_text)
    n = len(comments)
    comments_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))
    user_msg = USER_TEMPLATE.format(
        title=record.get("title", record.get("movie_id", "Unknown")),
        n_comments=n,
        comments_block=comments_block,
    )
    return SYSTEM_PROMPT, user_msg, n

# 3.10+ version
# def parse_llm_response(response_text: str) -> Tuple[str, dict | None, bool]:
def parse_llm_response(response_text: str) -> Tuple[str, Optional[dict], bool]:
    """
    Split the model response into (reasoning, features_dict, is_valid).
    Expects PART 1 reasoning followed by a JSON block in PART 2.
    """
    reasoning = ""
    features = None
    is_valid = False

    # Try to find the JSON object in the response
    brace_start = response_text.find("{")
    brace_end = response_text.rfind("}") + 1
    reasoning_raw = response_text[:brace_start].strip() if brace_start > 0 else response_text.strip()

    # Strip "PART 1 — REASONING:" / "PART 2 — JSON:" labels if present
    for marker in ["PART 1 — REASONING:", "PART 1 - REASONING:", "REASONING:"]:
        reasoning_raw = reasoning_raw.replace(marker, "").strip()
    for marker in ["PART 2 — JSON:", "PART 2 - JSON:", "JSON:"]:
        reasoning_raw = reasoning_raw.replace(marker, "").strip()
    reasoning = reasoning_raw

    if brace_start >= 0 and brace_end > brace_start:
        json_str = response_text[brace_start:brace_end]
        try:
            parsed = json.loads(json_str)
            # Validate expected keys and coerce to float in [0, 1]
            expected_keys = {
                "hype", "skepticism", "purchase_intent", "sequel_fatigue",
                "nostalgia", "controversy", "cast_interest", "vfx_excitement",
                "story_confusion", "positive_sentiment",
            }
            if expected_keys.issubset(parsed.keys()):
                features = {k: float(min(max(parsed[k], 0.0), 1.0)) for k in expected_keys}
                is_valid = True
            else:
                missing = expected_keys - parsed.keys()
                log.warning("Missing keys in JSON output: %s", missing)
                # Fill missing keys with 0.0 so the record is still usable
                features = {k: float(min(max(parsed.get(k, 0.0), 0.0), 1.0)) for k in expected_keys}
        except json.JSONDecodeError as e:
            log.warning("JSON parse error: %s | raw: %s", e, json_str[:200])

    return reasoning, features, is_valid


def load_model_and_tokenizer(model_name: str, device: str):
    log.info("Loading tokenizer from %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading model from %s (device=%s)", model_name, device)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    log.info("Model loaded. Peak GPU memory: %.2f GB",
             torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
    return model, tokenizer


def run_inference_single(
    model,
    tokenizer,
    system_prompt: str,
    user_msg: str,
    max_new_tokens: int = 512,
) -> Tuple[str, dict]:
    """Run inference for a single prompt. Returns (response_text, token_counts)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    # Qwen chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for reproducibility
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_ids.shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    token_counts = {
        "input_tokens": int(input_ids.shape[1]),
        "output_tokens": int(new_tokens.shape[0]),
    }
    return response_text, token_counts


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main(args):
    Path("logs").mkdir(exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model, args.device)

    # Stream through input JSONL
    input_path = Path(args.input)
    output_path = Path(args.output)

    stats = {"total": 0, "valid_json": 0, "invalid_json": 0, "errors": 0}
    t_start = time.time()

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line_idx, line in enumerate(fin):
            if args.sample_limit and line_idx >= args.sample_limit:
                log.info("Reached sample_limit=%d, stopping.", args.sample_limit)
                break

            line = line.strip()
            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping malformed input line %d", line_idx)
                stats["errors"] += 1
                continue

            movie_id = record.get("movie_id", f"unk_{line_idx}")
            batch_id = record.get("batch_id", f"batch_{line_idx}")
            video_id = record.get("video_id", "")

            log.info("[%d] Processing movie_id=%s batch_id=%s", line_idx, movie_id, batch_id)

            try:
                system_prompt, user_msg, n_comments = build_prompt(record, args.max_comments_per_batch)
                t0 = time.time()
                response_text, token_counts = run_inference_single(
                    model, tokenizer, system_prompt, user_msg, args.max_new_tokens
                )
                elapsed = time.time() - t0

                reasoning, features, is_valid = parse_llm_response(response_text)

                output_record = {
                    "movie_id": movie_id,
                    "batch_id": batch_id,
                    "video_id": video_id,
                    "title": record.get("title", ""),
                    "n_comments_processed": n_comments,
                    "chain_of_thought": reasoning,
                    "semantic_features": features if features else {},
                    "is_valid_json": is_valid,
                    "token_counts": token_counts,
                    "inference_seconds": round(elapsed, 3),
                }
                fout.write(json.dumps(output_record) + "\n")
                fout.flush()

                stats["total"] += 1
                if is_valid:
                    stats["valid_json"] += 1
                else:
                    stats["invalid_json"] += 1

                if (line_idx + 1) % 50 == 0:
                    elapsed_total = time.time() - t_start
                    rate = stats["total"] / elapsed_total
                    log.info("Progress: %d batches | %.2f batches/sec | valid JSON: %d/%d",
                             stats["total"], rate, stats["valid_json"], stats["total"])

            except Exception as e:
                log.error("Error on line %d (movie_id=%s): %s", line_idx, movie_id, e, exc_info=True)
                stats["errors"] += 1
                # Write error placeholder so we can audit later
                error_record = {
                    "movie_id": movie_id,
                    "batch_id": batch_id,
                    "video_id": video_id,
                    "title": record.get("title", ""),
                    "n_comments_processed": 0,
                    "chain_of_thought": "",
                    "semantic_features": {},
                    "is_valid_json": False,
                    "error": str(e),
                    "token_counts": {},
                    "inference_seconds": 0.0,
                }
                fout.write(json.dumps(error_record) + "\n")
                fout.flush()

    total_elapsed = time.time() - t_start
    log.info("=" * 60)
    log.info("DONE. Total batches processed: %d", stats["total"])
    log.info("Valid JSON outputs: %d (%.1f%%)",
             stats["valid_json"], 100 * stats["valid_json"] / max(stats["total"], 1))
    log.info("Invalid JSON: %d | Errors: %d", stats["invalid_json"], stats["errors"])
    log.info("Total runtime: %.2f seconds (%.4f sec/batch)", total_elapsed,
             total_elapsed / max(stats["total"], 1))
    if torch.cuda.is_available():
        log.info("Peak GPU memory: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)
    log.info("Output written to: %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract semantic features from YouTube comment batches.")
    parser.add_argument("--input",  default="data/raw/youtube_batches.jsonl",
                        help="Path to input JSONL file of comment batches")
    parser.add_argument("--output", default="data/processed/semantic_features_raw.jsonl",
                        help="Path to output JSONL file of semantic features")
    parser.add_argument("--model",  default="Qwen/Qwen2.5-7B-Instruct",
                        help="HuggingFace model name or local path")
    parser.add_argument("--device", default="auto",
                        help="Device map: 'auto', 'cuda', 'cpu'")
    parser.add_argument("--max_comments_per_batch", type=int, default=50,
                        help="Max comments to include per batch prompt (truncates longer batches)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate per batch")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Only process this many batches (for testing)")
    args = parser.parse_args()
    main(args)