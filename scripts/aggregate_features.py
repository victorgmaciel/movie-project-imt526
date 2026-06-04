"""
aggregate_features.py

Step 2 of the pipeline: Aggregate per-batch semantic features into one
movie-level record. Also computes engagement features (comment volume,
batch count) from the raw YouTube JSONL.

Input:
  data/processed/semantic_features_raw.jsonl   (from extract_semantic_features.py)
  data/raw/youtube_batches.jsonl               (for engagement counts)

Output:
  data/processed/movie_semantic_features.csv

Each row = one movie, with columns:
  movie_id, title,
  comment_count,         <- total pre-release comments
  batch_count,           <- number of batches
  valid_batch_count,     <- batches with valid LLM JSON
  hype_mean, hype_std,
  skepticism_mean, ...   (mean + std for each of the 10 semantic features)

Usage:
    python scripts/aggregate_features.py \
        --semantic  data/processed/semantic_features_raw.jsonl \
        --batches   data/raw/youtube_batches.jsonl \
        --output    data/processed/movie_semantic_features.csv
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/aggregate_features.log"),
    ],
)
log = logging.getLogger(__name__)

SEMANTIC_KEYS = [
    "hype", "skepticism", "purchase_intent", "sequel_fatigue",
    "nostalgia", "controversy", "cast_interest", "vfx_excitement",
    "story_confusion", "positive_sentiment",
]


def load_engagement_counts(batches_path: str) -> dict[str, dict]:
    """
    Read the raw JSONL to count total comments and batches per movie.
    Returns dict: movie_id -> {comment_count, batch_count, title}
    """
    counts: dict[str, dict] = defaultdict(lambda: {"comment_count": 0, "batch_count": 0, "title": ""})
    with open(batches_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                mid = rec.get("movie_id", "")
                counts[mid]["comment_count"] += len(rec.get("texts", []))
                counts[mid]["batch_count"] += 1
                if not counts[mid]["title"]:
                    counts[mid]["title"] = rec.get("title", "")
            except json.JSONDecodeError:
                continue
    return dict(counts)


def load_semantic_records(semantic_path: str) -> dict[str, list[dict]]:
    """
    Read the LLM output JSONL.
    Returns dict: movie_id -> list of feature dicts (valid only)
    """
    records: dict[str, list[dict]] = defaultdict(list)
    total, valid = 0, 0
    with open(semantic_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                total += 1
                if rec.get("is_valid_json") and rec.get("semantic_features"):
                    records[rec["movie_id"]].append(rec["semantic_features"])
                    valid += 1
            except json.JSONDecodeError:
                continue
    log.info("Loaded %d semantic records; %d with valid JSON (%.1f%%)",
             total, valid, 100 * valid / max(total, 1))
    return dict(records)


def aggregate(args):
    Path("logs").mkdir(exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading engagement counts from %s", args.batches)
    engagement = load_engagement_counts(args.batches)

    log.info("Loading semantic features from %s", args.semantic)
    semantic_by_movie = load_semantic_records(args.semantic)

    all_movie_ids = set(engagement.keys()) | set(semantic_by_movie.keys())
    log.info("Total unique movie IDs: %d", len(all_movie_ids))

    rows = []
    for mid in sorted(all_movie_ids):
        eng = engagement.get(mid, {"comment_count": 0, "batch_count": 0, "title": ""})
        feat_list = semantic_by_movie.get(mid, [])

        row = {
            "movie_id": mid,
            "title": eng["title"],
            "comment_count": eng["comment_count"],
            "batch_count": eng["batch_count"],
            "valid_batch_count": len(feat_list),
        }

        if feat_list:
            # Stack into (n_batches, n_features) array and compute stats
            arr = np.array([[f[k] for k in SEMANTIC_KEYS] for f in feat_list])
            means = arr.mean(axis=0)
            stds  = arr.std(axis=0) if len(feat_list) > 1 else np.zeros(len(SEMANTIC_KEYS))
            for i, k in enumerate(SEMANTIC_KEYS):
                row[f"{k}_mean"] = round(float(means[i]), 4)
                row[f"{k}_std"]  = round(float(stds[i]),  4)
        else:
            # No valid LLM output — fill with NaN so we can handle in downstream
            for k in SEMANTIC_KEYS:
                row[f"{k}_mean"] = float("nan")
                row[f"{k}_std"]  = float("nan")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Coverage report
    n_with_semantic = df["valid_batch_count"].gt(0).sum()
    log.info("Movies with ≥1 valid semantic batch: %d / %d (%.1f%%)",
             n_with_semantic, len(df), 100 * n_with_semantic / max(len(df), 1))

    df.to_csv(args.output, index=False)
    log.info("Saved movie-level semantic features to %s (%d rows, %d cols)",
             args.output, len(df), len(df.columns))
    log.info("Columns: %s", list(df.columns))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", default="data/processed/semantic_features_raw.jsonl")
    parser.add_argument("--batches",  default="data/raw/youtube_batches.jsonl")
    parser.add_argument("--output",   default="data/processed/movie_semantic_features.csv")
    args = parser.parse_args()
    aggregate(args)