"""
build_movie_table.py

Step 3 of the pipeline: Join TMDB metadata with the movie-level semantic
feature table into a single analysis-ready CSV with one row per movie.

Input:
  data/raw/tmdb_metadata.csv
  data/processed/movie_semantic_features.csv

Output:
  data/processed/movie_table.csv          <- full joined table
  data/processed/train.csv
  data/processed/val.csv
  data/processed/test.csv

Split strategy: deterministic by movie title hash (70 / 15 / 15).
The test set is locked before any downstream analysis.

TMDB genre_ids are one-hot encoded for the most common genres.
Log-transformed revenue is added as log_revenue.

Usage:
    python scripts/build_movie_table.py \
        --tmdb      data/raw/tmdb_metadata.csv \
        --semantic  data/processed/movie_semantic_features.csv \
        --output    data/processed/movie_table.csv
"""

import argparse
import ast
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/build_movie_table.log"),
    ],
)
log = logging.getLogger(__name__)

# TMDB genre_id -> human-readable name (most common genres)
GENRE_MAP = {
    28:    "genre_action",
    12:    "genre_adventure",
    16:    "genre_animation",
    35:    "genre_comedy",
    80:    "genre_crime",
    99:    "genre_documentary",
    18:    "genre_drama",
    10751: "genre_family",
    14:    "genre_fantasy",
    36:    "genre_history",
    27:    "genre_horror",
    10402: "genre_music",
    9648:  "genre_mystery",
    10749: "genre_romance",
    878:   "genre_scifi",
    10770: "genre_tv_movie",
    53:    "genre_thriller",
    10752: "genre_war",
    37:    "genre_western",
}


def parse_genre_ids(raw: str) -> list[int]:
    """Parse genre_ids from either '[28, 14]' or JSON string."""
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        return ast.literal_eval(raw)
    except Exception:
        try:
            return json.loads(raw)
        except Exception:
            return []


def genre_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary genre columns to dataframe."""
    for gid, col in GENRE_MAP.items():
        df[col] = df["genre_ids_parsed"].apply(lambda ids: int(gid in ids))
    return df


def title_to_split(title: str, seed: int = 42) -> str:
    """
    Assign a movie to train/val/test based on a deterministic hash of its title.
    This ensures the same movie always lands in the same split regardless of order.
    """
    h = int(hashlib.md5(f"{seed}:{title}".encode()).hexdigest(), 16) % 100
    if h < 70:
        return "train"
    elif h < 85:
        return "val"
    else:
        return "test"


def build_table(args):
    Path("logs").mkdir(exist_ok=True)
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load TMDB metadata
    # ------------------------------------------------------------------
    log.info("Loading TMDB metadata from %s", args.tmdb)
    tmdb = pd.read_csv(args.tmdb)
    log.info("TMDB shape: %s", tmdb.shape)

    # Parse genre_ids
    tmdb["genre_ids_parsed"] = tmdb["genre_ids"].apply(parse_genre_ids)
    tmdb = genre_one_hot(tmdb)

    # Log-transform revenue (if present)
    if "revenue" in tmdb.columns:
        tmdb["log_revenue"] = np.log1p(tmdb["revenue"].fillna(0).clip(lower=0))
    else:
        log.warning("'revenue' column not found in TMDB data — skipping log_revenue")
        tmdb["log_revenue"] = np.nan

    # Budget log-transform (if present)
    if "budget" in tmdb.columns:
        tmdb["log_budget"] = np.log1p(tmdb["budget"].fillna(0).clip(lower=0))
    else:
        tmdb["log_budget"] = np.nan

    # Release year/month for seasonality
    tmdb["release_date"] = pd.to_datetime(tmdb["release_date"], errors="coerce")
    tmdb["release_year"]  = tmdb["release_date"].dt.year
    tmdb["release_month"] = tmdb["release_date"].dt.month

    # Franchise proxy: any movie with a colon or number in title is *possibly* a sequel
    # (rough heuristic, not definitive)
    tmdb["is_sequel_proxy"] = (
        tmdb["title"].str.contains(r":\s|\b[2-9]\b|II|III|IV|VI|VII|VIII|IX", regex=True, na=False)
    ).astype(int)

    # ------------------------------------------------------------------
    # 2. Load semantic features
    # ------------------------------------------------------------------
    log.info("Loading semantic features from %s", args.semantic)
    sem = pd.read_csv(args.semantic)
    log.info("Semantic features shape: %s", sem.shape)

    # ------------------------------------------------------------------
    # 3. Match by title (fuzzy-tolerant: strip, lower, drop punctuation)
    # ------------------------------------------------------------------
    def normalize_title(t):
        import re
        return re.sub(r"[^a-z0-9 ]", "", str(t).lower().strip())

    tmdb["_title_norm"] = tmdb["title"].apply(normalize_title)
    sem["_title_norm"]  = sem["title"].apply(normalize_title)

    merged = tmdb.merge(sem, on="_title_norm", how="left", suffixes=("", "_sem"))
    matched = merged["movie_id"].notna().sum()
    log.info("TMDB rows: %d | matched with semantic features: %d (%.1f%%)",
             len(tmdb), matched, 100 * matched / max(len(tmdb), 1))

    # ------------------------------------------------------------------
    # 4. Assign train / val / test splits
    # ------------------------------------------------------------------
    merged["split"] = merged["title"].apply(title_to_split)
    split_counts = merged["split"].value_counts()
    log.info("Split distribution:\n%s", split_counts.to_string())

    # ------------------------------------------------------------------
    # 5. Select and order final columns
    # ------------------------------------------------------------------
    id_cols   = ["tmdb_id", "title", "split"]
    meta_cols = [
        "release_year", "release_month", "runtime", "popularity",
        "vote_average", "vote_count", "log_budget", "is_sequel_proxy",
        "original_language",
    ]
    genre_cols   = [c for c in merged.columns if c.startswith("genre_")]
    outcome_cols = ["revenue", "log_revenue"]
    eng_cols     = ["comment_count", "batch_count", "valid_batch_count"]
    sem_cols     = [c for c in merged.columns if c.endswith("_mean") or c.endswith("_std")]

    available = set(merged.columns)
    def keep(cols):
        return [c for c in cols if c in available]

    final_cols = keep(id_cols + meta_cols + genre_cols + outcome_cols + eng_cols + sem_cols)
    out_df = merged[final_cols].copy()

    log.info("Final table shape: %s", out_df.shape)
    log.info("Columns: %s", list(out_df.columns))

    # ------------------------------------------------------------------
    # 6. Save outputs
    # ------------------------------------------------------------------
    out_df.to_csv(args.output, index=False)
    log.info("Full table saved to %s", args.output)

    for split_name in ["train", "val", "test"]:
        split_df = out_df[out_df["split"] == split_name]
        split_path = out_dir / f"{split_name}.csv"
        split_df.to_csv(split_path, index=False)
        log.info("%s set: %d rows -> %s", split_name, len(split_df), split_path)

    # ------------------------------------------------------------------
    # 7. Quick data quality report
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info("DATA QUALITY REPORT")
    log.info("Missing value rates for key columns:")
    check_cols = keep(["log_revenue", "comment_count"] + sem_cols[:4])
    for col in check_cols:
        pct_missing = out_df[col].isna().mean() * 100
        log.info("  %s: %.1f%% missing", col, pct_missing)

    # Movies with revenue > 0
    if "revenue" in out_df.columns:
        n_with_revenue = (out_df["revenue"].fillna(0) > 0).sum()
        log.info("Movies with revenue > 0: %d / %d", n_with_revenue, len(out_df))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tmdb",     default="data/raw/tmdb_metadata.csv")
    parser.add_argument("--semantic", default="data/processed/movie_semantic_features.csv")
    parser.add_argument("--output",   default="data/processed/movie_table.csv")
    args = parser.parse_args()
    build_table(args)