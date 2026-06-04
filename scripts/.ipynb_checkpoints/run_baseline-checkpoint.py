"""
run_baseline.py

Step 4 of the pipeline: Train and evaluate engagement-only and metadata-only
regression baselines on the training set, then score on the validation set.

This establishes the bar that the semantic model must beat to justify
the added LLM complexity (per the project's signal-validation approach).

Baselines:
  B0 - Intercept only (predict mean)
  B1 - Engagement only: comment_count, batch_count (log-transformed)
  B2 - Metadata only: popularity, vote_average, vote_count, log_budget,
                      release_month, is_sequel_proxy, genre dummies
  B3 - Metadata + Engagement (combined baseline)

Metrics:
  R² (coefficient of determination)
  RMSE on log_revenue
  Pearson r between predicted and actual log_revenue

Output:
  outputs/baseline_results.json
  outputs/baseline_results.csv

Usage:
    python scripts/run_baseline.py \
        --train data/processed/train.csv \
        --val   data/processed/val.csv \
        --output outputs/baseline_results.json
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/run_baseline.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
ENGAGEMENT_FEATURES = ["log_comment_count", "log_batch_count"]

METADATA_FEATURES = [
    "popularity", "vote_average", "vote_count", "log_budget",
    "release_month", "is_sequel_proxy",
]

GENRE_FEATURES = [
    "genre_action", "genre_adventure", "genre_animation", "genre_comedy",
    "genre_crime", "genre_drama", "genre_family", "genre_fantasy",
    "genre_horror", "genre_romance", "genre_scifi", "genre_thriller",
]

OUTCOME = "log_revenue"


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and handle missing values."""
    df = df.copy()

    # Log-transform engagement counts (add 1 to handle zeros)
    df["log_comment_count"] = np.log1p(df["comment_count"].fillna(0))
    df["log_batch_count"]   = np.log1p(df["batch_count"].fillna(0))

    # Fill metadata NaNs with column median
    fill_cols = METADATA_FEATURES + GENRE_FEATURES
    for col in fill_cols:
        if col in df.columns:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    return df


def get_feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Extract feature matrix, filling any remaining NaNs with 0."""
    available = [c for c in feature_cols if c in df.columns]
    missing   = [c for c in feature_cols if c not in df.columns]
    if missing:
        log.warning("Missing feature columns (will be skipped): %s", missing)
    X = df[available].fillna(0).values.astype(float)
    return X, available


def evaluate_model(model, X_val: np.ndarray, y_val: np.ndarray, label: str) -> dict:
    """Evaluate a fitted model and return metrics dict."""
    preds = model.predict(X_val)
    r2    = r2_score(y_val, preds)
    rmse  = np.sqrt(mean_squared_error(y_val, preds))
    pearson_r, pearson_p = stats.pearsonr(preds, y_val)
    metrics = {
        "model": label,
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 6),
        "n_features": int(X_val.shape[1]),
        "n_val": int(len(y_val)),
    }
    log.info("%-40s R²=%.4f  RMSE=%.4f  r=%.4f  p=%.4f",
             label, r2, rmse, pearson_r, pearson_p)
    return metrics


def run_baseline(args):
    Path("logs").mkdir(exist_ok=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    log.info("Loading train from %s", args.train)
    train = pd.read_csv(args.train)
    log.info("Loading val from %s", args.val)
    val   = pd.read_csv(args.val)

    train = prepare_df(train)
    val   = prepare_df(val)

    # Drop rows with missing outcome
    train_ok = train[train[OUTCOME].notna() & (train["revenue"].fillna(0) > 0)].copy()
    val_ok   = val[val[OUTCOME].notna()     & (val["revenue"].fillna(0) > 0)].copy()
    log.info("Train rows with valid revenue: %d / %d", len(train_ok), len(train))
    log.info("Val rows with valid revenue:   %d / %d", len(val_ok),   len(val))

    y_train = train_ok[OUTCOME].values
    y_val   = val_ok[OUTCOME].values

    results = []

    # ------------------------------------------------------------------
    # B0: Intercept only (predict global mean)
    # ------------------------------------------------------------------
    mean_pred = np.full_like(y_val, y_train.mean())
    r2_b0     = r2_score(y_val, mean_pred)
    rmse_b0   = np.sqrt(mean_squared_error(y_val, mean_pred))
    b0 = {
        "model": "B0_intercept_only",
        "r2": round(float(r2_b0), 4),
        "rmse": round(float(rmse_b0), 4),
        "pearson_r": 0.0,
        "pearson_p": 1.0,
        "n_features": 0,
        "n_val": len(y_val),
    }
    log.info("%-40s R²=%.4f  RMSE=%.4f", "B0_intercept_only", r2_b0, rmse_b0)
    results.append(b0)

    # ------------------------------------------------------------------
    # B1: Engagement only
    # ------------------------------------------------------------------
    eng_cols = [c for c in ENGAGEMENT_FEATURES if c in train_ok.columns]
    if eng_cols:
        X_train_eng, used = get_feature_matrix(train_ok, ENGAGEMENT_FEATURES)
        X_val_eng,   _    = get_feature_matrix(val_ok,   ENGAGEMENT_FEATURES)
        pipe_b1 = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
        pipe_b1.fit(X_train_eng, y_train)
        results.append(evaluate_model(pipe_b1, X_val_eng, y_val, "B1_engagement_only"))
    else:
        log.warning("No engagement features available, skipping B1")

    # ------------------------------------------------------------------
    # B2: Metadata only
    # ------------------------------------------------------------------
    meta_genre_cols = METADATA_FEATURES + GENRE_FEATURES
    X_train_meta, used = get_feature_matrix(train_ok, meta_genre_cols)
    X_val_meta,   _    = get_feature_matrix(val_ok,   meta_genre_cols)
    pipe_b2 = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe_b2.fit(X_train_meta, y_train)
    results.append(evaluate_model(pipe_b2, X_val_meta, y_val, "B2_metadata_only"))

    # ------------------------------------------------------------------
    # B3: Metadata + Engagement
    # ------------------------------------------------------------------
    all_baseline_cols = METADATA_FEATURES + GENRE_FEATURES + ENGAGEMENT_FEATURES
    X_train_all, used = get_feature_matrix(train_ok, all_baseline_cols)
    X_val_all,   _    = get_feature_matrix(val_ok,   all_baseline_cols)
    pipe_b3 = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipe_b3.fit(X_train_all, y_train)
    results.append(evaluate_model(pipe_b3, X_val_all, y_val, "B3_metadata_plus_engagement"))

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_json = args.output
    output_csv  = str(Path(args.output).with_suffix(".csv"))

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Baseline results saved to %s", output_json)

    pd.DataFrame(results).to_csv(output_csv, index=False)
    log.info("Baseline results saved to %s", output_csv)

    log.info("=" * 60)
    log.info("BASELINE SUMMARY (validation set)")
    log.info("%-40s %8s %8s %8s", "Model", "R²", "RMSE", "Pearson-r")
    for r in results:
        log.info("%-40s %8.4f %8.4f %8.4f", r["model"], r["r2"], r["rmse"], r["pearson_r"])
    log.info("=" * 60)
    log.info("The semantic model must beat B3 (metadata + engagement) on R² and RMSE.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",  default="data/processed/train.csv")
    parser.add_argument("--val",    default="data/processed/val.csv")
    parser.add_argument("--output", default="outputs/baseline_results.json")
    args = parser.parse_args()
    run_baseline(args)