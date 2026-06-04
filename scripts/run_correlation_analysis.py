"""
run_correlation_analysis.py

Step 5 (final): Full correlation and regression analysis testing whether
LLM-extracted semantic features from YouTube comments carry incremental
explanatory power over engagement and metadata baselines.

This is the main hypothesis-testing script. It runs:
  1. Pearson and Spearman correlations of each semantic feature with log_revenue
  2. Incremental R² when semantic features are added to the best baseline (B3)
  3. A cross-cut validation by genre, is_sequel_proxy, and budget tier
  4. Feature importance summary via Ridge coefficients
  5. A final signal-verdict table comparing all models

All analysis runs on the held-out TEST set after being locked (no tuning on test).
During development, analysis is done on the validation set only.

Output:
  outputs/correlation_results.json
  outputs/correlation_results.csv
  outputs/feature_importance.csv
  outputs/crosscut_results.csv
  outputs/analysis_report.txt        <- human-readable summary

Usage (dev / validation phase):
    python scripts/run_correlation_analysis.py \
        --train data/processed/train.csv \
        --eval  data/processed/val.csv \
        --output_dir outputs/ \
        --split_label val

Usage (final test — run ONCE when done):
    python scripts/run_correlation_analysis.py \
        --train data/processed/train.csv \
        --eval  data/processed/test.csv \
        --output_dir outputs/ \
        --split_label test
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
        logging.FileHandler("logs/run_correlation_analysis.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature groups (must match run_baseline.py)
# ---------------------------------------------------------------------------
SEMANTIC_FEATURES = [
    "hype_mean", "skepticism_mean", "purchase_intent_mean", "sequel_fatigue_mean",
    "nostalgia_mean", "controversy_mean", "cast_interest_mean", "vfx_excitement_mean",
    "story_confusion_mean", "positive_sentiment_mean",
]

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

ALL_BASELINE_COLS = METADATA_FEATURES + GENRE_FEATURES + ENGAGEMENT_FEATURES
ALL_FEATURE_COLS  = ALL_BASELINE_COLS + SEMANTIC_FEATURES

OUTCOME = "log_revenue"


def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_comment_count"] = np.log1p(df["comment_count"].fillna(0))
    df["log_batch_count"]   = np.log1p(df["batch_count"].fillna(0))
    for col in METADATA_FEATURES + GENRE_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    return df


def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows with valid revenue and at least one semantic feature."""
    has_revenue  = df[OUTCOME].notna() & (df["revenue"].fillna(0) > 0)
    sem_cols_present = [c for c in SEMANTIC_FEATURES if c in df.columns]
    has_semantic = df[sem_cols_present].notna().any(axis=1) if sem_cols_present else pd.Series(True, index=df.index)
    return df[has_revenue & has_semantic].copy()


def get_X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    avail = [c for c in cols if c in df.columns]
    return df[avail].fillna(0).values.astype(float), avail


def fit_ridge(X_train, y_train, alpha=1.0):
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pipe.fit(X_train, y_train)
    return pipe


def score_model(model, X, y, label=""):
    preds = model.predict(X)
    r2    = r2_score(y, preds)
    rmse  = np.sqrt(mean_squared_error(y, preds))
    r, p  = stats.pearsonr(preds, y)
    return {"model": label, "r2": round(float(r2), 4), "rmse": round(float(rmse), 4),
            "pearson_r": round(float(r), 4), "pearson_p": round(float(p), 6),
            "n": len(y)}


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_analysis(args):
    Path("logs").mkdir(exist_ok=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading train from %s", args.train)
    train_raw = pd.read_csv(args.train)
    log.info("Loading eval (%s) from %s", args.split_label, args.eval)
    eval_raw  = pd.read_csv(args.eval)

    train = prepare_df(train_raw)
    evl   = prepare_df(eval_raw)

    train_ok = filter_valid(train)
    eval_ok  = filter_valid(evl)

    log.info("Train (valid): %d / %d rows", len(train_ok), len(train))
    log.info("Eval  (valid): %d / %d rows", len(eval_ok),  len(evl))

    y_train = train_ok[OUTCOME].values
    y_eval  = eval_ok[OUTCOME].values

    # -----------------------------------------------------------------------
    # 1. Univariate correlations of semantic features with log_revenue
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 1: Univariate correlations (on train set)")

    corr_rows = []
    sem_cols_avail = [c for c in SEMANTIC_FEATURES if c in train_ok.columns]
    for col in sem_cols_avail:
        valid_mask = train_ok[col].notna() & train_ok[OUTCOME].notna()
        x = train_ok.loc[valid_mask, col].values
        y = train_ok.loc[valid_mask, OUTCOME].values
        if len(x) < 10:
            log.warning("Too few valid rows for %s (%d), skipping", col, len(x))
            continue
        pr, pp = stats.pearsonr(x, y)
        sr, sp = stats.spearmanr(x, y)
        corr_rows.append({
            "feature": col,
            "pearson_r": round(float(pr), 4),
            "pearson_p": round(float(pp), 6),
            "spearman_r": round(float(sr), 4),
            "spearman_p": round(float(sp), 6),
            "n": int(valid_mask.sum()),
            "significant_05": int(pp < 0.05),
            "significant_10": int(pp < 0.10),
        })
        log.info("  %-30s Pearson r=%.4f (p=%.4f)  Spearman r=%.4f (p=%.4f)",
                 col, pr, pp, sr, sp)

    corr_df = pd.DataFrame(corr_rows).sort_values("pearson_r", key=abs, ascending=False)
    corr_df.to_csv(out_dir / "correlation_results.csv", index=False)
    log.info("Correlation table saved.")

    # -----------------------------------------------------------------------
    # 2. Regression model comparison
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 2: Model comparison on %s set", args.split_label)

    model_results = []

    # B3: baseline (metadata + engagement)
    X_tr_base, _ = get_X(train_ok, ALL_BASELINE_COLS)
    X_ev_base, _ = get_X(eval_ok,  ALL_BASELINE_COLS)
    m_b3 = fit_ridge(X_tr_base, y_train)
    model_results.append(score_model(m_b3, X_ev_base, y_eval, "B3_metadata_plus_engagement"))

    # S1: Semantic only
    sem_avail = [c for c in SEMANTIC_FEATURES if c in train_ok.columns]
    if sem_avail:
        X_tr_sem, _ = get_X(train_ok, sem_avail)
        X_ev_sem, _ = get_X(eval_ok,  sem_avail)
        m_s1 = fit_ridge(X_tr_sem, y_train)
        model_results.append(score_model(m_s1, X_ev_sem, y_eval, "S1_semantic_only"))

    # S2: Full model (baseline + semantic)
    X_tr_full, full_cols = get_X(train_ok, ALL_FEATURE_COLS)
    X_ev_full, _         = get_X(eval_ok,  ALL_FEATURE_COLS)
    m_full = fit_ridge(X_tr_full, y_train)
    model_results.append(score_model(m_full, X_ev_full, y_eval, "S2_full_baseline_plus_semantic"))

    log.info("%-45s %8s %8s %8s", "Model", "R²", "RMSE", "Pearson-r")
    for r in model_results:
        log.info("%-45s %8.4f %8.4f %8.4f", r["model"], r["r2"], r["rmse"], r["pearson_r"])

    # Incremental R² gain from adding semantic features
    r2_baseline = next(r["r2"] for r in model_results if r["model"] == "B3_metadata_plus_engagement")
    r2_full     = next(r["r2"] for r in model_results if r["model"] == "S2_full_baseline_plus_semantic")
    delta_r2    = r2_full - r2_baseline
    log.info("Incremental R² from semantic features over B3: %.4f", delta_r2)

    # -----------------------------------------------------------------------
    # 3. Feature importance (Ridge coefficients from full model)
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 3: Feature importance")

    scaler      = m_full.named_steps["scaler"]
    ridge       = m_full.named_steps["ridge"]
    coef_scaled = ridge.coef_                        # coefficients on standardized features
    feature_imp_rows = []
    for feat, coef in zip(full_cols, coef_scaled):
        feature_imp_rows.append({
            "feature": feat,
            "ridge_coef_standardized": round(float(coef), 6),
            "abs_coef": round(abs(float(coef)), 6),
            "feature_group": (
                "semantic"   if feat in SEMANTIC_FEATURES else
                "engagement" if feat in ENGAGEMENT_FEATURES else
                "genre"      if feat.startswith("genre_") else
                "metadata"
            ),
        })

    imp_df = pd.DataFrame(feature_imp_rows).sort_values("abs_coef", ascending=False)
    imp_df.to_csv(out_dir / "feature_importance.csv", index=False)
    log.info("Top 10 features by |coefficient|:")
    for _, row in imp_df.head(10).iterrows():
        log.info("  %-35s %.6f  [%s]", row["feature"], row["ridge_coef_standardized"], row["feature_group"])

    # -----------------------------------------------------------------------
    # 4. Cross-cut validation (genre, sequel, budget tier)
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("STEP 4: Cross-cut validation")

    eval_ok = eval_ok.copy()
    eval_ok["predicted_log_revenue"] = m_full.predict(X_ev_full)

    crosscut_rows = []

    # By sequel proxy
    for val, label in [(0, "non_sequel"), (1, "sequel_proxy")]:
        sub = eval_ok[eval_ok["is_sequel_proxy"] == val]
        if len(sub) >= 5:
            y_s = sub[OUTCOME].values
            p_s = sub["predicted_log_revenue"].values
            r2_s = r2_score(y_s, p_s)
            r_s, p_val = stats.pearsonr(p_s, y_s)
            crosscut_rows.append({"cut": f"sequel={label}", "n": len(sub),
                                   "r2": round(float(r2_s), 4), "pearson_r": round(float(r_s), 4)})
            log.info("  sequel=%s  n=%d  R²=%.4f  r=%.4f", label, len(sub), r2_s, r_s)

    # By budget tier (if log_budget present)
    if "log_budget" in eval_ok.columns and eval_ok["log_budget"].notna().any():
        budget_med = eval_ok["log_budget"].median()
        eval_ok["budget_tier"] = (eval_ok["log_budget"] >= budget_med).map({True: "high_budget", False: "low_budget"})
        for tier in ["high_budget", "low_budget"]:
            sub = eval_ok[eval_ok["budget_tier"] == tier]
            if len(sub) >= 5:
                y_s = sub[OUTCOME].values
                p_s = sub["predicted_log_revenue"].values
                r2_s = r2_score(y_s, p_s)
                r_s, pv = stats.pearsonr(p_s, y_s)
                crosscut_rows.append({"cut": tier, "n": len(sub),
                                       "r2": round(float(r2_s), 4), "pearson_r": round(float(r_s), 4)})
                log.info("  budget_tier=%s  n=%d  R²=%.4f  r=%.4f", tier, len(sub), r2_s, r_s)

    # By top genre columns
    for gcol in ["genre_action", "genre_animation", "genre_comedy", "genre_drama"]:
        if gcol in eval_ok.columns:
            sub = eval_ok[eval_ok[gcol] == 1]
            if len(sub) >= 5:
                y_s = sub[OUTCOME].values
                p_s = sub["predicted_log_revenue"].values
                r2_s = r2_score(y_s, p_s)
                r_s, pv = stats.pearsonr(p_s, y_s)
                crosscut_rows.append({"cut": gcol, "n": len(sub),
                                       "r2": round(float(r2_s), 4), "pearson_r": round(float(r_s), 4)})
                log.info("  %s  n=%d  R²=%.4f  r=%.4f", gcol, len(sub), r2_s, r_s)

    crosscut_df = pd.DataFrame(crosscut_rows)
    crosscut_df.to_csv(out_dir / "crosscut_results.csv", index=False)

    # -----------------------------------------------------------------------
    # 5. Signal verdict
    # -----------------------------------------------------------------------
    SIGNIFICANCE_THRESHOLD = 0.05
    DELTA_R2_THRESHOLD     = 0.02   # 2 pp incremental R² considered meaningful

    n_sig_corr  = sum(1 for r in corr_rows if r["pearson_p"] < SIGNIFICANCE_THRESHOLD)
    verdict     = "SIGNAL DETECTED" if (delta_r2 >= DELTA_R2_THRESHOLD or n_sig_corr >= 3) else "WEAK / NO SIGNAL"

    # -----------------------------------------------------------------------
    # 6. Write human-readable report
    # -----------------------------------------------------------------------
    report_lines = [
        "=" * 70,
        "ANALYSIS REPORT — YouTube Trailer Comment Semantic Signal Study",
        f"Evaluation split: {args.split_label}",
        "=" * 70,
        "",
        "CORRELATION ANALYSIS (univariate, train set):",
        f"  Semantic features tested:     {len(corr_rows)}",
        f"  Significant at p < 0.05:      {n_sig_corr}",
        f"  Significant at p < 0.10:      {sum(1 for r in corr_rows if r['pearson_p'] < 0.10)}",
        "",
        "Top 5 correlations with log_revenue:",
    ]
    for _, row in corr_df.head(5).iterrows():
        report_lines.append(f"  {row['feature']:<35} r={row['pearson_r']:+.4f}  p={row['pearson_p']:.4f}")

    report_lines += [
        "",
        "MODEL COMPARISON:",
        f"  {'Model':<45} {'R²':>8} {'RMSE':>8} {'Pearson-r':>10}",
    ]
    for r in model_results:
        report_lines.append(f"  {r['model']:<45} {r['r2']:>8.4f} {r['rmse']:>8.4f} {r['pearson_r']:>10.4f}")

    report_lines += [
        "",
        f"  Incremental R² (semantic over B3 baseline): {delta_r2:+.4f}",
        f"  Threshold for 'meaningful': {DELTA_R2_THRESHOLD:.2f}",
        "",
        "SIGNAL VERDICT:",
        f"  >>> {verdict} <<<",
        "",
        "Interpretation:",
    ]

    if verdict == "SIGNAL DETECTED":
        report_lines += [
            "  LLM-extracted semantic features add meaningful explanatory power",
            "  beyond engagement metrics and movie metadata.",
            "  Recommendation: proceed with the semantic-hybrid model for production.",
        ]
    else:
        report_lines += [
            "  LLM-extracted semantic features do not add meaningful explanatory",
            "  power over the engagement + metadata baseline.",
            "  Recommendation: reframe as 'hybrid engagement model' or revisit",
            "  semantic feature design and prompt engineering.",
        ]

    report_lines += ["", "=" * 70]
    report_text = "\n".join(report_lines)

    report_path = out_dir / f"analysis_report_{args.split_label}.txt"
    with open(report_path, "w") as f:
        f.write(report_text)
    log.info("Report written to %s", report_path)
    print("\n" + report_text)

    # Save all model results as JSON
    all_results = {
        "split": args.split_label,
        "delta_r2_semantic": round(float(delta_r2), 4),
        "n_significant_correlations": n_sig_corr,
        "verdict": verdict,
        "model_results": model_results,
    }
    with open(out_dir / f"correlation_results_{args.split_label}.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",       default="data/processed/train.csv")
    parser.add_argument("--eval",        default="data/processed/val.csv")
    parser.add_argument("--output_dir",  default="outputs/")
    parser.add_argument("--split_label", default="val",
                        help="'val' for development, 'test' for final evaluation")
    args = parser.parse_args()
    run_analysis(args)