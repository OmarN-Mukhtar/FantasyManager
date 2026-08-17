"""
Compares the gameweeks_ahead-aware model against a single-step baseline (h=1 training only,
iterated across fixtures — what predictor.py did before this feature) at increasing horizons.

Holdout is buffered: training anchors sit at least MAX_TRAINED_HORIZON rows before the holdout
region, so no training pair's target ever lands inside it — otherwise the horizon model would
partly train on the same (anchor, horizon) pairs used to test it. Test anchors are drawn only
from what's left over, keeping this a genuine out-of-sample comparison.

Run: python -m prediction.evaluate_horizons
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

import prediction.predictor as predictor_module
from prediction.predictor import PlayerPredictor, TRAIN_HORIZONS, MAX_TRAINED_HORIZON, RECENCY_DECAY

TEST_HORIZONS = [1, 2, 3, 5, 7, 10]
HOLDOUT_LAST_N_GWS = 15

# Production caps each player's training history at 25 rows (TRAIN_LAST_N_GWS), which doesn't
# leave enough room for a leakage-safe buffered holdout at these horizons. Widen it for this
# evaluation only — _build_features reads the module global at call time.
predictor_module.TRAIN_LAST_N_GWS = 60


def _fit(X, y, sample_weight, max_depth=6):
    m = xgb.XGBRegressor(
        max_depth=max_depth, learning_rate=0.1, n_estimators=150,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    m.fit(X, y, sample_weight=sample_weight)
    return m


def _build_examples(df, feature_cols, horizons):
    """One row per (anchor, h): rolling stats frozen at the anchor, target/fixture-context
    pulled from h gameweeks ahead of it."""
    grouped = df.groupby('name', sort=False)
    base_cols = [c for c in feature_cols if c != 'gameweeks_ahead']
    parts = []
    for h in horizons:
        part = df[base_cols + ['season_ordinal', 'rank_from_end']].copy()
        part['target'] = grouped['total_points'].shift(-h)
        part['opponent_team_feature'] = grouped['opponent_team_feature'].shift(-h)
        part['is_home_feature'] = grouped['is_home_feature'].shift(-h)
        part['fdr_feature'] = grouped['fdr_feature'].shift(-h)
        part['gameweeks_ahead'] = float(h)
        part['target_rank_from_end'] = part['rank_from_end'] - h
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def main():
    p = PlayerPredictor()
    p.load_data()
    df = p._build_features()
    df['rank_from_end'] = df.groupby('name')['season_ordinal'].rank(method='first', ascending=False)

    buffer = HOLDOUT_LAST_N_GWS + MAX_TRAINED_HORIZON
    train_anchors = df[df['rank_from_end'] > buffer]
    test_anchor_pool = df[df['rank_from_end'] <= buffer]

    dropna_cols = p.feature_cols + ['target']
    baseline_cols = [c for c in p.feature_cols if c != 'gameweeks_ahead']

    stacked_new = _build_examples(train_anchors, p.feature_cols, TRAIN_HORIZONS).dropna(subset=dropna_cols)
    Xn = stacked_new[p.feature_cols].astype(np.float32)
    yn = stacked_new['target'].astype(np.float32)
    wn = RECENCY_DECAY ** (stacked_new['season_ordinal'].max() - stacked_new['season_ordinal'])
    horizon_model = _fit(Xn, yn, wn)

    stacked_base = _build_examples(train_anchors, p.feature_cols, [1]).dropna(subset=baseline_cols + ['target'])
    Xb = stacked_base[baseline_cols].astype(np.float32)
    yb = stacked_base['target'].astype(np.float32)
    wb = RECENCY_DECAY ** (stacked_base['season_ordinal'].max() - stacked_base['season_ordinal'])
    baseline_model = _fit(Xb, yb, wb)

    print(f"training rows: horizon-aware={len(stacked_new)}, baseline={len(stacked_base)}\n")
    print(f"{'h':>3} | {'new MAE':>8} | {'baseline MAE':>13} | {'n_test':>7}")
    print("-" * 42)
    for h in TEST_HORIZONS:
        examples = _build_examples(test_anchor_pool, p.feature_cols, [h]).dropna(subset=dropna_cols)
        examples = examples[examples['target_rank_from_end'] <= HOLDOUT_LAST_N_GWS]
        if len(examples) < 20:
            print(f"{h:>3} | {'--':>8} | {'--':>13} | {len(examples):>7}  (too few test rows)")
            continue

        Xn_test = examples[p.feature_cols].astype(np.float32)
        Xb_test = examples[baseline_cols].astype(np.float32)
        y_test = examples['target']

        new_mae = mean_absolute_error(y_test, horizon_model.predict(Xn_test))
        base_mae = mean_absolute_error(y_test, baseline_model.predict(Xb_test))
        print(f"{h:>3} | {new_mae:>8.3f} | {base_mae:>13.3f} | {len(examples):>7}")


if __name__ == "__main__":
    main()
