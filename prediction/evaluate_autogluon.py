"""
One-off diagnostic: does AutoGluon's model search/ensembling beat the hand-picked XGBoost
model on the same features and the same leakage-safe holdout used in evaluate_horizons.py?

Not part of the production pipeline — autogluon.tabular is a heavy dependency (torch,
lightgbm, catboost, ...) that has no place in the free-tier GitHub Actions pipeline. This is
purely "did we leave accuracy on the table by hand-picking XGBoost with untuned hyperparameters."

Run: python -m prediction.evaluate_autogluon
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from autogluon.tabular import TabularPredictor

import prediction.predictor as predictor_module
from prediction.predictor import PlayerPredictor, TRAIN_HORIZONS, MAX_TRAINED_HORIZON, RECENCY_DECAY
from prediction.evaluate_horizons import _build_examples, _fit, HOLDOUT_LAST_N_GWS

predictor_module.TRAIN_LAST_N_GWS = 60  # see evaluate_horizons.py for why

TEST_HORIZONS = [1, 3, 7]  # subset — AutoGluon runs are slow, no need for all 6
TIME_LIMIT_SECS = 300


def main():
    p = PlayerPredictor()
    p.load_data()
    df = p._build_features()
    df['rank_from_end'] = df.groupby('name')['season_ordinal'].rank(method='first', ascending=False)

    buffer = HOLDOUT_LAST_N_GWS + MAX_TRAINED_HORIZON
    train_anchors = df[df['rank_from_end'] > buffer]
    test_anchor_pool = df[df['rank_from_end'] <= buffer]

    dropna_cols = p.feature_cols + ['target']
    stacked_train = _build_examples(train_anchors, p.feature_cols, TRAIN_HORIZONS).dropna(subset=dropna_cols)

    Xn = stacked_train[p.feature_cols].astype(np.float32)
    yn = stacked_train['target'].astype(np.float32)
    wn = RECENCY_DECAY ** (stacked_train['season_ordinal'].max() - stacked_train['season_ordinal'])

    print(f"training rows: {len(stacked_train)}")

    # Our current production-shaped model, for reference.
    xgb_model = _fit(Xn, yn, wn)

    # AutoGluon: same features, same target, same recency sample weight, searches/ensembles
    # across model families (LightGBM, CatBoost, RF, NN, XGBoost, ...) instead of one fixed one.
    ag_train = stacked_train[p.feature_cols + ['target']].copy()
    ag_train['target'] = ag_train['target'].astype(np.float32)
    ag_train['sample_weight'] = wn.values

    predictor = TabularPredictor(
        label='target', problem_type='regression', eval_metric='mean_absolute_error',
        sample_weight='sample_weight', verbosity=1,
    ).fit(ag_train, time_limit=TIME_LIMIT_SECS, presets='medium_quality')

    print(f"\n{'h':>3} | {'xgb MAE':>8} | {'autogluon MAE':>13} | {'n_test':>7}")
    print("-" * 42)
    for h in TEST_HORIZONS:
        examples = _build_examples(test_anchor_pool, p.feature_cols, [h]).dropna(subset=dropna_cols)
        examples = examples[examples['target_rank_from_end'] <= HOLDOUT_LAST_N_GWS]
        if len(examples) < 20:
            print(f"{h:>3} | {'--':>8} | {'--':>13} | {len(examples):>7}  (too few test rows)")
            continue

        X_test = examples[p.feature_cols].astype(np.float32)
        y_test = examples['target']

        xgb_mae = mean_absolute_error(y_test, xgb_model.predict(X_test))
        ag_mae = mean_absolute_error(y_test, predictor.predict(X_test[p.feature_cols]))
        print(f"{h:>3} | {xgb_mae:>8.3f} | {ag_mae:>13.3f} | {len(examples):>7}")

    print("\nleaderboard:")
    print(predictor.leaderboard(silent=True).to_string(index=False))


if __name__ == "__main__":
    main()
