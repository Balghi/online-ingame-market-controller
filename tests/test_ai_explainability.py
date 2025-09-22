import pandas as pd
import numpy as np
import pytest

from features.accounts import make_account_daily_features
from detect.unsupervised import anomaly_accounts_iforest, explain_account_day_features

def test_ai_explainability_columns_and_counts():
    # Create minimal trades for two accounts across two days
    rows = []
    ts0 = pd.Timestamp("2025-01-01 00:00:00")
    for i in range(10):
        rows.append({"trade_id":f"T{i}","ts":ts0 + pd.Timedelta(hours=i),
                     "buyer_id":"P1","seller_id":"S1","item_id":"I1","qty":1,"price":100,"listing_id":f"L{i}"})
    for i in range(8):
        rows.append({"trade_id":f"T{i+10}","ts":ts0 + pd.Timedelta(hours=i),
                     "buyer_id":"P2","seller_id":"S2","item_id":"I1","qty":1,"price":110,"listing_id":f"L{i+10}"})
    trades = pd.DataFrame(rows)

    acct_df = make_account_daily_features(trades)
    anoms, model, thresh = anomaly_accounts_iforest(acct_df, contamination=0.2, random_state=0)  # high contamination for test
    expl = explain_account_day_features(acct_df)
    # Merge to attach flags
    merged = expl.merge(anoms[["player_id","day","anomaly_score","is_suspicious"]], on=["player_id","day"], how="left")
    assert {"player_id","day","top_features","top_feature_abs_z","anomaly_score","is_suspicious"}.issubset(set(merged.columns))
    # Ensure we actually mark something as suspicious with high contamination
    assert (merged["is_suspicious"].fillna(0).astype(int).sum() >= 1), "Expected at least one suspicious account-day"
