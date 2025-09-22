import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

EXPLAIN_TOP_K = 3  # number of top features to list

def anomaly_accounts_iforest(account_df: pd.DataFrame, contamination=0.02, random_state=42):
    if account_df.empty:
        return (account_df.assign(anomaly_score=0.0, is_suspicious=0), None, 1.0)

    feat_cols = [c for c in account_df.columns if c not in ("player_id","day")]
    X = account_df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values

    model = IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X)

    raw = -model.score_samples(X)  # higher = more anomalous
    scores = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)

    out = account_df[["player_id","day"]].copy()
    out["anomaly_score"] = scores
    thresh = np.quantile(out["anomaly_score"], 1.0 - contamination)
    out["is_suspicious"] = (out["anomaly_score"] >= thresh).astype(int)

    return out.sort_values(["anomaly_score"], ascending=False), model, thresh

def explain_account_day_features(account_df: pd.DataFrame) -> pd.DataFrame:
    """Return per account-day feature z-scores and top EXPLAIN_TOP_K feature names."""
    df = account_df.copy()
    feat_cols = [c for c in df.columns if c not in ("player_id","day")]
    # z-score features
    mu = df[feat_cols].mean(axis=0)
    sd = df[feat_cols].std(axis=0).replace(0, 1.0)
    z = (df[feat_cols] - mu) / sd
    z = z.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # pick top-k by absolute z
    tops = []
    for i, row in z.iterrows():
        abs_vals = row.abs().sort_values(ascending=False)
        top_names = abs_vals.head(EXPLAIN_TOP_K).index.tolist()
        top_vals  = abs_vals.head(EXPLAIN_TOP_K).round(2).tolist()
        tops.append((";".join(top_names), ";".join(map(str, top_vals))))
    expl = df[["player_id","day"]].copy()
    expl["top_features"] = [t[0] for t in tops]
    expl["top_feature_abs_z"] = [t[1] for t in tops]
    return expl

def map_account_anomalies_to_trades(trades: pd.DataFrame, acct_anoms: pd.DataFrame):
    T = trades.copy()
    T["ts"] = pd.to_datetime(T["ts"])
    T["day"] = T["ts"].dt.floor("D")
    A = acct_anoms[acct_anoms["is_suspicious"]==1][["player_id","day"]].drop_duplicates()

    T = T.merge(A.rename(columns={"player_id":"buyer_id"}).assign(buyer_susp=1), on=["buyer_id","day"], how="left")
    T = T.merge(A.rename(columns={"player_id":"seller_id"}).assign(seller_susp=1), on=["seller_id","day"], how="left")
    T["susp_any"] = T[["buyer_susp","seller_susp"]].fillna(0).max(axis=1)

    flagged = T[T["susp_any"]==1]["trade_id"].unique().tolist()
    return pd.DataFrame({
        "entity_type":"trade",
        "entity_id": flagged,
        "risk": 0.85,
        "code": "AI_ACCOUNT",
        "detail": "account-day flagged by IsolationForest"
    })
