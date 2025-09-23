import os
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="In-Game Market Controller — Demo", layout="wide")
def list_run_dirs():
    """Return a sorted list of run directories under demo/ and data/runs/ ."""
    candidates = []
    for base in ["demo", "data/runs"]:
        p = Path(base)
        if p.exists():
            for child in sorted(p.iterdir()):
                if child.is_dir():
                    if (child / "metrics_flat.csv").exists() or (child / "flags.csv").exists():
                        candidates.append(str(child))
    return candidates

@st.cache_data
def load_csv(path: str):
    if not Path(path).exists():
        return None
    try:
        if path.endswith(".csv"):
            return pd.read_csv(path)
        return None
    except Exception:
        return None

def load_run(run_dir: str):
    paths = {
        "metrics": f"{run_dir}/metrics_flat.csv",
        "flags": f"{run_dir}/flags.csv",
        "rule_contrib": f"{run_dir}/rule_contribution.csv",
        "ablation": f"{run_dir}/ablation_unique_tp.csv",
        "missed_by_type": f"{run_dir}/missed_by_type.csv",
        "miss_audit": f"{run_dir}/miss_audit.csv",
        "ai_explain": f"{run_dir}/ai_account_day_explain.csv",
    }
    return {k: load_csv(p) for k, p in paths.items()}

def kpi_row(metrics_df):
    overall_precision = metrics_df["overall_precision"].iloc[0] if "overall_precision" in metrics_df else None
    overall_recall = metrics_df["overall_recall"].iloc[0] if "overall_recall" in metrics_df else None
    tp = int(metrics_df["tp"].iloc[0]) if "tp" in metrics_df else None
    fp = int(metrics_df["fp"].iloc[0]) if "fp" in metrics_df else None
    fn = int(metrics_df["fn"].iloc[0]) if "fn" in metrics_df else None
    ca, cb, cc, cd, ce = st.columns(5)
    ca.metric("Precision", f"{overall_precision:.3f}" if overall_precision is not None else "—",
              help="Share of flags that are correct (higher = safer for fair players)")
    cb.metric("Recall", f"{overall_recall:.3f}" if overall_recall is not None else "—",
              help="Share of fraud caught (higher = more coverage)")
    cc.metric("TP", tp if tp is not None else "—")
    cd.metric("FP", fp if fp is not None else "—")
    ce.metric("FN", fn if fn is not None else "—")

def per_type_table(metrics_df):
    st.dataframe(
        metrics_df[["fraud_type","type_precision","type_recall","type_tp","type_fp","type_fn"]].rename(
            columns={
                "fraud_type":"Fraud Type",
                "type_precision":"Type P",
                "type_recall":"Type R",
                "type_tp":"TP",
                "type_fp":"FP",
                "type_fn":"FN",
            }
        ),
        use_container_width=True
    )

# Feature descriptions for AI explainability
FEATURE_EXPLAIN = {
    "b_trades": "Number of trades as buyer that day (spikes can be suspicious).",
    "s_trades": "Number of trades as seller that day (spikes can be suspicious).",
    "b_items": "Number of unique items bought that day (very low or very high can be odd).",
    "s_items": "Number of unique items sold that day (very low or very high can be odd).",
    "b_counterparties": "Number of unique sellers traded with as buyer (very low suggests collusion).",
    "s_counterparties": "Number of unique buyers traded with as seller (very low suggests collusion).",
    "b_price_z_mean": "Average price deviation from item median when buying (far from typical prices can be risky).",
    "s_price_z_mean": "Average price deviation from item median when selling (far from typical prices can be risky).",
    "b_price_z_max": "Maximum price deviation from item median when buying that day.",
    "s_price_z_max": "Maximum price deviation from item median when selling that day.",
    "max_pair_as_seller": "Maximum trades with any single buyer (high values suggest potential collusion).",
    "max_pair_as_buyer": "Maximum trades with any single seller (high values suggest potential collusion).",
    "buyer_hour_entropy": "How evenly the buyer's activity is spread across hours (low = concentrated bursts, bot-like).",
    "seller_hour_entropy": "How evenly the seller's activity is spread across hours (low = concentrated bursts, bot-like).",
    "trade_count": "How many trades the account made that day (spikes can be suspicious).",
    "distinct_counterparties": "Number of unique partners traded with (very low or very high can be odd).",
    "repeat_trade_ratio": "Share of trades with the same partners (high values suggest circles).",
    "mean_price_z": "Average z-score of prices vs item median (far from typical prices can be risky).",
    "max_price_z": "Maximum price deviation vs item median that day.",
    "flip_count": "Number of rapid flips the account participated in.",
    "concentration_topk_share": "Share of volume with the top few partners (high = potential collusion).",
    "pct_high_value_items": "Share of trades in high-value items (unusually high can be suspicious).",
    "qty_sum": "Total quantity traded that day (spikes are noteworthy).",
}

def explain_feature(name: str) -> str:
    n = str(name).strip()
    return FEATURE_EXPLAIN.get(n, f"{n}: Account/day pattern correlated with suspicious behavior in our data.")

st.sidebar.title("Run selector")
run_dirs = list_run_dirs()
if not run_dirs:
    st.sidebar.warning("No run folders found under demo/ or data/runs/. Generate artifacts or add the demo folder.")
    st.stop()

default_choice = [p for p in run_dirs if "fest_on_seed21" in p] or [run_dirs[0]]
run_dir = st.sidebar.selectbox("Primary run folder", run_dirs, index=run_dirs.index(default_choice[0]))

compare_mode = st.sidebar.checkbox("Compare with a second run")
run_dir_b = None
if compare_mode:
    rest = [p for p in run_dirs if p != run_dir]
    if not rest:
        st.sidebar.info("No other run folders available to compare.")
    else:
        if "fest_on" in run_dir:
            defaults = [p for p in rest if "fest_off" in p] or rest
        else:
            defaults = [p for p in rest if "fest_on" in p] or rest
        run_dir_b = st.sidebar.selectbox("Secondary run folder", rest, index=rest.index(defaults[0]))

tabs = st.tabs(["Single run", "Compare runs", "Festival summary"])

with tabs[0]:
    st.title("In-Game Market Controller — Demo Viewer")
    st.caption("Detectors prioritize **precision** (player trust) while keeping **recall** high. "
               "The AI layer surfaces suspicious account-days with plain-English reasons.")
    st.caption(f"Run: `{run_dir}`")
    data = load_run(run_dir)

    colA, colB = st.columns([3, 2])
    with colA:
        st.subheader("Topline KPIs")
        m = data["metrics"]
        if m is not None and not m.empty:
            kpi_row(m)
            st.caption("Per-type breakdown")
            per_type_table(m)
        else:
            st.info("metrics_flat.csv not found in this run.")

    with colB:
        st.subheader("Rule contribution (TP by rule)")
        rc = data["rule_contrib"]
        if rc is not None and not rc.empty:
            rc_sorted = rc.sort_values("tp_count", ascending=False)
            st.bar_chart(rc_sorted.set_index("code")["tp_count"])
            st.dataframe(rc_sorted, use_container_width=True, hide_index=True)
        else:
            st.info("rule_contribution.csv not found.")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ablation — Unique TPs by family")
        abl = data["ablation"]
        if abl is not None and not abl.empty:
            st.bar_chart(abl.set_index("family")["unique_tp"])
            st.dataframe(abl, use_container_width=True, hide_index=True)
        else:
            st.info("ablation_unique_tp.csv not found.")

    with col2:
        st.subheader("Miss audit (if any)")
        miss = data["miss_audit"]
        if miss is not None and not miss.empty:
            st.dataframe(miss, use_container_width=True)
            by_type = miss.groupby("fraud_type")["trade_id"].nunique().reset_index(name="missed")
            st.caption("Missed by type")
            st.dataframe(by_type, use_container_width=True, hide_index=True)
            st.download_button("Download miss_audit.csv", data=miss.to_csv(index=False), file_name="miss_audit.csv")
        else:
            st.success("No misses for this run.")

    st.markdown("---")

    st.subheader("AI explainability")
    ai = data["ai_explain"]
    if ai is not None and not ai.empty:
        colx, coly = st.columns([2, 3])
        with colx:
            st.caption("Suspicious account-days")
            susp = ai[ai["is_suspicious"]==1].copy() if "is_suspicious" in ai else ai.iloc[0:0].copy()
            st.metric("Count", int(susp.shape[0]))
            if not susp.empty and "top_features" in susp:
                top_features = (
                    susp["top_features"].fillna("").str.split(";").explode().str.strip()
                    .value_counts().reset_index()
                )
                top_features.columns = ["feature","count"]
                st.caption("Top features (frequency across suspicious account-days)")
                st.dataframe(top_features.head(10), use_container_width=True, hide_index=True)

                st.caption("What these mean:")
                for _, row in top_features.head(5).iterrows():
                    st.write(f"- **{row['feature']}** — {explain_feature(row['feature'])}")
        with coly:
            st.caption("Inspect account/day (suspicious only)")
            if not susp.empty:
                players = sorted(susp["player_id"].astype(str).unique().tolist())
                pid = st.selectbox("Player", options=players)
                days = sorted(susp[susp["player_id"].astype(str)==pid]["day"].astype(str).unique().tolist())
                day = st.selectbox("Day", options=days)
                row = susp[(susp["player_id"].astype(str)==pid) & (susp["day"].astype(str)==day)].head(1)
                st.dataframe(row, use_container_width=True, hide_index=True)
                st.download_button("Download AI explain CSV", data=ai.to_csv(index=False), file_name="ai_account_day_explain.csv")
    else:
        st.info("ai_account_day_explain.csv not found.")

with tabs[1]:
    st.header("Compare two runs")
    if not compare_mode or run_dir_b is None:
        st.info("Enable 'Compare with a second run' in the sidebar to use this tab.")
    else:
        st.caption(f"Primary: `{run_dir}`  |  Secondary: `{run_dir_b}`")
        st.caption("**Note:** Positive Δ recall means the secondary run outperforms the primary for that fraud type.")
        A = load_run(run_dir)
        B = load_run(run_dir_b)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Run A KPIs")
            if A["metrics"] is not None and not A["metrics"].empty:
                kpi_row(A["metrics"])
                st.caption("Per-type recall (A)")
                st.bar_chart(A["metrics"].set_index("fraud_type")["type_recall"])
            else:
                st.info("metrics_flat.csv missing for A")

        with col2:
            st.subheader("Run B KPIs")
            if B["metrics"] is not None and not B["metrics"].empty:
                kpi_row(B["metrics"])
                st.caption("Per-type recall (B)")
                st.bar_chart(B["metrics"].set_index("fraud_type")["type_recall"])
            else:
                st.info("metrics_flat.csv missing for B")

        st.markdown("#### Delta (B − A) — per-type recall")
        if A["metrics"] is not None and B["metrics"] is not None and not A["metrics"].empty and not B["metrics"].empty:
            mA = A["metrics"][["fraud_type","type_recall"]].set_index("fraud_type")
            mB = B["metrics"][["fraud_type","type_recall"]].set_index("fraud_type")
            delta = (mB - mA).fillna(0.0)
            st.dataframe(delta.rename(columns={"type_recall":"Δ recall"}), use_container_width=True)
        else:
            st.info("Need metrics for both runs to compute deltas.")

with tabs[2]:
    st.header("Festival summary (from data/sweeps/summary.csv)")
    summary = load_csv("data/sweeps/summary.csv")
    if summary is None or summary.empty:
        st.info("summary.csv not found. Run the sweep to generate it (python notebook.py).")
    else:
        overall = (summary.groupby("festival_on")
                   .agg(P_mean=("overall_precision","mean"),
                        P_std=("overall_precision","std"),
                        R_mean=("overall_recall","mean"),
                        R_std=("overall_recall","std"))
                   .reset_index())
        overall["festival_on"] = overall["festival_on"].map({0:"OFF",1:"ON"})
        st.subheader("Topline (overall)")
        st.dataframe(overall, use_container_width=True, hide_index=True)

        st.subheader("Per-type recall by festival")
        pertype = (summary.groupby(["festival_on","fraud_type"])
                   .agg(type_R=("type_recall","mean"))
                   .reset_index())
        pertype["festival_on"] = pertype["festival_on"].map({0:"OFF",1:"ON"})
        pivot = pertype.pivot(index="fraud_type", columns="festival_on", values="type_R")
        st.bar_chart(pivot)
        st.dataframe(pertype, use_container_width=True, hide_index=True)
