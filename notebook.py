# notebook.py — seed sweeps with/without Festival, rich metrics per run
import os
import pandas as pd
from sim.world import WorldConfig, gen_world
from sim.market import MarketConfig, simulate_market
from sim.fraud import FraudConfig, inject_fraud, FlipConfig, inject_collusive_flips
from sim.fraud import inject_busy_traders_nonfraud  # you added this earlier
from detect.rules import (
    rolling_item_baselines, flag_under_overpriced, flag_rapid_flip, to_flags_df,
    flag_mule_transfers, flag_wash_trading_heuristic, flag_wash_trading_concentration,
    # flag_wash_cycles_k3,  # uncomment if you added it
)
from features.accounts import make_account_daily_features
from detect.unsupervised import anomaly_accounts_iforest, map_account_anomalies_to_trades, explain_account_day_features

# ---------- CONFIG ----------
SEEDS_ON  = [1337, 7, 21, 101, 303]     # festival ON
SEEDS_OFF = [808, 909, 111, 222, 333]   # festival OFF
PLAYERS, ITEMS, HOURS = 600, 40, 96

# Rules / detector knobs (tune here, or wire to CLI later)
UNDER_OVER_K = 2.5
FLIP_WINDOW_MIN = 25
FLIP_MIN_MARKUP = 0.25
WASH_PRICE_WINDOW = 180
WASH_PRICE_SIGMA  = 1.8
WASH_CONC_WINDOW  = 240
WASH_CONC_TOPK    = 0.75
WASH_CONC_MAX_G   = 3
WASH_CONC_MIN_TR  = 6
IF_CONTAM = 0.02

# Festival knobs (only used when festival_on=True)
FESTIVAL_HOURS = 24
FESTIVAL_AMP   = 0.25
FESTIVAL_START = "2025-01-03 00:00:00"

# Busy-but-benign traders (non-fraud, to challenge wash rules)
BUSY_CLUSTERS = 4
BUSY_GROUP_SZ = 3
BUSY_WINDOW_H = 3

# Collusive flips injector config
FLIP_PAIRS = 10
FLIP_DELAY_POOL = [5,7,12,18,35,40]  # used inside injector (you may have added this)
# ---------- /CONFIG ----------

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    
def unique_tp_by_family(labels_path: str, flags_path: str) -> pd.DataFrame:
    L = pd.read_csv(labels_path)
    F = pd.read_csv(flags_path)
    L = L[L.entity_type=="trade"][["entity_id","fraud_type","is_fraud"]].rename(columns={"entity_id":"trade_id"})
    F = F[F.entity_type=="trade"][["entity_id","code"]].rename(columns={"entity_id":"trade_id"})
    df = L.merge(F, on="trade_id", how="left", indicator=True)
    tp_df = df[(df["_merge"]=="both") & (df["is_fraud"]==1)]

    families = {
      "RULES_PRICE": {"UNDERPRICED","OVERPRICED"},
      "RULES_MULE": {"MULE_RULE"},
      "RULES_WASH": {"WASH_CONC","WASH_HEUR","WASH_CYCLE3"},
      "RULES_FLIP": {"RAPID_FLIP"},
      "AI": {"AI_ACCOUNT"},
    }
    rows = []
    for name, codes in families.items():
        tps = tp_df[tp_df["code"].isin(codes)]["trade_id"].nunique()
        others = tp_df[~tp_df["code"].isin(codes)]["trade_id"].unique().tolist()
        uniq = tp_df[tp_df["code"].isin(codes) & ~tp_df["trade_id"].isin(others)]["trade_id"].nunique()
        rows.append({"family":name, "tp":int(tps), "unique_tp":int(uniq)})
    return pd.DataFrame(rows).sort_values("unique_tp", ascending=False)


def run_pipeline(seed: int, outdir: str, festival_on: bool):
    ensure_dir(outdir)

    # 1) world + market
    wcfg = WorldConfig(n_players=PLAYERS, n_items=ITEMS, seed=seed)
    P, I, rng = gen_world(wcfg)
    mcfg = MarketConfig(hours=HOURS, rng=rng,
                        festival_on=festival_on,
                        festival_start=FESTIVAL_START,
                        festival_hours=FESTIVAL_HOURS,
                        festival_amp=FESTIVAL_AMP)
    L, T = simulate_market(P, I, mcfg)

    # 2) fraud injectors (wash + mules)
    fcfg = FraudConfig(seed=seed+99)
    L2, T2, labels = inject_fraud(P, I, L, T, fcfg)

    # 2b) collusive flips
    flipcfg = FlipConfig(pairs=FLIP_PAIRS, min_markup=0.40, flip_delay_minutes=5, underprice_factor=0.7)
    L3, T3, labels_flip = inject_collusive_flips(P, I, L2, T2, fcfg, flipcfg)
    labels = pd.concat([labels, labels_flip], ignore_index=True)
    L, T = L3, T3

    # 2c) busy-but-benign concentration (non-fraud)
    L, T = inject_busy_traders_nonfraud(P, I, L, T, fcfg,
                                        n_clusters=BUSY_CLUSTERS,
                                        group_size=BUSY_GROUP_SZ,
                                        window_hours=BUSY_WINDOW_H)

    # 3) rules
    tb = rolling_item_baselines(T)
    ou = flag_under_overpriced(tb, k=UNDER_OVER_K)
    rf = flag_rapid_flip(L, T, window_minutes=FLIP_WINDOW_MIN, min_markup=FLIP_MIN_MARKUP)
    mule = flag_mule_transfers(T, L, P, price_pct=0.5, max_buyer_age_days=14)
    wash_price = flag_wash_trading_heuristic(T, window_minutes=WASH_PRICE_WINDOW, max_group_size=3,
                                             min_trades_in_window=5, overprice_sigma=WASH_PRICE_SIGMA)
    wash_conc  = flag_wash_trading_concentration(T, window_minutes=WASH_CONC_WINDOW,
                                                 max_group_size=WASH_CONC_MAX_G,
                                                 min_trades=WASH_CONC_MIN_TR,
                                                 top_k_share=WASH_CONC_TOPK)
    # If you added cycles:
    # wash_cyc = flag_wash_cycles_k3(T, window_minutes=240, min_cycles=1)

    rule_flags = pd.concat([
        to_flags_df(ou, rf),
        mule.assign(entity_type="trade", risk=0.9),
        wash_price.assign(entity_type="trade", risk=0.9),
        wash_conc.assign(entity_type="trade", risk=0.9),
        # wash_cyc.assign(entity_type="trade", risk=0.9),
    ], ignore_index=True).drop_duplicates(subset=["entity_type","entity_id"])

    # 4) unsupervised (IsolationForest on account/day) + map to trades
    acct_df = make_account_daily_features(T)
    acct_anoms, _, _ = anomaly_accounts_iforest(acct_df, contamination=IF_CONTAM, random_state=seed)
    acct_explain = explain_account_day_features(acct_df).merge(
        acct_anoms[["player_id","day","anomaly_score","is_suspicious"]],
        on=["player_id","day"], how="left"
    )
    acct_explain.to_csv(os.path.join(outdir, "ai_account_day_explain.csv"), index=False)

    ai_flags = map_account_anomalies_to_trades(T, acct_anoms)

    flags = pd.concat([rule_flags, ai_flags], ignore_index=True)\
              .drop_duplicates(subset=["entity_type","entity_id"])

    # 5) persist run artifacts
    P.to_csv(os.path.join(outdir, "players.csv"), index=False)
    I.to_csv(os.path.join(outdir, "items.csv"), index=False)
    L.to_csv(os.path.join(outdir, "listings.csv"), index=False)
    T.to_csv(os.path.join(outdir, "trades.csv"), index=False)
    labels.to_csv(os.path.join(outdir, "labels.csv"), index=False)
    flags.to_csv(os.path.join(outdir, "flags.csv"), index=False)

    return labels, flags

def evaluate_detailed(labels: pd.DataFrame, flags: pd.DataFrame) -> dict:
    # Merge on trades only
    L = labels[labels["entity_type"]=="trade"][["entity_id","is_fraud","fraud_type"]]\
            .rename(columns={"entity_id":"trade_id"})
    F = flags[flags["entity_type"]=="trade"][["entity_id","code"]].rename(columns={"entity_id":"trade_id"})
    df = L.merge(F, on="trade_id", how="left", indicator=True)

    def pr_counts(sub):
        tp = ((sub["_merge"]=="both") & (sub["is_fraud"]==1)).sum()
        fp = ((sub["_merge"]=="both") & (sub["is_fraud"]==0)).sum()
        fn = ((sub["_merge"]=="left_only") & (sub["is_fraud"]==1)).sum()
        p  = tp/(tp+fp) if (tp+fp)>0 else 0.0
        r  = tp/(tp+fn) if (tp+fn)>0 else 0.0
        return tp, fp, fn, p, r

    # Overall
    tp, fp, fn, p, r = pr_counts(df)
    out = {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 3), "recall": round(r, 3),
    }

    # Per-type
    per_type = []
    for ft, g in df.groupby("fraud_type"):
        tpi, fpi, fni, pi, ri = pr_counts(g)
        per_type.append({"fraud_type": ft, "tp": tpi, "fp": fpi, "fn": fni,
                         "precision": round(pi,3), "recall": round(ri,3)})
    out["per_type"] = per_type

    # Rule contributions on detected fraud trades
    detected = df[(df["_merge"]=="both") & (df["is_fraud"]==1)]
    if not detected.empty:
        contrib = detected.groupby("code")["trade_id"].nunique().reset_index().rename(columns={"trade_id":"tp_count"})
        out["rule_contribution"] = contrib.sort_values("tp_count", ascending=False).to_dict(orient="records")
    else:
        out["rule_contribution"] = []

    # Missed by type + lightweight audit rows
    missed = df[(df["_merge"]=="left_only") & (df["is_fraud"]==1)].copy()
    out["missed_by_type"] = missed.groupby("fraud_type")["trade_id"].nunique().reset_index()\
                                  .rename(columns={"trade_id":"missed"}).to_dict(orient="records")
    out["missed_trade_ids"] = missed["trade_id"].tolist()   # used by notebook to build audit CSV

    return out


def flatten_metrics(seed: int, festival_on: bool, metrics: dict) -> pd.DataFrame:
    rows = []
    base = {
        "seed": seed,
        "festival_on": int(festival_on),
        "overall_precision": metrics["precision"],
        "overall_recall": metrics["recall"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
    }
    # per-type rows
    for pt in metrics["per_type"]:
        row = base | {
            "fraud_type": pt["fraud_type"],
            "type_precision": pt["precision"],
            "type_recall": pt["recall"],
            "type_tp": pt["tp"],
            "type_fp": pt["fp"],
            "type_fn": pt["fn"],
        }
        rows.append(row)
    return pd.DataFrame(rows)

def run_sweep():
    summary_rows = []
    ensure_dir("data/sweeps")

    # two passes: festival ON + OFF
    for festival_on, seeds in [(True, SEEDS_ON), (False, SEEDS_OFF)]:
        for seed in seeds:
            tag = "fest_on" if festival_on else "fest_off"
            outdir = f"data/runs/{tag}_seed{seed}"
            labels, flags = run_pipeline(seed, outdir, festival_on=festival_on)
            metrics = evaluate_detailed(labels, flags)

            abl = unique_tp_by_family(os.path.join(outdir,"labels.csv"), os.path.join(outdir,"flags.csv"))
            abl.to_csv(os.path.join(outdir, "ablation_unique_tp.csv"), index=False)

            # Save per-run detail CSVs
            flat = flatten_metrics(seed, festival_on, metrics)
            flat.to_csv(os.path.join(outdir, "metrics_flat.csv"), index=False)

            # Save rule contributions & misses for this run (for later inspection)
            pd.DataFrame(metrics["rule_contribution"]).to_csv(os.path.join(outdir, "rule_contribution.csv"), index=False)
            pd.DataFrame(metrics["missed_by_type"]).to_csv(os.path.join(outdir, "missed_by_type.csv"), index=False)

            if metrics["missed_trade_ids"]:
                # Re-read trades/listings/players to enrich the audit
                T = pd.read_csv(os.path.join(outdir, "trades.csv"), parse_dates=["ts"])
                Ltab = pd.read_csv(os.path.join(outdir, "labels.csv"))  # to get fraud_type
                Ltab = Ltab[Ltab["entity_type"]=="trade"][["entity_id","fraud_type"]].rename(columns={"entity_id":"trade_id"})
                # item-level median to compute quick z
                item_med = T.groupby("item_id")["price"].median().rename("item_median")
                audit = T[T["trade_id"].isin(metrics["missed_trade_ids"])].merge(item_med, on="item_id", how="left")
                audit["price_vs_median_pct"] = ((audit["price"] - audit["item_median"]) / audit["item_median"]).round(3)
                audit = audit.merge(Ltab, on="trade_id", how="left")
                audit = audit[["trade_id","fraud_type","item_id","seller_id","buyer_id","ts","price","item_median","price_vs_median_pct"]]
                audit.to_csv(os.path.join(outdir, "miss_audit.csv"), index=False)
            # Append to master summary
            summary_rows.append(flat)

            # Console summary
            print(f"\n=== {tag.upper()} | seed={seed} ===")
            print(f"OVERALL  P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  TP={metrics['tp']} FP={metrics['fp']} FN={metrics['fn']}")
            for pt in metrics["per_type"]:
                print(f"{pt['fraud_type']:>14}  P={pt['precision']:.3f}  R={pt['recall']:.3f}  TP={pt['tp']} FN={pt['fn']}")
            if metrics["rule_contribution"]:
                print("Rule contribution (TP by rule):")
                for rc in metrics["rule_contribution"]:
                    print(f"  {rc['code']:>12}: {rc['tp_count']}")
            if metrics["missed_by_type"]:
                print("Missed by type:")
                for m in metrics["missed_by_type"]:
                    print(f"  {m['fraud_type']:>14}: {m['missed']}")

    # Combine all and write a single summary
    summary = pd.concat(summary_rows, ignore_index=True)
    summary.to_csv("data/sweeps/summary.csv", index=False)

    # Also print an aggregate view (mean±std by festival_on & fraud_type)
    agg = (summary
           .groupby(["festival_on","fraud_type"])
           .agg(overall_P=("overall_precision","mean"),
                overall_R=("overall_recall","mean"),
                type_P=("type_precision","mean"),
                type_R=("type_recall","mean"))
           .reset_index())
    print("\n=== Aggregate means by festival_on & fraud_type ===")
    print(agg.to_string(index=False))

    # Quick topline (overall only)
    topline = (summary
               .groupby("festival_on")
               .agg(P_mean=("overall_precision","mean"), P_std=("overall_precision","std"),
                    R_mean=("overall_recall","mean"),    R_std=("overall_recall","std"))
               .reset_index())
    topline["festival_on"] = topline["festival_on"].map({0:"OFF",1:"ON"})
    print("\n=== Topline (overall) ===")
    print(topline.to_string(index=False))

if __name__ == "__main__":
    run_sweep()
    print("\nSaved sweep summary to data/sweeps/summary.csv (per-run details live under data/runs/…) ")
