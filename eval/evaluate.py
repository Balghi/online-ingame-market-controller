# eval/evaluate.py
import argparse
import pandas as pd

def pr_counts(df):
    tp = ((df["_merge"]=="both") & (df["is_fraud"]==1)).sum()
    fp = ((df["_merge"]=="both") & (df["is_fraud"]==0)).sum()
    fn = ((df["_merge"]=="left_only") & (df["is_fraud"]==1)).sum()
    precision = tp/(tp+fp) if (tp+fp)>0 else 0.0
    recall    = tp/(tp+fn) if (tp+fn)>0 else 0.0
    return tp, fp, fn, precision, recall

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--flags", required=True)
    args = ap.parse_args()

    labels = pd.read_csv(args.labels)  # entity_type, entity_id, fraud_type, is_fraud
    flags  = pd.read_csv(args.flags)   # entity_type, entity_id, risk, code, detail
    L = labels[labels["entity_type"]=="trade"][["entity_id","is_fraud","fraud_type"]].rename(columns={"entity_id":"trade_id"})
    F = flags[flags["entity_type"]=="trade"][["entity_id","code"]].rename(columns={"entity_id":"trade_id"})

    df = L.merge(F, on="trade_id", how="left", indicator=True)

    # Overall
    tp, fp, fn, p, r = pr_counts(df)
    print(f"OVERALL -> TP:{tp} FP:{fp} FN:{fn} | Precision:{p:.3f} Recall:{r:.3f}")

    # By fraud_type (labels)
    for ft, g in df.groupby("fraud_type"):
        tp, fp, fn, p, r = pr_counts(g)
        print(f"{ft:>14} -> TP:{tp} FP:{fp} FN:{fn} | Precision:{p:.3f} Recall:{r:.3f}")

    # By rule code (flags)
    with_flags = df[df["_merge"]=="both"].copy()
    if not with_flags.empty and "code" in with_flags.columns:
        print("\nRule contribution (counts on detected fraud trades):")
        print(with_flags.groupby("code")["trade_id"].count().sort_values(ascending=False).to_string())
    else:
        print("\nNo rule attributions on detections.")

if __name__ == "__main__":
    main()
