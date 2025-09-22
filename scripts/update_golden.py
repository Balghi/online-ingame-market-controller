import os, json, pandas as pd, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", default="data/runs/fest_on_seed1337")
ap.add_argument("--snapshot", default="tests/golden_flags_trade_ids.json")
args = ap.parse_args()

flags = pd.read_csv(os.path.join(args.run_dir, "flags.csv"))
ids = sorted(flags.loc[flags["entity_type"]=="trade","entity_id"].astype(str).unique())

os.makedirs(os.path.dirname(args.snapshot), exist_ok=True)
with open(args.snapshot, "w") as f:
    json.dump(ids, f, indent=2)

print(f"Wrote {len(ids)} IDs to {args.snapshot}")
