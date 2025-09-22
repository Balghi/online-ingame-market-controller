import argparse
import os
import pandas as pd
from sim.world import WorldConfig, gen_world
from sim.market import MarketConfig, simulate_market

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=500)
    ap.add_argument("--items", type=int, default=30)
    ap.add_argument("--hours", type=int, default=72)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="data/runs/baseline_seed42")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # world
    wcfg = WorldConfig(n_players=args.players, n_items=args.items, seed=args.seed)
    players, items, rng = gen_world(wcfg)

    # market
    mcfg = MarketConfig(hours=args.hours, rng=rng)
    listings, trades = simulate_market(players, items, mcfg)

    # save
    players.to_csv(os.path.join(args.outdir, "players.csv"), index=False)
    items.to_csv(os.path.join(args.outdir, "items.csv"), index=False)
    listings.to_csv(os.path.join(args.outdir, "listings.csv"), index=False)
    trades.to_csv(os.path.join(args.outdir, "trades.csv"), index=False)

    # quick sanity summary
    print(f"Saved to {args.outdir}")
    print(f"Players: {len(players)} | Items: {len(items)}")
    print(f"Listings: {len(listings)} | Trades: {len(trades)}")
    print("Time span:", listings['ts'].min(), "→", listings['ts'].max())

if __name__ == "__main__":
    main()
