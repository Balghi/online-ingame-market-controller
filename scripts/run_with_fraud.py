# scripts/run_with_fraud.py
import time
import argparse, os, pandas as pd
from sim.world import WorldConfig, gen_world
from features.accounts import make_account_daily_features
from detect.unsupervised import anomaly_accounts_iforest, map_account_anomalies_to_trades
from sim.market import MarketConfig, simulate_market
from sim.fraud import FraudConfig, inject_fraud, FlipConfig, inject_collusive_flips, inject_busy_traders_nonfraud
from detect.rules import (rolling_item_baselines, flag_under_overpriced, flag_rapid_flip,
                          to_flags_df, flag_mule_transfers, flag_wash_trading_heuristic, flag_wash_trading_concentration)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--players", type=int, default=600)
    ap.add_argument("--items", type=int, default=40)
    ap.add_argument("--hours", type=int, default=96)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--outdir", type=str, default="data/runs/mixed_seed1337")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    t0 = time.perf_counter()
    # 1) world + normal market
    wcfg = WorldConfig(n_players=args.players, n_items=args.items, seed=args.seed)
    players, items, rng = gen_world(wcfg)
    mcfg = MarketConfig(hours=args.hours, rng=rng)
    listings, trades = simulate_market(players, items, mcfg)

    t1 = time.perf_counter()
    # 2) inject fraud + labels
    fcfg = FraudConfig(seed=args.seed+99)
    listings2, trades2, labels = inject_fraud(players, items, listings, trades, fcfg)
    
    # Add busy traders
    listings2, trades2 = inject_busy_traders_nonfraud(players, items, listings2, trades2, fcfg,
                                                  n_clusters=4, group_size=3, window_hours=3)
    # Add flips
    flipcfg = FlipConfig(pairs=10, min_markup=0.40, flip_delay_minutes=5, underprice_factor=0.7)
    listings3, trades3, labels_flip = inject_collusive_flips(players, items, listings2, trades2, fcfg, flipcfg)
    labels = pd.concat([labels, labels_flip], ignore_index=True)


    # save listings3/trades3 as the new truth
    listings2, trades2 = listings3, trades3

    # save data
    players.to_csv(os.path.join(args.outdir, "players.csv"), index=False)
    items.to_csv(os.path.join(args.outdir, "items.csv"), index=False)
    listings2.to_csv(os.path.join(args.outdir, "listings.csv"), index=False)
    trades2.to_csv(os.path.join(args.outdir, "trades.csv"), index=False)
    labels.to_csv(os.path.join(args.outdir, "labels.csv"), index=False)

    t2 = time.perf_counter()
    # 3) baseline rules
    tb = rolling_item_baselines(trades2)
    ou = flag_under_overpriced(tb, k=2.5)
    rf = flag_rapid_flip(listings2, trades2, window_minutes=10, min_markup=0.4)
    flags = to_flags_df(ou, rf)
    mule = flag_mule_transfers(trades2, listings2, players, price_pct=0.5, max_buyer_age_days=14)
    wash_price = flag_wash_trading_heuristic(trades2, window_minutes=180, max_group_size=3,
                                            min_trades_in_window=5, overprice_sigma=1.8)

    wash_conc  = flag_wash_trading_concentration(trades2, window_minutes=240, max_group_size=3,
                                                min_trades=6, top_k_share=0.75)

    acct_df = make_account_daily_features(trades2)
    t3 = time.perf_counter()
    acct_anoms, _, _ = anomaly_accounts_iforest(acct_df, contamination=0.02, random_state=args.seed)
    ai_flags = map_account_anomalies_to_trades(trades2, acct_anoms)

    flags = pd.concat([
        to_flags_df(ou, rf),
        mule.assign(entity_type="trade", risk=0.9),
        wash_price.assign(entity_type="trade", risk=0.9),
        wash_conc.assign(entity_type="trade", risk=0.9),
        #wash_cycle.assign(entity_type="trade", risk=0.9),
        ai_flags
    ], ignore_index=True).drop_duplicates(subset=["entity_type","entity_id"])

    flags.to_csv(os.path.join(args.outdir, "flags.csv"), index=False)

    print(f"Saved mixed run with fraud to {args.outdir}")
    print(f"Players: {len(players)} | Items: {len(items)}")
    print(f"Listings: {len(listings2)} | Trades: {len(trades2)}")
    print(f"Labels: {len(labels)} | Flags: {len(flags)}")
    
    t4 = time.perf_counter()

    elapsed = {
        "gen_world_market_s": round(t1 - t0, 3),
        "injectors_s": round(t2 - t1, 3),
        "rules_s": round(t3 - t2, 3),
        "unsup_s": round(t4 - t3, 3),
        "total_s": round(t4 - t0, 3),
        "trades_n": len(trades2),
        "events_per_s": round(len(trades2) / max(1e-9, (t4 - t0)), 1)
    }
    pd.DataFrame([elapsed]).to_csv(os.path.join(args.outdir, "timings.csv"), index=False)
    print("Perf:", elapsed)

if __name__ == "__main__":
    main()
