import pandas as pd
import numpy as np

def rolling_item_baselines(trades: pd.DataFrame, window="24h"):
    df = trades.copy()
    df = df.sort_values("ts")
    df["ts"] = pd.to_datetime(df["ts"])
    out = []
    for it, g in df.groupby("item_id"):
        g = g.set_index("ts").sort_index()
        med = g["price"].rolling(window).median().rename("roll_median")
        mad = (g["price"] - med).abs().rolling(window).median().rename("roll_mad")
        tmp = g.assign(roll_median=med, roll_mad=mad).reset_index()
        tmp["item_id"] = it
        out.append(tmp)
    return pd.concat(out, ignore_index=True)

def flag_under_overpriced(trades_with_baseline: pd.DataFrame, k=3.5):
    df = trades_with_baseline.copy()
    df["mad"] = df["roll_mad"].replace(0, np.nan).fillna(df["price"].median() * 0.05 + 1e-6)
    z = (df["price"] - df["roll_median"]) / (1.4826 * df["mad"])
    df["z"] = z
    flags = []
    for i, row in df.iterrows():
        if pd.isna(row["roll_median"]): 
            continue
        if row["z"] >= k:
            flags.append((row["trade_id"], "OVERPRICED", f"+{row['z']:.1f}σ vs median"))
        elif row["z"] <= -k:
            flags.append((row["trade_id"], "UNDERPRICED", f"{row['z']:.1f}σ vs median"))
    return pd.DataFrame(flags, columns=["entity_id","code","detail"])

def flag_rapid_flip(listings: pd.DataFrame, trades: pd.DataFrame,
                    window_minutes=25, min_markup=0.25,
                    adaptive=True, slow_window_extra=65, slow_min_markup=0.45):
    """
    Detect rapid flip transactions using adaptive window sizes.
    Flags both the buy leg and the resale leg.
    """
    L = listings.copy(); T = trades.copy()
    L["ts"] = pd.to_datetime(L["ts"]); T["ts"] = pd.to_datetime(T["ts"])

    item_median = T.groupby("item_id")["price"].median().rename("item_median")
    TT = T.merge(L[["listing_id","list_price","ts"]].rename(columns={"ts":"list_ts"}), on="listing_id", how="left")\
          .merge(item_median, on="item_id", how="left")

    flips = []

    def dyn_win(row, base):
        if not adaptive or pd.isna(row.get("item_median", None)):
            return base
        alt = min(40, 10 + 0.005 * float(row["item_median"]))
        return max(base, int(alt))
    L_by_item = L.sort_values("ts").groupby("item_id")
    for _, r in TT.iterrows():
        buyer = r["buyer_id"]; it = r["item_id"]; buy_ts = r["ts"]; buy_price = r["price"]
        grp = L_by_item.get_group(it)
        w = dyn_win(r, window_minutes)
        mask = (grp["seller_id"] == buyer) & (grp["ts"] >= buy_ts) & (grp["ts"] <= buy_ts + pd.Timedelta(minutes=w))
        nxt = grp[mask]
        if not nxt.empty:
            best = nxt.sort_values("ts").iloc[0]
            markup = (best["list_price"] - buy_price) / max(1.0, buy_price)
            if markup >= min_markup:
                flips.append((r["trade_id"], "RAPID_FLIP",
                              f"{int(markup*100)}% relist in {int((best['ts']-buy_ts).total_seconds()/60)}m"))
    T_by_item = T.sort_values("ts").groupby("item_id")
    for _, r in TT.iterrows():
        buyer = r["buyer_id"]; it = r["item_id"]; buy_ts = r["ts"]; buy_price = r["price"]
        grp = T_by_item.get_group(it)
        w_fast = dyn_win(r, window_minutes)
        w_slow = w_fast + slow_window_extra
        mask_fast = (grp["seller_id"] == buyer) & (grp["ts"] >= buy_ts) & (grp["ts"] <= buy_ts + pd.Timedelta(minutes=w_fast))
        mask_slow = (grp["seller_id"] == buyer) & (grp["ts"] >  buy_ts + pd.Timedelta(minutes=w_fast)) & (grp["ts"] <= buy_ts + pd.Timedelta(minutes=w_slow))

        for label, msk, req_markup in [("fast", mask_fast, min_markup), ("slow", mask_slow, slow_min_markup)]:
            nxt_sale = grp[msk]
            if not nxt_sale.empty:
                s = nxt_sale.sort_values("ts").iloc[0]
                markup = (s["price"] - buy_price) / max(1.0, buy_price)
                if markup >= req_markup:
                    mins = int((s["ts"] - buy_ts).total_seconds() / 60)
                    flips.append((r["trade_id"], "RAPID_FLIP", f"{int(markup*100)}% resale in {mins}m ({label})"))
                    flips.append((s["trade_id"], "RAPID_FLIP", "second leg of collusive flip"))

    if not flips:
        return pd.DataFrame(columns=["entity_id","code","detail"])
    return pd.DataFrame(flips, columns=["entity_id","code","detail"]).drop_duplicates("entity_id")



def flag_mule_transfers(trades: pd.DataFrame,
                        listings: pd.DataFrame,
                        players: pd.DataFrame,
                        price_pct=0.5,
                        max_buyer_age_days=14,
                        extended_age_days=30,
                        high_value_multiplier=2.5):
    """
    Flags cheap sales to new accounts.
    Cheapness is required in all cases: price <= price_pct * reference.
    Age rule:
      - Buyer age <= max_buyer_age_days, OR
      - Buyer age <= extended_age_days AND the *item* is high-value
        (item_p95 >= high_value_multiplier * item_median).
    Reference price is a robust per-item baseline (rolling median fallback to global median).
    """
    T = trades.copy(); L = listings.copy()
    T["ts"] = pd.to_datetime(T["ts"]); L["ts"] = pd.to_datetime(L["ts"])
    P = players[["player_id","account_age_days"]].rename(columns={"player_id":"buyer_id"})

    # Global item median
    item_global_med = T.groupby("item_id")["price"].median().rename("item_global_median")

    # Rolling proxy (optional/simple): rolling median per item on time-sorted series
    T_sorted = T.sort_values("ts").merge(item_global_med, on="item_id", how="left")
    T_sorted["roll24_median"] = T_sorted.groupby("item_id")["price"]\
        .transform(lambda s: s.rolling(window=200, min_periods=10).median())
    # Reference = rolling median fallback to global median
    T_sorted["item_ref"] = T_sorted["roll24_median"].fillna(T_sorted["item_global_median"])

    # Quantiles to determine high-value items
    item_p95 = T.groupby("item_id")["price"].quantile(0.95).rename("item_p95")
    item_med = T.groupby("item_id")["price"].median().rename("item_median")

    TT = (T_sorted
          .merge(P, on="buyer_id", how="left")
          .merge(item_p95, on="item_id", how="left")
          .merge(item_med, on="item_id", how="left"))

    flags = []
    for _, r in TT.iterrows():
        ref = r["item_ref"] if pd.notna(r["item_ref"]) else r["item_global_median"]
        buyer_age = r.get("account_age_days", 999999)

        # Cheapness (always required)
        too_cheap = r["price"] <= price_pct * ref

        # High-value *item* (not trade): p95 much higher than median
        item_is_high_value = (pd.notna(r.get("item_p95")) and pd.notna(r.get("item_median"))
                              and (r["item_p95"] >= high_value_multiplier * r["item_median"]))

        # Age rule: very new OR (somewhat new AND item is high-value)
        age_rule = (buyer_age <= max_buyer_age_days) or (buyer_age <= extended_age_days and item_is_high_value)

        if too_cheap and age_rule:
            flags.append((r["trade_id"], "MULE_RULE",
                          f"buyer_age={int(buyer_age)}d, price {r['price']:.0f} ≤ {int(price_pct*100)}% of ref {ref:.0f}"))

    return pd.DataFrame(flags, columns=["entity_id","code","detail"])



def flag_wash_trading_heuristic(trades: pd.DataFrame,
                                window_minutes=120,
                                max_group_size=3,
                                min_trades_in_window=5,
                                overprice_sigma=2.0):
    df = trades.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    # robust baseline per item for z-scores
    med = df.groupby("item_id")["price"].median().rename("med")
    mad = df.groupby("item_id")["price"].apply(lambda s: (s - s.median()).abs().median()).rename("mad")
    df = df.merge(med, on="item_id", how="left").merge(mad, on="item_id", how="left")
    df["mad"] = df["mad"].replace(0, df["price"].median()*0.05 + 1e-6)
    df["z"] = (df["price"] - df["med"]) / (1.4826 * df["mad"])

    flags = []
    # For each item, slide a simple window and look for dense tiny groups with overprice
    for it, g in df.groupby("item_id"):
        g = g.sort_values("ts").reset_index(drop=True)
        for i in range(len(g)):
            t0 = g.loc[i, "ts"]; t1 = t0 + pd.Timedelta(minutes=window_minutes)
            w = g[(g["ts"] >= t0) & (g["ts"] <= t1)]
            actors = set(w["buyer_id"]).union(set(w["seller_id"]))
            if len(w) >= min_trades_in_window and len(actors) <= max_group_size:
                # require most trades to be overpriced
                over = (w["z"] >= overprice_sigma).sum()
                if over >= max(3, int(0.6 * len(w))):
                    # flag every trade in window as suspicious wash trades
                    for tid in w["trade_id"].tolist():
                        flags.append((tid, "WASH_HEUR", f"{len(w)} trades among {len(actors)} accounts, overpriced cluster"))
    if not flags:
        return pd.DataFrame(columns=["entity_id","code","detail"])
    out = pd.DataFrame(flags, columns=["entity_id","code","detail"]).drop_duplicates("entity_id")
    return out

def flag_wash_trading_concentration(trades: pd.DataFrame,
                                    window_minutes=180,
                                    max_group_size=3,
                                    min_trades=6,
                                    top_k_share=0.8):
    """
    Flags windows where a small group of accounts executes the majority of trades (by count),
    indicating circular trading, regardless of price level.
    """
    df = trades.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    flags = []

    for it, g in df.groupby("item_id"):
        g = g.sort_values("ts").reset_index(drop=True)
        for i in range(len(g)):
            t0 = g.loc[i, "ts"]; t1 = t0 + pd.Timedelta(minutes=window_minutes)
            w = g[(g["ts"] >= t0) & (g["ts"] <= t1)]
            if len(w) < min_trades:
                continue
            actors = pd.concat([w["buyer_id"], w["seller_id"]]).value_counts()
            # take smallest set of actors that explain top_k_share of actor appearances
            cum = actors.cumsum() / actors.sum()
            k = (cum <= top_k_share).sum()
            if k <= max_group_size:
                # strong concentration among very few accounts
                for tid in w["trade_id"].tolist():
                    flags.append((tid, "WASH_CONC", f"{len(w)} trades; {k} actors cover ≥{int(top_k_share*100)}% of appearances"))
    if not flags:
        return pd.DataFrame(columns=["entity_id","code","detail"])
    return pd.DataFrame(flags, columns=["entity_id","code","detail"]).drop_duplicates("entity_id")
def flag_wash_cycles_k3(trades: pd.DataFrame, window_minutes=240, min_cycles=1):
    df = trades.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    flags = []
    for it, g in df.groupby("item_id"):
        g = g.sort_values("ts").reset_index(drop=True)
        for i in range(len(g)):
            t0 = g.loc[i, "ts"]; t1 = t0 + pd.Timedelta(minutes=window_minutes)
            w = g[(g["ts"] >= t0) & (g["ts"] <= t1)]
            # build directed edges (seller -> buyer)
            edges = list(zip(w["seller_id"], w["buyer_id"], w["trade_id"]))
            # index edges by pair for O(1) check
            pair_to_trades = {}
            for s,b,tid in edges:
                pair_to_trades.setdefault((s,b), []).append(tid)
            # try to find A->B, B->C, C->A triangles
            seen_cycles = 0
            for (a,b) in list(pair_to_trades.keys()):
                # neighbors from b
                for (bb,c) in [k for k in pair_to_trades.keys() if k[0] == b]:
                    if (c,a) in pair_to_trades:
                        seen_cycles += 1
                        # flag all trades participating in this cycle window
                        for tid in pair_to_trades[(a,b)] + pair_to_trades[(b,c)] + pair_to_trades[(c,a)]:
                            flags.append((tid, "WASH_CYCLE3", "3-node trade cycle within window"))
            if seen_cycles >= min_cycles:
                # we already appended flags; move window forward
                continue
    if not flags:
        return pd.DataFrame(columns=["entity_id","code","detail"])
    return pd.DataFrame(flags, columns=["entity_id","code","detail"]).drop_duplicates("entity_id")


def to_flags_df(over_under_df: pd.DataFrame, rapid_flip_df: pd.DataFrame):
    flags = pd.concat([over_under_df, rapid_flip_df], ignore_index=True)
    flags["entity_type"] = "trade"
    flags["risk"] = 0.85  # default; you can refine later
    return flags[["entity_type","entity_id","risk"]].join(
        pd.concat([over_under_df, rapid_flip_df], ignore_index=True)[["code","detail"]]
    )
