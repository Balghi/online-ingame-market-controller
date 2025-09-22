# sim/fraud.py
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

@dataclass
class FraudConfig:
    seed: int = 42
    wash_groups: int = 6
    wash_group_size: int = 3
    wash_cycles_per_group: int = 8
    wash_overprice_factor: float = 2.2
    mule_pairs: int = 12
    mule_underprice_factor: float = 0.25

def inject_fraud(players: pd.DataFrame,
                 items: pd.DataFrame,
                 listings: pd.DataFrame,
                 trades: pd.DataFrame,
                 cfg: FraudConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns: (listings2, trades2, labels)
    - Appends fraudulent listings/trades for wash trading + mule transfers.
    - labels: entity_type, entity_id, fraud_type, is_fraud (1)
    """
    rng = np.random.default_rng(cfg.seed)

    listings2 = listings.copy()
    trades2 = trades.copy()
    labels = []

    # --- helper: get typical price per item (rolling or overall median) ---
    ref_price = trades2.groupby("item_id")["price"].median().to_dict()

    # --- 1) Wash trading: small circles repeatedly trade same item at inflated price ---
    # pick groups
    traders = players[players["playstyle"].isin(["trader", "raider"])].sample(
        n=min(len(players), cfg.wash_groups * cfg.wash_group_size),
        random_state=cfg.seed
    )["player_id"].tolist()
    wash_groups: List[List[str]] = [
        traders[i:i+cfg.wash_group_size] for i in range(0, len(traders), cfg.wash_group_size)
    ][:cfg.wash_groups]

    # choose items with reasonable liquidity (common/rare)
    liquid_items = items[items["rarity"] != "legendary"]["item_id"].tolist()
    base_ts = pd.to_datetime(listings2["ts"].min()) if not listings2.empty else pd.Timestamp("2025-01-01 00:00:00")
    next_list_id = _next_numeric_id(listings2, "listing_id", "L")
    next_trade_id = _next_numeric_id(trades2, "trade_id", "T")

    for g in wash_groups:
        if not liquid_items or len(g) < 2: 
            continue
        it = rng.choice(liquid_items)
        base = ref_price.get(it, float(items.set_index("item_id").loc[it, "base_value"]))
        inflated = round(max(1.0, base * cfg.wash_overprice_factor), 2)

        # cycle trades A->B->C->A...
        order = list(g)
        for k in range(cfg.wash_cycles_per_group):
            seller = order[k % len(order)]
            buyer  = order[(k + 1) % len(order)]
            t = base_ts + pd.Timedelta(minutes=int(rng.integers(30, 90))) + pd.Timedelta(minutes=5*k)

            lid = f"L{next_list_id:07d}"; next_list_id += 1
            listings2.loc[len(listings2)] = {
                "listing_id": lid, "ts": t, "seller_id": seller, "region": None,
                "item_id": it, "qty": 1, "list_price": inflated, "expires_at": t + pd.Timedelta(hours=6)
            }
            tid = f"T{next_trade_id:07d}"; next_trade_id += 1
            trades2.loc[len(trades2)] = {
                "trade_id": tid, "ts": t + pd.Timedelta(minutes=1), "buyer_id": buyer,
                "seller_id": seller, "item_id": it, "qty": 1, "price": inflated, "listing_id": lid
            }
            labels.append(("trade", tid, "wash_trading", 1))
            labels.append(("listing", lid, "wash_trading", 1))

    # --- 2) Mule transfers: high-value to new accounts at absurdly low price ---
    # pick new-ish accounts & random counterparts
    new_accounts = players.sort_values("account_age_days").head(max(10, cfg.mule_pairs*2))["player_id"].tolist()
    sellers = players["player_id"].sample(n=min(len(players), cfg.mule_pairs), random_state=cfg.seed+1).tolist()

    hi_value_items = items.sort_values("base_value", ascending=False)["item_id"].head(10).tolist() or items["item_id"].tolist()

    for i in range(min(cfg.mule_pairs, len(sellers), len(new_accounts))):
        seller = sellers[i]
        buyer  = new_accounts[i]
        it = rng.choice(hi_value_items)
        base = ref_price.get(it, float(items.set_index("item_id").loc[it, "base_value"]))
        cheap = round(max(1.0, base * cfg.mule_underprice_factor), 2)
        t = base_ts + pd.Timedelta(hours=int(rng.integers(6, 48))) + pd.Timedelta(minutes=int(rng.integers(0, 59)))

        lid = f"L{next_list_id:07d}"; next_list_id += 1
        listings2.loc[len(listings2)] = {
            "listing_id": lid, "ts": t, "seller_id": seller, "region": None,
            "item_id": it, "qty": 1, "list_price": cheap, "expires_at": t + pd.Timedelta(hours=6)
        }
        tid = f"T{next_trade_id:07d}"; next_trade_id += 1
        trades2.loc[len(trades2)] = {
            "trade_id": tid, "ts": t + pd.Timedelta(minutes=2), "buyer_id": buyer,
            "seller_id": seller, "item_id": it, "qty": 1, "price": cheap, "listing_id": lid
        }
        labels.append(("trade", tid, "mule_transfer", 1))
        labels.append(("listing", lid, "mule_transfer", 1))
        labels.append(("account", buyer, "mule_transfer", 1))

    labels_df = pd.DataFrame(labels, columns=["entity_type","entity_id","fraud_type","is_fraud"])
    listings2 = listings2.sort_values("ts").reset_index(drop=True)
    trades2   = trades2.sort_values("ts").reset_index(drop=True)
    return listings2, trades2, labels_df

def inject_busy_traders_nonfraud(players, items, listings, trades, cfg, n_clusters=4, group_size=3, window_hours=3):
    """Create concentrated-but-fair trading to challenge wash heuristics. No labels (non-fraud)."""
    rng = np.random.default_rng(cfg.seed + 404)
    listings2, trades2 = listings.copy(), trades.copy()
    base_ts = pd.to_datetime(listings2["ts"].min()) if not listings2.empty else pd.Timestamp("2025-01-01 00:00:00")
    next_list_id = _next_numeric_id(listings2, "listing_id", "L")
    next_trade_id = _next_numeric_id(trades2, "trade_id", "T")

    # item median as fair price
    med = trades2.groupby("item_id")["price"].median()
    pool = items[items["rarity"]!="legendary"]["item_id"].tolist() or items["item_id"].tolist()

    picks = players["player_id"].sample(n=min(len(players), n_clusters*group_size), random_state=cfg.seed+405).tolist()
    clusters = [picks[i:i+group_size] for i in range(0, len(picks), group_size)][:n_clusters]

    for c in clusters:
        if len(c) < 2: continue
        it = rng.choice(pool)
        fair = float(med.get(it, items.set_index("item_id").loc[it, "base_value"]))
        # within ±5% of fair
        def jitter_price():
            return round(fair * float(rng.normal(1.0, 0.03)), 2)

        t0 = base_ts + pd.Timedelta(hours=int(rng.integers(8, 48)))
        for k in range(int(6 + rng.integers(5))):  # ~6–10 trades
            a, b = rng.choice(c, size=2, replace=False)
            t = t0 + pd.Timedelta(minutes=int(rng.integers(0, window_hours*60)))
            lid = f"L{next_list_id:07d}"; next_list_id += 1
            listings2.loc[len(listings2)] = {"listing_id": lid, "ts": t, "seller_id": a, "region": None,
                                             "item_id": it, "qty": 1, "list_price": jitter_price(),
                                             "expires_at": t + pd.Timedelta(hours=6)}
            tid = f"T{next_trade_id:07d}"; next_trade_id += 1
            trades2.loc[len(trades2)] = {"trade_id": tid, "ts": t + pd.Timedelta(minutes=1),
                                         "buyer_id": b, "seller_id": a, "item_id": it, "qty": 1,
                                         "price": listings2.iloc[-1]["list_price"], "listing_id": lid}
    return listings2, trades2

def _next_numeric_id(df: pd.DataFrame, col: str, prefix: str) -> int:
    if df.empty: 
        return 1
    s = df[col].astype(str).str.replace(prefix, "", regex=False).astype(int)
    return int(s.max()) + 1


@dataclass
class FlipConfig:
    pairs: int = 10
    min_markup: float = 0.40       # 40% markup between buy and relist
    flip_delay_minutes: int = 5    # relist quickly
    underprice_factor: float = 0.7 # buy price relative to median to make the flip profitable

def inject_collusive_flips(players: pd.DataFrame,
                           items: pd.DataFrame,
                           listings: pd.DataFrame,
                           trades: pd.DataFrame,
                           cfg: FraudConfig,
                           flipcfg: FlipConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates pairs (A->B cheap), then B relists same item within minutes at higher price.
    We directly emit the second sale to ensure the pattern exists in data.
    Returns (listings3, trades3, labels_append)
    """
    rng = np.random.default_rng(cfg.seed + 202)
    listings3 = listings.copy()
    trades3   = trades.copy()
    labels = []

    base_ts = pd.to_datetime(listings3["ts"].min()) if not listings3.empty else pd.Timestamp("2025-01-01 00:00:00")
    next_list_id = _next_numeric_id(listings3, "listing_id", "L")
    next_trade_id = _next_numeric_id(trades3, "trade_id", "T")

    # candidate accounts & items
    A = players["player_id"].sample(n=min(len(players), flipcfg.pairs*2), random_state=cfg.seed+7).tolist()
    if len(A) < 2: 
        return listings3, trades3, pd.DataFrame(columns=["entity_type","entity_id","fraud_type","is_fraud"])
    pairs = [(A[i], A[i+1]) for i in range(0, len(A)-1, 2)]
    items_pool = items[items["rarity"] != "legendary"]["item_id"].tolist() or items["item_id"].tolist()

    # item median price baseline
    med = trades3.groupby("item_id")["price"].median()
    if med.empty:
        # fallback to base_value
        med = items.set_index("item_id")["base_value"]

    for i, (seller_A, buyer_B) in enumerate(pairs[:flipcfg.pairs]):
        it = rng.choice(items_pool)
        m = float(med.get(it, items.set_index("item_id").loc[it, "base_value"]))
        cheap = round(max(1.0, m * flipcfg.underprice_factor), 2)
        t0 = base_ts + pd.Timedelta(hours=int(rng.integers(8, 72))) + pd.Timedelta(minutes=int(rng.integers(0, 59)))

        # A lists cheap to B (trade 1)
        lid1 = f"L{next_list_id:07d}"; next_list_id += 1
        listings3.loc[len(listings3)] = {
            "listing_id": lid1, "ts": t0, "seller_id": seller_A, "region": None,
            "item_id": it, "qty": 1, "list_price": cheap, "expires_at": t0 + pd.Timedelta(hours=6)
        }
        tid1 = f"T{next_trade_id:07d}"; next_trade_id += 1
        trades3.loc[len(trades3)] = {
            "trade_id": tid1, "ts": t0 + pd.Timedelta(minutes=1), "buyer_id": buyer_B,
            "seller_id": seller_A, "item_id": it, "qty": 1, "price": cheap, "listing_id": lid1
        }
        labels.append(("trade", tid1, "collusive_flip", 1))
        labels.append(("listing", lid1, "collusive_flip", 1))

        # B relists within minutes at markup (trade 2 to a random buyer C)
        markup_price = round(max(cheap * (1.0 + flipcfg.min_markup), m * 0.95), 2)  # profitable but still sellable
        delay = int(rng.choice([5,7,12,18,35,40], p=[.25,.25,.2,.15,.1,.05]))
        t1 = t0 + pd.Timedelta(minutes=delay)

        lid2 = f"L{next_list_id:07d}"; next_list_id += 1
        listings3.loc[len(listings3)] = {
            "listing_id": lid2, "ts": t1, "seller_id": buyer_B, "region": None,
            "item_id": it, "qty": 1, "list_price": markup_price, "expires_at": t1 + pd.Timedelta(hours=6)
        }
        buyer_C = players["player_id"].sample(n=1, random_state=cfg.seed + 999 + i).iloc[0]
        tid2 = f"T{next_trade_id:07d}"; next_trade_id += 1
        trades3.loc[len(trades3)] = {
            "trade_id": tid2, "ts": t1 + pd.Timedelta(minutes=2), "buyer_id": buyer_C,
            "seller_id": buyer_B, "item_id": it, "qty": 1, "price": markup_price, "listing_id": lid2
        }
        labels.append(("trade", tid2, "collusive_flip", 1))
        labels.append(("listing", lid2, "collusive_flip", 1))

    labs = pd.DataFrame(labels, columns=["entity_type","entity_id","fraud_type","is_fraud"])
    listings3 = listings3.sort_values("ts").reset_index(drop=True)
    trades3   = trades3.sort_values("ts").reset_index(drop=True)
    return listings3, trades3, labs
