from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd

@dataclass
class MarketConfig:
    hours: int = 72
    # knobs to shape behavior
    list_rate_per_player_hour: float = 0.08     # how often players list
    buy_rate_per_player_hour: float = 0.10      # how often players buy
    price_noise_sigma: float = 0.12             # lognormal noise around base value
    trend_amp: float = 0.10                      # weekly-ish sinusoid amplitude
    regional_price_spread: float = 0.05         # regional deviations
    rng: np.random.Generator = None
    festival_on: bool = False
    festival_start: str = "2025-01-03 00:00:00"
    festival_hours: int = 24
    festival_amp: float = 0.25  # +25% smooth lift

def simulate_market(players: pd.DataFrame,
                    items: pd.DataFrame,
                    cfg: MarketConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = cfg.rng
    assert rng is not None, "Provide rng from world.gen_world for reproducibility."

    # Time axis (hourly ticks)
    ts = pd.date_range("2025-01-01", periods=cfg.hours, freq="h")

    # Precompute item baseline price paths (base_value * trend * noise)
    item_paths = _price_paths(items, ts, cfg, rng)  # shape: (n_items, len(ts))

    listings = []
    trades = []
    listing_id = 0
    trade_id = 0

    # Simple inventory budget so sellers don’t list infinite quantities
    inv = _initial_inventories(players, items, rng)

    # Pre-sample listing & buying intents for speed
    player_list_mask = rng.random((len(players), len(ts))) < cfg.list_rate_per_player_hour
    player_buy_mask  = rng.random((len(players), len(ts))) < cfg.buy_rate_per_player_hour

    # Region price multipliers (small)
    region_adj = {r: rng.normal(1.0, cfg.regional_price_spread) for r in players["region"].unique()}

    for t_idx, t in enumerate(ts):
        # 1) create listings
        seller_idx = np.where(player_list_mask[:, t_idx])[0]
        for s_i in seller_idx:
            seller = players.iloc[s_i]
            # pick an item the seller plausibly has
            item_id = _pick_list_item(inv, seller["player_id"], items, rng)
            if item_id is None:
                continue
            qty = int(np.clip(rng.poisson(1) + 1, 1, 5))  # small lots
            if inv[(seller["player_id"], item_id)] < qty:
                qty = inv[(seller["player_id"], item_id)]
            if qty <= 0:
                continue

            price_base = item_paths.loc[item_id, t]
            region_factor = region_adj[seller["region"]]
            human_variation = rng.lognormal(mean=0, sigma=cfg.price_noise_sigma)
            price = max(1.0, price_base * region_factor * human_variation)

            listing_id += 1
            listings.append({
                "listing_id": f"L{listing_id:07d}",
                "ts": t,
                "seller_id": seller["player_id"],
                "region": seller["region"],
                "item_id": item_id,
                "qty": qty,
                "list_price": round(price, 2),
                "expires_at": t + pd.Timedelta(hours=int(12 + 24*rng.random()))
            })

            # Reserve inventory
            inv[(seller["player_id"], item_id)] -= qty

        # 2) match buyers to best-price listings (very simplified order book)
        # Build order book snapshot
        if listings:
            book = pd.DataFrame(listings)
            open_book = book[(book["ts"] <= t) & (book["expires_at"] > t) & (book["qty"] > 0)]
        else:
            open_book = pd.DataFrame(columns=["listing_id","item_id","list_price","qty","seller_id","region","ts","expires_at"])

        buyer_idx = np.where(player_buy_mask[:, t_idx])[0]
        rng.shuffle(buyer_idx)
        for b_i in buyer_idx:
            buyer = players.iloc[b_i]
            # choose an item to buy with preference to rarer items for raiders/traders
            item_id = _pick_buy_item(buyer["playstyle"], items, rng)
            ob = open_book[open_book["item_id"] == item_id]
            if ob.empty:
                continue
            # choose lowest price (simple)
            pick = ob.sort_values("list_price", ascending=True).iloc[0]
            qty = 1  # keep it simple
            price = float(pick["list_price"])

            trade_id += 1
            trades.append({
                "trade_id": f"T{trade_id:07d}",
                "ts": t,
                "buyer_id": buyer["player_id"],
                "seller_id": pick["seller_id"],
                "item_id": item_id,
                "qty": qty,
                "price": round(price, 2),
                "listing_id": pick["listing_id"]
            })

            # decrement listing qty
            open_book_idx = open_book.index[open_book["listing_id"] == pick["listing_id"]][0]
            open_book.at[open_book_idx, "qty"] -= qty

    listings_df = pd.DataFrame(listings).sort_values("ts").reset_index(drop=True)
    trades_df   = pd.DataFrame(trades).sort_values("ts").reset_index(drop=True)
    return listings_df, trades_df

def _price_paths(items, ts, cfg, rng):
    n_items = len(items)
    t = np.arange(len(ts))
    seasonal = 1.0 + cfg.trend_amp * np.sin(2 * np.pi * t / (24*7))

    # NEW: smooth “festival” bump
    fest = np.ones_like(t, dtype=float)
    if cfg.festival_on:
        fs = pd.to_datetime(cfg.festival_start)
        fe = fs + pd.Timedelta(hours=cfg.festival_hours)
        mask = (ts >= fs) & (ts <= fe)
        # cosine ramp up/down for smoothness
        x = np.linspace(-np.pi, np.pi, mask.sum())
        fest[mask] = 1.0 + (cfg.festival_amp * (0.5*(1+np.cos(x))))  # bell-shaped

    item_jitter = rng.normal(1.0, 0.05, size=n_items)
    paths = []
    for i, row in items.iterrows():
        base = row["base_value"] * item_jitter[i]
        noise = rng.lognormal(mean=0, sigma=0.05, size=len(ts))
        series = base * seasonal * fest * noise
        paths.append(series)
    return pd.DataFrame(np.vstack(paths), index=items["item_id"], columns=ts)


def _initial_inventories(players: pd.DataFrame, items: pd.DataFrame, rng: np.random.Generator):
    inv = {}
    for p in players["player_id"]:
        # more items for older accounts
        inv_size = int(rng.integers(5, 30))
        owned_items = rng.choice(items["item_id"].values, size=inv_size, replace=True,
                                 p=_item_ownership_probs(items))
        for it in owned_items:
            inv[(p, it)] = inv.get((p, it), 0) + int(1 + rng.poisson(1))
    return inv

def _item_ownership_probs(items: pd.DataFrame):
    # More commons than rares, more rares than legendaries
    rarity_weight = items["rarity"].map({"common": 1.0, "rare": 0.35, "legendary": 0.08}).values
    w = rarity_weight / rarity_weight.sum()
    return w

def _pick_list_item(inv, player_id, items: pd.DataFrame, rng: np.random.Generator):
    candidates = [key[1] for key in inv.keys() if key[0] == player_id and inv[key] > 0]
    if not candidates:
        return None
    return rng.choice(candidates)

def _pick_buy_item(playstyle: str, items: pd.DataFrame, rng: np.random.Generator):
    weights = items["rarity"].map({
        "common": 1.0 if playstyle in ("casual","farmer") else 0.8,
        "rare": 0.6 if playstyle in ("casual","farmer") else 1.0,
        "legendary": 0.15 if playstyle in ("casual","farmer") else 0.4
    }).values.astype(float)
    weights = weights / weights.sum()
    return rng.choice(items["item_id"].values, p=weights)
