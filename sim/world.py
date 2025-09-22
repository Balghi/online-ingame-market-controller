from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd

@dataclass
class WorldConfig:
    n_players: int = 500
    n_items: int = 30
    seed: int = 42
    regions: Tuple[str, ...] = ("EU", "NA", "ASIA")
    playstyles: Tuple[str, ...] = ("casual", "raider", "farmer", "trader")

def gen_world(cfg: WorldConfig) -> Tuple[pd.DataFrame, pd.DataFrame, np.random.Generator]:
    rng = np.random.default_rng(cfg.seed)

    # Players
    player_ids = [f"P{idx:05d}" for idx in range(cfg.n_players)]
    regions = rng.choice(cfg.regions, size=cfg.n_players, p=_normalize([0.45, 0.35, 0.20]))
    playstyles = rng.choice(cfg.playstyles, size=cfg.n_players, p=_normalize([0.55, 0.15, 0.15, 0.15]))
    account_age_days = rng.integers(low=1, high=365*3, size=cfg.n_players)  # up to ~3 years

    players = pd.DataFrame({
        "player_id": player_ids,
        "region": regions,
        "playstyle": playstyles,
        "account_age_days": account_age_days
    })

    # Items
    item_ids = [f"I{idx:04d}" for idx in range(cfg.n_items)]
    rarity_probs = _normalize([0.70, 0.25, 0.05])  # common/rare/legendary
    rarity = rng.choice(["common", "rare", "legendary"], size=cfg.n_items, p=rarity_probs)
    base_value = _rarity_to_base_value(rng, rarity)  # e.g., common ~ 50, rare ~ 400, legendary ~ 3000
    drop_rate = _rarity_to_drop_rate(rng, rarity)    # lower for higher rarity

    items = pd.DataFrame({
        "item_id": item_ids,
        "rarity": rarity,
        "base_value": base_value,
        "drop_rate": drop_rate
    })

    return players, items, rng

def _rarity_to_base_value(rng: np.random.Generator, rarity: np.ndarray) -> np.ndarray:
    out = np.empty_like(rarity, dtype=float)
    for r in ("common", "rare", "legendary"):
        mask = rarity == r
        if r == "common":
            out[mask] = rng.normal(50, 10, mask.sum()).clip(10, 200)
        elif r == "rare":
            out[mask] = rng.normal(400, 80, mask.sum()).clip(80, 1000)
        else:
            out[mask] = rng.normal(3000, 600, mask.sum()).clip(800, 10000)
    return out.round(2)

def _rarity_to_drop_rate(rng: np.random.Generator, rarity: np.ndarray) -> np.ndarray:
    out = np.empty_like(rarity, dtype=float)
    for r in ("common", "rare", "legendary"):
        mask = rarity == r
        if r == "common":
            out[mask] = rng.uniform(0.02, 0.15, mask.sum())  # plentiful
        elif r == "rare":
            out[mask] = rng.uniform(0.005, 0.02, mask.sum())
        else:
            out[mask] = rng.uniform(0.0005, 0.003, mask.sum())  # very scarce
    return out
       
def _normalize(v):
    arr = np.array(v, dtype=float)
    return arr / arr.sum()
