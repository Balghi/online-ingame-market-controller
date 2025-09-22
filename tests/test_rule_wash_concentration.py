import pandas as pd
import numpy as np
import pytest
from detect.rules import flag_wash_trading_concentration

def _mk_trades_ring(item="I1", base_ts="2025-01-01 00:00:00"):
    ts0 = pd.Timestamp(base_ts)
    rows = []
    # 3-player ring A->B, B->C, C->A repeated within 2h
    players = ["A","B","C"]
    listing_id = 1; trade_id = 1
    for k in range(3):
        seller = players[k]; buyer = players[(k+1)%3]
        t = ts0 + pd.Timedelta(minutes=30*k)
        lid = f"L{listing_id}"; listing_id += 1
        rows.append({"trade_id":f"T{trade_id}","ts":t,"buyer_id":buyer,"seller_id":seller,
                     "item_id":item,"qty":1,"price":100,"listing_id":lid})
        trade_id += 1
    return pd.DataFrame(rows)

def test_wash_concentration_detects_dense_cluster():
    trades = _mk_trades_ring()
    flags = flag_wash_trading_concentration(trades, window_minutes=240, max_group_size=3, min_trades=3, top_k_share=0.75)
    got = set(flags["entity_id"].astype(str).tolist())
    assert len(got) >= 1, "Expected at least one trade flagged in a tight 3-player ring"

def test_wash_concentration_spares_uniform_distribution():
    # uniformly distributed random counterparties: expect no flags under strict settings
    rng = np.random.default_rng(0)
    buyers = [f"B{i}" for i in range(20)]
    sellers= [f"S{i}" for i in range(20)]
    rows = []
    ts0 = pd.Timestamp("2025-01-01 00:00:00")
    for i in range(30):
        b = rng.choice(buyers); s = rng.choice(sellers)
        rows.append({"trade_id":f"U{i}","ts":ts0 + pd.Timedelta(minutes=10*i),
                     "buyer_id":b,"seller_id":s,"item_id":"I2","qty":1,"price":100,"listing_id":f"LU{i}"})
    trades = pd.DataFrame(rows)
    flags = flag_wash_trading_concentration(trades, window_minutes=120, max_group_size=3, min_trades=5, top_k_share=0.8)
    assert flags.empty or flags.shape[0] <= 2, "Uniform counterparties should not trigger many concentration flags"
