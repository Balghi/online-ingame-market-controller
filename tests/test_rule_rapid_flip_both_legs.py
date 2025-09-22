import pandas as pd
import numpy as np
import pytest

from detect.rules import flag_rapid_flip

def _ts(h, m=0):
    return pd.Timestamp('2025-01-01') + pd.Timedelta(hours=h, minutes=m)

def test_rapid_flip_flags_both_legs():
    # Synthetic listing and two trades for same item: A sells to B, B sells to C within 12 min at 40% markup
    listings = pd.DataFrame([
        {"listing_id":"L1","ts":_ts(10,0),"seller_id":"A","region":"R","item_id":"I1","qty":1,"list_price":100,"expires_at":_ts(16)},
        {"listing_id":"L2","ts":_ts(10,10),"seller_id":"B","region":"R","item_id":"I1","qty":1,"list_price":140,"expires_at":_ts(16)},
    ])
    trades = pd.DataFrame([
        {"trade_id":"T1","ts":_ts(10,1),"buyer_id":"B","seller_id":"A","item_id":"I1","qty":1,"price":100,"listing_id":"L1"},
        {"trade_id":"T2","ts":_ts(10,12),"buyer_id":"C","seller_id":"B","item_id":"I1","qty":1,"price":140,"listing_id":"L2"},
    ])
    flags = flag_rapid_flip(listings, trades, window_minutes=25, min_markup=0.25, adaptive=False)
    got = set(flags["entity_id"].astype(str).tolist())
    assert {"T1","T2"}.issubset(got), f"Expected both legs flagged, got {got}"
