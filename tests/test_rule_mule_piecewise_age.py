import pandas as pd
import numpy as np
import pytest

from detect.rules import flag_mule_transfers

def test_mule_piecewise_age_logic():
    # item I1 has high p95 (simulated via trades)
    trades_hist = pd.DataFrame({
        "trade_id":[f"H{i}" for i in range(1,51)],
        "ts": pd.date_range("2025-01-01", periods=50, freq="H"),
        "buyer_id":["X"]*50,
        "seller_id":["Y"]*50,
        "item_id":["I1"]*50,
        "qty":[1]*50,
        "price":[100]*45 + [300,320,340,360,380],  # p95 ~ around 300+
        "listing_id":[f"LH{i}" for i in range(1,51)]
    })

    # target trade: price 60 (<=50% of ref ~100) from seller S to buyer B
    new_trade = pd.DataFrame([{
        "trade_id":"T1","ts":pd.Timestamp("2025-01-03 00:00:00"),
        "buyer_id":"B","seller_id":"S","item_id":"I1","qty":1,"price":60,"listing_id":"LX"
    }])

    trades = pd.concat([trades_hist, new_trade], ignore_index=True)

    listings = pd.DataFrame([
        {"listing_id":"LX","ts":pd.Timestamp("2025-01-03 00:00:00"),
         "seller_id":"S","region":"R","item_id":"I1","qty":1,"list_price":60,"expires_at":pd.Timestamp("2025-01-03 06:00:00")}
    ])

    # Case 1: very new buyer (10 days)
    players_new = pd.DataFrame([{"player_id":"B","account_age_days":10}])
    flags_new = flag_mule_transfers(trades, listings, players_new, price_pct=0.5, max_buyer_age_days=14,
                                    extended_age_days=30, high_value_quantile=0.95)
    assert "T1" in set(flags_new["entity_id"]), "New buyer under price threshold should be flagged"

    # Case 2: 28-day buyer but HIGH value (>= p95): craft a second trade with price >= p95
    high_trade = new_trade.copy(); high_trade.loc[0,"trade_id"]="T2"; high_trade.loc[0,"price"]=360
    trades2 = pd.concat([trades_hist, high_trade], ignore_index=True)
    players_28 = pd.DataFrame([{"player_id":"B","account_age_days":28}])
    flags_28 = flag_mule_transfers(trades2, listings, players_28, price_pct=0.5, max_buyer_age_days=14,
                                   extended_age_days=30, high_value_quantile=0.95)
    assert "T2" in set(flags_28["entity_id"]), "28d buyer at high value (>=p95) should trigger extended age rule"
