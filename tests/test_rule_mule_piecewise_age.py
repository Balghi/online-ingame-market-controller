import pandas as pd
import numpy as np
import pytest

from detect.rules import flag_mule_transfers

def test_mule_piecewise_age_logic():
    trades_hist = pd.DataFrame({
        "trade_id":[f"H{i}" for i in range(1,51)],
        "ts": pd.date_range("2025-01-01", periods=50, freq="h"),
        "buyer_id":["X"]*50,
        "seller_id":["Y"]*50,
        "item_id":["I1"]*50,
        "qty":[1]*50,
        "price":[100]*45 + [300,320,340,360,380],
        "listing_id":[f"LH{i}" for i in range(1,51)]
    })

    new_trade = pd.DataFrame([{
        "trade_id":"T1","ts":pd.Timestamp("2025-01-03 00:00:00"),
        "buyer_id":"B","seller_id":"S","item_id":"I1","qty":1,"price":50,"listing_id":"LX"
    }])

    trades = pd.concat([trades_hist, new_trade], ignore_index=True)

    listings = pd.DataFrame([
        {"listing_id":"LX","ts":pd.Timestamp("2025-01-03 00:00:00"),
         "seller_id":"S","region":"R","item_id":"I1","qty":1,"list_price":50,"expires_at":pd.Timestamp("2025-01-03 06:00:00")}
    ])

    players_new = pd.DataFrame([{"player_id":"B","account_age_days":10}])
    flags_new = flag_mule_transfers(trades, listings, players_new, price_pct=0.5, max_buyer_age_days=14,
                                    extended_age_days=30, high_value_multiplier=2.5)
    assert "T1" in set(flags_new["entity_id"]), "New buyer at or under 50% price threshold should be flagged"

    high_trade = new_trade.copy()
    high_trade.loc[0,"trade_id"] = "T2"
    high_trade.loc[0,"price"] = 50
    high_trade.loc[0,"listing_id"] = "LY"
    trades2 = pd.concat([trades_hist, high_trade], ignore_index=True)
    players_28 = pd.DataFrame([{"player_id":"B","account_age_days":28}])
    flags_28 = flag_mule_transfers(trades2, listings, players_28, price_pct=0.5, max_buyer_age_days=14,
                                   extended_age_days=30, high_value_multiplier=2.5)
    assert "T2" in set(flags_28["entity_id"]), "28d buyer at high value (>=p95) should trigger extended age rule"
