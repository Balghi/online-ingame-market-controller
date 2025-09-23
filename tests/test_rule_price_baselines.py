import pandas as pd
import numpy as np

from detect.rules import (
    rolling_item_baselines,
    flag_under_overpriced,
)

def _mk_trades_series():
  
    ts0 = pd.Timestamp("2025-01-01 00:00:00")
    rows = []
    # Stable block around 100
    for i in range(24):
        rows.append({
            "trade_id": f"T{i:03d}",
            "ts": ts0 + pd.Timedelta(hours=i),
            "buyer_id": f"B{i%5}",
            "seller_id": f"S{i%5}",
            "item_id": "I1",
            "qty": 1,
            "price": 100,
            "listing_id": f"L{i:03d}",
        })
    rows.append({
        "trade_id": "T_UNDER",
        "ts": ts0 + pd.Timedelta(hours=30),
        "buyer_id": "B9",
        "seller_id": "S9",
        "item_id": "I1",
        "qty": 1,
        "price": 50,
        "listing_id": "L_UNDER",
    })
    rows.append({
        "trade_id": "T_OVER",
        "ts": ts0 + pd.Timedelta(hours=31),
        "buyer_id": "B8",
        "seller_id": "S8",
        "item_id": "I1",
        "qty": 1,
        "price": 160,
        "listing_id": "L_OVER",
    })
    rows.append({
        "trade_id": "T_OK",
        "ts": ts0 + pd.Timedelta(hours=32),
        "buyer_id": "B7",
        "seller_id": "S7",
        "item_id": "I1",
        "qty": 1,
        "price": 105,
        "listing_id": "L_OK",
    })
    return pd.DataFrame(rows)


def _as_df(flags_like):
    
    if isinstance(flags_like, pd.DataFrame):
        df = flags_like.copy()
        if "trade_id" in df.columns and "entity_id" not in df.columns:
            df = df.rename(columns={"trade_id": "entity_id"})
        return df
    
    try:
        cols = ["entity_id", "code", "detail"]
        df = pd.DataFrame(list(flags_like), columns=cols[:len(list(flags_like)[0])])
        if "detail" not in df.columns:
            df["detail"] = ""
        return df
    except Exception:
        
        return pd.DataFrame(flags_like)


def test_under_and_overpriced_are_flagged():
    trades = _mk_trades_series()

    
    tb = rolling_item_baselines(trades)

    
    flags_raw = flag_under_overpriced(tb, k=2.0)
    flags = _as_df(flags_raw)

    codes_by_id = {str(r["entity_id"]): r["code"] for _, r in flags.iterrows()}

    assert "T_UNDER" in codes_by_id, "Expected the ~50% price trade to be flagged"
    assert codes_by_id["T_UNDER"] == "UNDERPRICED", f"Expected UNDERPRICED, got {codes_by_id['T_UNDER']}"
    assert "T_OVER" in codes_by_id, "Expected the ~160% price trade to be flagged"
    assert codes_by_id["T_OVER"] == "OVERPRICED", f"Expected OVERPRICED, got {codes_by_id['T_OVER']}"


def test_near_median_trade_not_flagged():
    trades = _mk_trades_series()
    tb = rolling_item_baselines(trades)

    flags_raw = flag_under_overpriced(tb, k=2.0)
    flags = _as_df(flags_raw)

    if flags.empty:
        
        return

    flagged_ids = set(flags["entity_id"].astype(str))
    assert "T_OK" not in flagged_ids, "Near-median price should not be flagged as under/overpriced"
