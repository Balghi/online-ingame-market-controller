import os
import pandas as pd
import pytest

RUN_DIR = os.environ.get("RUN_DIR", "data/runs/fest_on_seed1337")

@pytest.mark.skipif(not os.path.exists(RUN_DIR), reason="Run dir not found; generate artifacts first (python notebook.py).")
def test_basic_data_properties():
    trades = pd.read_csv(os.path.join(RUN_DIR, "trades.csv"), parse_dates=["ts"])
    listings = pd.read_csv(os.path.join(RUN_DIR, "listings.csv"), parse_dates=["ts", "expires_at"])

    # Non-negative
    assert (trades["price"] >= 0).all()
    assert (trades["qty"] >= 0).all()
    assert (listings["list_price"] >= 0).all()
    assert (listings["qty"] >= 0).all()

    # Orphan checks: every trade references a listing
    assert trades["listing_id"].isin(listings["listing_id"]).all(), "Found trades referencing missing listings."

    # Time sanity: listing ts <= trade ts <= expires_at (when expires_at exists)
    merged = trades.merge(listings[["listing_id","ts","expires_at"]].rename(columns={"ts":"list_ts"}), on="listing_id", how="left")
    assert (merged["list_ts"] <= merged["ts"]).all()
    assert (merged["expires_at"].isna() | (merged["ts"] <= merged["expires_at"])).all()
