import os
import pandas as pd
import pytest

RUN_DIR = os.environ.get("RUN_DIR", "data/runs/fest_on_seed1337")

rules = pytest.importorskip("detect.rules")
flag_under_overpriced = getattr(rules, "flag_under_overpriced")
rolling_item_baselines = getattr(rules, "rolling_item_baselines")

@pytest.mark.skipif(not os.path.exists(RUN_DIR), reason="Run dir not found; generate artifacts first (python notebook.py).")
def test_under_overpriced_time_shift_invariance():
    trades = pd.read_csv(os.path.join(RUN_DIR, "trades.csv"), parse_dates=["ts"])

    # Baseline
    tb0 = rolling_item_baselines(trades)
    ou0 = flag_under_overpriced(tb0, k=2.5)
    ids0 = set(ou0["entity_id"].astype(str).tolist())

    # Shift times by +24h, recompute baselines & rule
    trades_shift = trades.copy()
    trades_shift["ts"] = trades_shift["ts"] + pd.Timedelta(hours=24)
    tb1 = rolling_item_baselines(trades_shift)
    ou1 = flag_under_overpriced(tb1, k=2.5)
    ids1 = set(ou1["entity_id"].astype(str).tolist())

    assert ids0 == ids1, "UNDER/OVER flags should be invariant to a uniform time shift."
