import os
import json
import pandas as pd
import pytest

RUN_DIR = os.environ.get("RUN_DIR", "data/runs/fest_on_seed1337")
SNAPSHOT = os.environ.get("GOLDEN_SNAPSHOT", "tests/golden_flags_trade_ids.json")

@pytest.mark.skipif(not os.path.exists(RUN_DIR), reason="Run dir not found; generate artifacts first (python notebook.py).")
def test_golden_flags_ids_match():
    flags_path = os.path.join(RUN_DIR, "flags.csv")
    assert os.path.exists(flags_path), "flags.csv missing; run the pipeline first."
    flags = pd.read_csv(flags_path)
    got = sorted(flags[flags.entity_type == "trade"]["entity_id"].astype(str).unique().tolist())

    if not os.path.exists(SNAPSHOT):
        pytest.skip(f"Snapshot {SNAPSHOT} not found; create it from current best run.")

    with open(SNAPSHOT, "r") as f:
        want = sorted(json.load(f))

    assert got == want, "Golden trade IDs changed—inspect diff; adjust thresholds or update snapshot intentionally."
