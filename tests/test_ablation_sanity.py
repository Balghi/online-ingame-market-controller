import os
import pandas as pd
import pytest

RUN_DIR = os.environ.get("RUN_DIR", "data/runs/fest_on_seed1337")

@pytest.mark.skipif(not os.path.exists(RUN_DIR), reason="Run dir not found; generate artifacts first (python notebook.py).")
def test_ablation_sanity():
    path = os.path.join(RUN_DIR, "ablation_unique_tp.csv")
    assert os.path.exists(path), "ablation_unique_tp.csv missing; run notebook to generate."
    df = pd.read_csv(path)
    assert ((df["unique_tp"] >= 0) & (df["unique_tp"] <= df["tp"])).all(), "unique_tp must be between 0 and tp."
