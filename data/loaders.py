import pandas as pd
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "raw"


def load_synthetic(path: Optional[str] = None) -> pd.DataFrame:
    """Load synthetic transaction dataset."""
    if path is None:
        path = DATA_DIR / "synthetic_transactions.csv"
    return pd.read_csv(path, parse_dates=["timestamp"])


def load_ieee_cis(path: str) -> pd.DataFrame:
    """Load IEEE-CIS Fraud Detection dataset (Kaggle).
    Placeholder — implement when dataset is downloaded.
    """
    raise NotImplementedError("Download IEEE-CIS dataset from Kaggle first")


def load_paysim(path: str) -> pd.DataFrame:
    """Load PaySim dataset.
    Placeholder — implement when dataset is downloaded.
    """
    raise NotImplementedError("Download PaySim dataset first")


def load_red_team_feed(endpoint_url: str) -> pd.DataFrame:
    """Load transactions from Red Team API feed.
    Placeholder — implement during integration.
    """
    raise NotImplementedError("Red Team integration not configured yet")


def load_dataset(source: str = "synthetic", **kwargs) -> pd.DataFrame:
    """Unified data loader. Source: 'synthetic', 'ieee_cis', 'paysim', 'red_team'."""
    loaders = {
        "synthetic": load_synthetic,
        "ieee_cis": load_ieee_cis,
        "paysim": load_paysim,
        "red_team": load_red_team_feed,
    }
    if source not in loaders:
        raise ValueError(f"Unknown source '{source}'. Available: {list(loaders.keys())}")
    return loaders[source](**kwargs)
