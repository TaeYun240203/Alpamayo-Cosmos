"""scripts/ft_pools.moving_props_only (user 10/28): drop an episode only when every moving object is a prop (4xxxxxxx)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ft_pools import moving_props_only  # noqa: E402


def test_moving_props_only():
    meta = pd.DataFrame({"episode": ["props", "ped+prop", "cyclist", "static", "none"],
                         "actor_ids": [[40100008, 40200011], [40100010, 50100001], [50100084], [40100002], []],
                         "actor_kph": [[12.0, 5.0], [15.0, 4.0], [14.0], [0.0], []]})
    assert moving_props_only(meta) == {"props"}                                  # a parked or tunnel static prop does not count
