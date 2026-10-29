"""scripts/popout_relabel.py Front-input visibility geometry (popout audit 10/29): a person-sized box 10 m straight ahead
projects to the image centre column, ~40 px tall on the 448x256 input (fy 227.6 x 1.7 m / 9.65 m); a parked car 5 m ahead
on the same line hides it fully; 6 m to the side (rays clear the car) it is in view again. No data files, CPU."""
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")                      # the camera onset's mask (train venvs without it skip)
pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import popout_relabel as pr  # noqa: E402

CAM = (0.0, 0.0, 0.0)                                   # camera at the origin looking +x


def px(x, y, occluders=()):
    uv, dep = pr._project(pr._box(x, y, 0.0, 0.7, 0.7, 1.7), CAM)
    m = pr._mask(uv)
    for o in occluders:
        m &= ~pr._mask(pr._project(pr._box(*o), CAM)[0])
    rows = np.flatnonzero(m.any(1))
    return int(m.sum()), int(rows[-1] - rows[0] + 1) if len(rows) else 0, uv


def test_ahead_visible_and_centred():
    n, h, uv = px(10.0, 0.0)
    assert n >= 30 and 38 <= h <= 42
    assert abs(uv[:, 0].mean() - pr.W / 2) < 1.0


def test_parked_car_hides_and_side_reveals():
    car = (5.0, 0.0, 0.0, 4.6, 1.9, 1.5)
    assert px(10.0, 0.0, [car])[0] == 0
    n, h, _ = px(10.0, 6.0, [car])
    assert n >= 30 and h >= 8
