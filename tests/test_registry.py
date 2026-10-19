"""M2: AOI registry schema and validators (HANDOFF 5.1)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scd.config import REPO_ROOT
from scd.imagery import registry


def test_targets_registry_roundtrip(tmp_path: Path) -> None:
    df = registry.init_from_targets(REPO_ROOT / "configs" / "targets.csv")
    assert len(df) == 42 and (df["role"] == "targets").all()
    assert registry.validate(df) == []
    assert df.loc[df.aoi_id == "A_target_01", "utm_epsg"].item() == 32645
    p = registry.save(df, tmp_path / "registry" / "aois.parquet")
    back = registry.load(p)
    assert list(back.columns) == list(registry.COLUMNS)
    assert back["geometry"].iloc[0].equals(df["geometry"].iloc[0])
    assert (tmp_path / "registry" / "aois.csv").exists()


def _row(aoi_id: str, stratum: str, role: str, lat: float, lon: float, target_type: str = "solar") -> dict:
    return {
        "aoi_id": aoi_id,
        "stratum": stratum,
        "target_type": target_type,
        "role": role,
        "centre_lat": lat,
        "centre_lon": lon,
        "size_km": 10.0,
        "lead_source": "gem",
        "lead_ref": "x",
        "event_year": 2021,
        "t1_window": "2019-05-01/2019-10-31",
        "t2_window": "2023-05-01/2023-10-31",
        "notes": "",
    }


def test_validate_catches_each_rule() -> None:
    rows = [
        _row("A_a_01", "A", "train", 30.0, 90.0),
        _row("A_a_02", "A", "val", 30.05, 90.0),  # 5.6 km from the first -> too close
        _row("B_b_01", "B", "train", 40.0, 100.0),
        _row("B_b_01", "B", "train", 41.0, 101.0),  # duplicate id
        _row("C_c_01", "Z", "train", 25.0, 70.0),  # bad stratum
        _row("D_d_01", "D", "tester", 32.0, 76.0),  # bad role
        _row("E_x_01", "E", "train", 18.2, 109.5),
        _row("E_target_07", "E", "targets", 18.22, 109.5),  # train AOI within 10 km of a target
        _row("F_f_01", "F", "train", 36.0, 76.0, target_type="weird"),
    ]
    df = pd.DataFrame(rows)
    errors = registry.validate(df)
    text = "\n".join(errors)
    assert "duplicate aoi_id" in text
    assert "stratum not in" in text and "role not in" in text and "target_type not in" in text
    assert "A_a_01 and A_a_02" in text
    assert "overlaps target" in text


def test_cluster_cap() -> None:
    rows = [_row(f"B_city_{i:02d}", "B", "train", 40.0 + 0.1 * i, 100.0) for i in range(10)]  # 10 AOIs ~11 km apart
    errors = registry.validate(pd.DataFrame(rows))
    assert any("cluster of 10" in e for e in errors)
    rows = [_row(f"B_far_{i:02d}", "B", "train", 40.0 + 0.5 * i, 100.0) for i in range(10)]  # 55 km apart
    assert registry.validate(pd.DataFrame(rows)) == []


def test_append_from_csv_and_strict_load(tmp_path: Path) -> None:
    df = registry.init_from_targets(REPO_ROOT / "configs" / "targets.csv")
    csv = tmp_path / "more.csv"
    pd.DataFrame([_row("A_new_01", "A", "train", 31.0, 95.0)]).to_csv(csv, index=False)
    df2 = registry.append_from_csv(df, csv)
    assert len(df2) == 43 and df2.iloc[-1]["geometry"].is_valid
    bad = pd.concat([df2, pd.DataFrame([_row("A_new_01", "A", "train", 31.0, 95.0)])], ignore_index=True)
    p = registry.save(bad, tmp_path / "aois.parquet")
    with pytest.raises(ValueError, match="duplicate"):
        registry.load(p)
