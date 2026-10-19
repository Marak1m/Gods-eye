"""M0: configuration and labeling-guide validation (HANDOFF.md section 9, `test_config`)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scd.config import REPO_ROOT, ConfigError, load, load_strata
from scd.constants import BAND_ORDER, STRATA
from scd.labels import conventions

CONFIGS = REPO_ROOT / "configs"
DEFAULT = CONFIGS / "default.yaml"


def test_default_loads_and_is_stable(cfg) -> None:
    assert cfg.bands.order == BAND_ORDER
    assert cfg.classes.ids == {"none": 0, "buildup": 1, "building": 2, "road": 3, "solar": 4}
    assert cfg.infer.tta_temporal is False  # DEVIATIONS U2
    assert cfg.infer.invalid_value == 0  # DEVIATIONS U3
    assert cfg.hash() == load(DEFAULT).hash()
    assert cfg.labeling_guide_path() == REPO_ROOT / "labeling_guide.yaml"
    assert cfg.runs_dir == cfg.data_root / "runs"


def test_missing_env_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SCD_DATA", raising=False)
    monkeypatch.delenv("SCD_RUNS", raising=False)
    with pytest.raises(ConfigError, match="SCD_DATA"):
        load(DEFAULT)


def test_dotlist_overrides(data_root: Path) -> None:
    c = load(DEFAULT, ["train.epochs=2", "sr.method=bicubic", "train.precision=32"])
    assert c.train.epochs == 2
    assert c.sr.method == "bicubic"
    assert c.train.precision == "32"
    assert c.hash() != load(DEFAULT).hash()


def test_unknown_key_rejected(data_root: Path) -> None:
    with pytest.raises(ConfigError):
        load(DEFAULT, ["train.epochz=2"])


def test_invalid_band_order_rejected(data_root: Path) -> None:
    with pytest.raises(ConfigError, match="bands.order"):
        load(DEFAULT, ["bands.order=[B04,B03,B02,B08,B11,B12]"])


@pytest.mark.parametrize(
    "override",
    [
        "strata.shares.A=0.5",
        "sampler.class_shares.solar=0.5",
        "sampler.tile_type_shares.default.STAR=0.5",
        "classes.rubric_weights.solar=0.2",
        "finetune.select_weights.targets=0.9",
    ],
)
def test_shares_must_sum_to_one(data_root: Path, override: str) -> None:
    with pytest.raises(ConfigError, match="shares sum"):
        load(DEFAULT, [override])


def test_enum_values_rejected(data_root: Path) -> None:
    with pytest.raises(ConfigError):
        load(DEFAULT, ["model.fusion=concat"])
    with pytest.raises(ConfigError):
        load(DEFAULT, ["infer.invalid_value=7"])


def test_base_inheritance_for_every_config_file(data_root: Path) -> None:
    files = sorted((CONFIGS / "ablations").glob("*.yaml")) + [
        CONFIGS / "smoke.yaml",
        CONFIGS / "finetune.yaml",
    ]
    assert len(files) >= 21
    for f in files:
        c = load(f)
        assert c.bands.order == BAND_ORDER, f
    smoke = load(CONFIGS / "smoke.yaml")
    assert smoke.grid.chip_px == 256 and smoke.sr.method == "bicubic" and smoke.cloud.method == "scl"
    assert smoke.train.max_steps == 2 and smoke.train.precision == "32"
    assert smoke.labeling_guide_path().name == "labeling_guide_smoke.yaml"
    assert load(CONFIGS / "ablations" / "A1_bicubic.yaml").sr.method == "bicubic"
    assert load(CONFIGS / "ablations" / "A3_2p5m.yaml").grid.res_m == 2.5
    assert load(CONFIGS / "ablations" / "A5_softmax5.yaml").model.head_mode == "softmax5"
    assert load(CONFIGS / "ablations" / "A7_synth30.yaml").sampler.shares_for("A")["SYNTHETIC"] == 0.30


def test_labeling_guide_with_fill_is_rejected() -> None:
    with pytest.raises(conventions.ConventionsError, match="fill"):
        conventions.load(REPO_ROOT / "labeling_guide.yaml")


def test_smoke_labeling_guide_loads(conventions) -> None:  # noqa: F811 - fixture shadows module
    conv = conventions
    assert conv.priority == ("building", "solar", "road", "buildup")
    assert conv.priority_ids() == [2, 4, 3, 1]
    assert conv.burn_order() == [1, 3, 4, 2]
    assert conv.min_area_m2 == {"buildup": 2500.0, "building": 1800.0, "solar": 5000.0}
    assert conv.min_length_m == {"road": 200.0}
    assert conv.removals_labeled is True


def test_strata_yaml() -> None:
    s = load_strata()
    assert set(s) == set(STRATA)
    assert s["E"]["allowed_months"] == [11, 12, 1, 2, 3, 4]
    for v in s.values():
        assert len(v["contrast_seasons"]) == 2


def test_dump_roundtrip(cfg, tmp_path: Path) -> None:
    out = tmp_path / "resolved.yaml"
    cfg.dump(out)
    assert "B02" in out.read_text()
