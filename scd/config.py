"""Configuration loading and validation (HANDOFF.md section 4).

`load(path, overrides)` reads a YAML file (following `_base_: other.yaml` inheritance), merges
`key=value` dot-list overrides, resolves `${oc.env:...}` interpolations and validates the result
into the frozen pydantic model `Config`. Every script and the notebook go through this function.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from scd.constants import BAND_ORDER, CHANGE_CLASSES, CLASS_IDS, STRATA, TILE_TYPES

REPO_ROOT = Path(__file__).resolve().parents[1]
_SHARE_TOL = 1e-6


class ConfigError(ValueError):
    """Raised for unreadable, unresolvable or invalid configuration."""


def _check_shares(name: str, shares: Mapping[str, float], keys: Sequence[str] | None) -> None:
    if any(v < 0 for v in shares.values()):
        raise ValueError(f"{name}: negative share in {dict(shares)}")
    total = float(sum(shares.values()))
    if abs(total - 1.0) > _SHARE_TOL:
        raise ValueError(f"{name}: shares sum to {total:.6f}, expected 1.0")
    if keys is not None and set(shares) != set(keys):
        raise ValueError(f"{name}: keys {sorted(shares)} must be exactly {sorted(keys)}")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsCfg(_Model):
    data_root: Path
    runs: Path


class GridCfg(_Model):
    res_m: float = Field(gt=0)
    snap_m: float = Field(gt=0)
    tile_px: int = Field(gt=0)
    chip_px: int = Field(gt=0)

    @model_validator(mode="after")
    def _chip_not_smaller_than_tile(self) -> GridCfg:
        if self.chip_px < self.tile_px:
            raise ValueError(f"grid.chip_px ({self.chip_px}) < grid.tile_px ({self.tile_px})")
        return self


class BandsCfg(_Model):
    order: tuple[str, ...]
    scale: float = Field(gt=0)
    clip_max: float = Field(gt=0)
    normalise: Literal["fixed", "percentile", "irmad_sym"] = "fixed"

    @field_validator("order")
    @classmethod
    def _fixed_band_order(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if tuple(v) != BAND_ORDER:
            raise ValueError(f"bands.order must be exactly {list(BAND_ORDER)}, got {list(v)}")
        return tuple(v)


class SrCfg(_Model):
    method: Literal["sen2sr_lite_rgbn", "bicubic", "sen2sr_lite_main"]
    model_dir: Path
    model_dir_main: Path
    pool_to_5m: bool = True


class CloudCfg(_Model):
    method: Literal["omnicloudmask", "scl"]
    dilate_px: int = Field(ge=0)


class CoregCfg(_Model):
    method: Literal["phase_corr_half", "arosics_local_half", "none"]
    max_shift_px: float = Field(gt=0)


class ClassesCfg(_Model):
    ids: dict[str, int]
    rubric_weights: dict[str, float]
    priority: tuple[str, ...]
    min_area_m2: dict[str, float]
    min_length_m: dict[str, float]
    road_width_m: dict[str, float]

    @field_validator("ids")
    @classmethod
    def _ids_fixed(cls, v: dict[str, int]) -> dict[str, int]:
        if dict(v) != CLASS_IDS:
            raise ValueError(f"classes.ids must be {CLASS_IDS}, got {dict(v)}")
        return dict(v)

    @field_validator("rubric_weights")
    @classmethod
    def _rubric(cls, v: dict[str, float]) -> dict[str, float]:
        _check_shares("classes.rubric_weights", v, CHANGE_CLASSES)
        return dict(v)

    @field_validator("priority")
    @classmethod
    def _priority(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        if sorted(v) != sorted(CHANGE_CLASSES):
            raise ValueError(f"classes.priority must permute {CHANGE_CLASSES}, got {v}")
        return tuple(v)

    @model_validator(mode="after")
    def _sizes(self) -> ClassesCfg:
        for name, d in (("min_area_m2", self.min_area_m2), ("min_length_m", self.min_length_m)):
            for k, val in d.items():
                if k not in CHANGE_CLASSES:
                    raise ValueError(f"classes.{name}: unknown class {k!r}")
                if val <= 0:
                    raise ValueError(f"classes.{name}.{k} must be > 0")
        if "road" not in self.min_length_m:
            raise ValueError("classes.min_length_m must define road")
        if set(self.road_width_m) != {"highway", "access"}:
            raise ValueError("classes.road_width_m must define highway and access")
        return self


class StrataCfg(_Model):
    shares: dict[str, float]

    @field_validator("shares")
    @classmethod
    def _shares(cls, v: dict[str, float]) -> dict[str, float]:
        _check_shares("strata.shares", v, STRATA)
        return dict(v)


class ChipsCfg(_Model):
    stride_px: int = Field(gt=0)
    min_valid_fraction: float = Field(ge=0, le=1)
    min_change_px: int = Field(gt=0)


class SamplerCfg(_Model):
    tile_type_shares: dict[str, dict[str, float]]
    class_shares: dict[str, float]
    density_exponent: float = Field(ge=0)
    change_centred_prob: float = Field(ge=0, le=1)
    crop_centre_radius_px: int = Field(ge=0)
    auto_label_weight: float = Field(ge=0, le=1)

    @field_validator("tile_type_shares")
    @classmethod
    def _tile_shares(cls, v: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        if "default" not in v:
            raise ValueError("sampler.tile_type_shares must contain a 'default' table")
        for key, table in v.items():
            if key != "default" and key not in STRATA:
                raise ValueError(f"sampler.tile_type_shares: unknown stratum {key!r}")
            _check_shares(f"sampler.tile_type_shares.{key}", table, TILE_TYPES)
        return {k: dict(t) for k, t in v.items()}

    @field_validator("class_shares")
    @classmethod
    def _class_shares(cls, v: dict[str, float]) -> dict[str, float]:
        _check_shares("sampler.class_shares", v, CHANGE_CLASSES)
        return dict(v)

    def shares_for(self, stratum: str) -> dict[str, float]:
        return self.tile_type_shares.get(stratum, self.tile_type_shares["default"])


class ModelCfg(_Model):
    encoder: str
    encoder_init: Literal["imagenet", "v3_roads", "dinov3_convnext_tiny", "random"]
    in_channels: Literal[6]
    decoder_channels: tuple[int, int, int, int]
    fusion: Literal["absdiff_sum", "absdiff", "absdiff_prod"]
    full_res_skip: bool
    aux_semantic: bool
    head_mode: Literal["factorized", "softmax5"] = "factorized"
    n_change_classes: Literal[4]
    n_sem_classes: Literal[5]


class LossCfg(_Model):
    pos_weight: float = Field(gt=0)
    lambda_sem: float = Field(ge=0)
    class_weight_power: float = Field(ge=0)
    cldice_road: bool = False


class CheckpointCfg(_Model):
    save_top_k: int = Field(ge=1)
    milestone_every: int = Field(ge=1)
    monitor: str


class PretrainCfg(_Model):
    ckpt: Path | None = None
    chips_dir: Path | None = None


class TrainCfg(_Model):
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    crops_per_epoch: int = Field(ge=1)
    num_workers: int = Field(ge=0)
    lr_decoder: float = Field(gt=0)
    lr_encoder: float = Field(gt=0)
    lr_min: float = Field(ge=0)
    weight_decay: float = Field(ge=0)
    warmup_epochs: int = Field(ge=0)
    precision: str
    ema_decay: float = Field(ge=0, lt=1)
    loss: LossCfg
    checkpoint: CheckpointCfg
    early_stop_patience: int = Field(ge=1)
    robustness_subset: int = Field(ge=1)
    seed: int
    max_steps: int | None = None
    pretrain: PretrainCfg = PretrainCfg()

    @field_validator("precision", mode="before")
    @classmethod
    def _precision_str(cls, v: Any) -> str:
        s = str(v)
        if s not in {"32", "16-mixed", "bf16-mixed", "16", "bf16", "64"}:
            raise ValueError(f"train.precision {s!r} is not a Lightning precision string")
        return s


class GeometricAug(_Model):
    d4: float = Field(ge=0, le=1)
    rotate_deg: float = Field(ge=0)
    rotate_p: float = Field(ge=0, le=1)
    scale: tuple[float, float]
    scale_p: float = Field(ge=0, le=1)


class RadiometricAug(_Model):
    brightness: float = Field(ge=0)
    contrast: float = Field(ge=0)
    gamma: tuple[float, float]
    band_gain: float = Field(ge=0)
    haze_p: float = Field(ge=0, le=1)
    noise_sigma: tuple[float, float]


class ShiftAug(_Model):
    max_px: int = Field(ge=0)
    p: float = Field(ge=0, le=1)


class AugmentCfg(_Model):
    geometric: GeometricAug
    temporal_swap_p: float = Field(ge=0, le=1)
    radiometric: RadiometricAug
    shift: ShiftAug
    cloud_paste_p: float = Field(ge=0, le=1)
    band_dropout_p: float = Field(ge=0, le=1)


class PostprocessCfg(_Model):
    min_area_m2: dict[str, float]
    min_length_m: dict[str, float]
    hole_fill_m2: dict[str, float]
    solar_close_px: int = Field(ge=0)


class InferCfg(_Model):
    window_px: int = Field(gt=0)
    stride_px: int = Field(gt=0)
    tta_geometric: Literal[1, 4, 8]
    tta_temporal: bool
    identity_eps: float = Field(gt=0)
    nodata_value: Literal[255]
    invalid_value: Literal[0, 255] = 0
    simplify_m: float = Field(ge=0)
    water_ndwi: float
    postprocess: PostprocessCfg

    @model_validator(mode="after")
    def _stride(self) -> InferCfg:
        if self.stride_px > self.window_px:
            raise ValueError("infer.stride_px must be <= infer.window_px")
        return self


class FinetuneCfg(_Model):
    base_run: str | None = None
    epochs: int = Field(ge=1)
    lr_decoder: float = Field(gt=0)
    lr_encoder: float = Field(gt=0)
    freeze_encoder_stages: tuple[int, ...]
    freeze_epochs: int = Field(ge=0)
    mix_base: float = Field(ge=0, le=1)
    select_weights: dict[str, float]
    forgetting_tolerance: float = Field(ge=0)

    @field_validator("select_weights")
    @classmethod
    def _select(cls, v: dict[str, float]) -> dict[str, float]:
        _check_shares("finetune.select_weights", v, ("targets", "strata"))
        return dict(v)


class Config(_Model):
    paths: PathsCfg
    labeling_guide: Path = Path("labeling_guide.yaml")
    grid: GridCfg
    bands: BandsCfg
    sr: SrCfg
    cloud: CloudCfg
    coreg: CoregCfg
    classes: ClassesCfg
    strata: StrataCfg
    chips: ChipsCfg
    sampler: SamplerCfg
    model: ModelCfg
    train: TrainCfg
    augment: AugmentCfg
    infer: InferCfg
    finetune: FinetuneCfg

    # --- derived paths (HANDOFF 3.2) ---------------------------------------------------------
    @property
    def data_root(self) -> Path:
        return self.paths.data_root

    @property
    def runs_dir(self) -> Path:
        return self.paths.runs

    @property
    def registry_dir(self) -> Path:
        return self.data_root / "registry"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw" / "s2"

    @property
    def external_dir(self) -> Path:
        return self.data_root / "external"

    @property
    def pairs_dir(self) -> Path:
        return self.data_root / "pairs"

    @property
    def candidates_dir(self) -> Path:
        return self.data_root / "candidates"

    @property
    def reviewed_dir(self) -> Path:
        return self.data_root / "reviewed"

    @property
    def labels_dir(self) -> Path:
        return self.data_root / "labels"

    @property
    def outputs_dir(self) -> Path:
        return self.data_root / "outputs"

    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    def chips_dir(self, version: int) -> Path:
        return self.data_root / "chips" / f"v{version}"

    def objects_dir(self, version: int) -> Path:
        return self.data_root / "objects" / f"v{version}"

    def run_dir(self, run_name: str) -> Path:
        return self.runs_dir / run_name

    def labeling_guide_path(self) -> Path:
        p = self.labeling_guide
        return p if p.is_absolute() else REPO_ROOT / p

    # --- serialisation -----------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

    def to_yaml(self) -> str:
        return str(OmegaConf.to_yaml(OmegaConf.create(self.to_dict()), sort_keys=True))

    def hash(self) -> str:
        """Stable sha256 (first 16 hex chars) of the resolved configuration."""
        blob = json.dumps(self.to_dict(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())


# --- loading ---------------------------------------------------------------------------------
def _read_with_base(path: Path, _seen: tuple[Path, ...] = ()) -> DictConfig:
    path = path.resolve()
    if path in _seen:
        raise ConfigError(f"_base_ cycle: {' -> '.join(str(p) for p in (*_seen, path))}")
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ConfigError(f"{path} is not a mapping")
    base = cfg.pop("_base_", None)
    if base is not None:
        base_cfg = _read_with_base(path.parent / str(base), (*_seen, path))
        merged = OmegaConf.merge(base_cfg, cfg)
        assert isinstance(merged, DictConfig)
        return merged
    return cfg


def load(path: str | Path = "configs/default.yaml", overrides: Sequence[str] = ()) -> Config:
    """Load a config file (with `_base_` inheritance) plus `key=value` overrides into `Config`."""
    cfg = _read_with_base(Path(path))
    if overrides:
        try:
            cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))  # type: ignore[assignment]
        except OmegaConfBaseException as e:  # pragma: no cover - message formatting
            raise ConfigError(f"bad --set override {list(overrides)}: {e}") from e
    try:
        container = OmegaConf.to_container(cfg, resolve=True)
    except OmegaConfBaseException as e:
        raise ConfigError(
            f"cannot resolve config {path}: {e}. "
            "Are SCD_DATA (and optionally SCD_RUNS) set? See HANDOFF.md section 2.2."
        ) from e
    assert isinstance(container, dict)
    try:
        return Config.model_validate(container)
    except ValidationError as e:
        raise ConfigError(f"invalid config {path}:\n{e}") from e


def load_strata(path: str | Path = "configs/strata.yaml") -> dict[str, dict[str, Any]]:
    """Stratum definitions (contrast seasons, allowed months) from configs/strata.yaml."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    raw = OmegaConf.to_container(OmegaConf.load(p), resolve=True)
    assert isinstance(raw, dict)
    strata = raw["strata"]
    assert isinstance(strata, dict)
    if set(strata) != set(STRATA):
        raise ConfigError(f"strata.yaml must define exactly {STRATA}, got {sorted(strata)}")
    return {str(k): dict(v) for k, v in strata.items()}
