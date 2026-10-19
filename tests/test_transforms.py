"""M4: augmentation contracts (HANDOFF 9 `test_transforms`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scd.constants import BAND_INDEX, IGNORE
from scd.data import transforms as T
from scd.data.dataset import DrawRequest


def _sample(size: int = 32, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    change = torch.randint(0, 5, (size, size), generator=g)
    return {
        "x1": torch.rand(6, size, size, generator=g),
        "x2": torch.rand(6, size, size, generator=g),
        "change": change,
        "sem_t1": torch.randint(0, 5, (size, size), generator=g),
        "sem_t2": torch.randint(0, 5, (size, size), generator=g),
        "valid": torch.ones(size, size, dtype=torch.bool),
    }


def test_d4_identical_on_images_and_labels() -> None:
    s = _sample()
    out = T.d4(s, k=1, flip=True)
    ref = torch.flip(torch.rot90(s["change"], 1, dims=(-2, -1)), dims=(-1,))
    assert torch.equal(out["change"], ref)
    assert torch.equal(out["x1"], torch.flip(torch.rot90(s["x1"], 1, dims=(-2, -1)), dims=(-1,)))
    assert torch.equal(out["x2"][0], torch.flip(torch.rot90(s["x2"][0], 1, dims=(-2, -1)), dims=(-1,)))
    assert torch.equal(out["valid"], torch.flip(torch.rot90(s["valid"], 1, dims=(-2, -1)), dims=(-1,)))


def test_affine_shared_consistency() -> None:
    s = _sample()
    rows, cols = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    s["change"] = (rows // 8 + cols // 8) % 5  # blocky labels so bilinear and nearest agree inside blocks
    s["x1"][0] = s["change"].float()  # image channel carrying the label pattern
    out = T.affine_shared(s, angle_deg=25.0, scale=1.0)
    oob = out["change"] == IGNORE
    assert oob.any() and torch.equal(oob, ~out["valid"])  # corners fall outside after rotation
    assert torch.equal(out["sem_t1"] == IGNORE, oob)
    # the same geometric map was applied to images (bilinear) and labels (nearest): agree where in-bounds
    inb = ~oob
    agree = (out["x1"][0][inb].round().long() == out["change"][inb]).float().mean()
    assert agree > 0.8
    ninety = T.affine_shared(_sample(), angle_deg=90.0, scale=1.0)
    assert ninety["valid"].all()


def test_relative_shift_moves_one_image_only() -> None:
    s = _sample()
    out = T.relative_shift(s, dx=2, dy=0, which="x2")
    assert torch.equal(out["x2"][..., :, 2:], s["x2"][..., :, :-2])
    assert torch.equal(out["x2"][..., :, :2], torch.zeros_like(out["x2"][..., :, :2]))
    assert torch.equal(out["x1"], s["x1"])
    for k in ("change", "sem_t1", "sem_t2", "valid"):
        assert torch.equal(out[k], s[k])  # labels untouched (HANDOFF 14)


def test_temporal_swap_flips_dates() -> None:
    s = _sample()
    out = T.temporal_swap(s)
    assert torch.equal(out["x1"], s["x2"]) and torch.equal(out["x2"], s["x1"])
    assert torch.equal(out["sem_t1"], s["sem_t2"]) and torch.equal(out["sem_t2"], s["sem_t1"])
    assert torch.equal(out["change"], s["change"])
    back = T.temporal_swap(out)
    assert torch.equal(back["x1"], s["x1"])


def test_band_dropout_identical_on_both_dates() -> None:
    s = _sample()
    out = T.band_dropout(s)
    for key in ("x1", "x2"):
        assert (out[key][[BAND_INDEX["B11"], BAND_INDEX["B12"]]] == 0).all()
        assert torch.equal(out[key][:4], s[key][:4])


def test_cloud_paste_marks_blob_invalid_and_ignored() -> None:
    s = _sample(64)
    out = T.cloud_paste(s, np.random.default_rng(0), which="x2", clip_max=1.2)
    blob = ~out["valid"]
    assert blob.any() and not blob.all()
    assert (out["change"][blob] == IGNORE).all() and torch.equal(out["change"][~blob], s["change"][~blob])
    assert torch.equal(out["x1"], s["x1"])
    assert (out["x2"][:, blob] != s["x2"][:, blob]).any()


def test_radiometric_stays_in_range(cfg) -> None:
    s = _sample()
    out = T.radiometric(s["x1"], np.random.default_rng(1), cfg)
    assert out.shape == s["x1"].shape and out.min() >= 0 and out.max() <= cfg.bands.clip_max
    assert not torch.equal(out, s["x1"])


def test_apply_all_identical_draw_keeps_dates_equal(cfg) -> None:
    s = _sample(64)
    s["x2"] = s["x1"].clone()
    s["change"] = torch.zeros_like(s["change"])
    req = DrawRequest("c", "IDENTICAL", seed=3, swap=False)
    for seed in range(5):
        out = T.apply_all(dict(s), req, cfg, np.random.default_rng(seed))
        assert torch.equal(out["x1"], out["x2"])  # no date-asymmetric op touched an IDENTICAL draw
    req2 = DrawRequest("c", "CHANGE_POS", seed=3, swap=False)
    out2 = T.apply_all(dict(s), req2, cfg, np.random.default_rng(0))
    assert not torch.equal(out2["x1"], out2["x2"])  # asymmetric radiometric jitter applied


@pytest.mark.parametrize("seed", range(3))
def test_apply_all_shapes_and_label_values(cfg, seed: int) -> None:
    s = _sample(64)
    req = DrawRequest("c", "CHANGE_POS", seed=seed, swap=bool(seed % 2))
    out = T.apply_all(s, req, cfg, np.random.default_rng(seed))
    assert out["x1"].shape == (6, 64, 64) and out["change"].shape == (64, 64)
    vals = set(torch.unique(out["change"]).tolist())
    assert vals <= {0, 1, 2, 3, 4, IGNORE}
