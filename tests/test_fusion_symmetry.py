"""M5: `torch.equal(fusion(f1, f2), fusion(f2, f1))` for every fusion mode (HANDOFF 9)."""

from __future__ import annotations

import pytest
import torch

from scd.model.fusion import FUSION_MODES, SymmetricFusion

CH = (8, 16, 32, 64)


def _feats(seed: int, batch: int = 2, size: int = 32) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(batch, c, size >> i, size >> i, generator=g) for i, c in enumerate(CH)]


@pytest.mark.parametrize("mode", FUSION_MODES)
def test_fusion_symmetry(mode: str) -> None:
    torch.manual_seed(0)
    fusion = SymmetricFusion(CH, mode=mode).eval()
    f1, f2 = _feats(1), _feats(2)
    with torch.no_grad():
        a = fusion(f1, f2)
        b = fusion(f2, f1)
    assert len(a) == 4
    for x, y, c in zip(a, b, CH):
        assert torch.equal(x, y)  # exact, not allclose
        assert x.shape[1] == c


def test_identical_inputs_have_zero_difference_half() -> None:
    f1 = _feats(3)
    d = SymmetricFusion.diff_half(f1, [t.clone() for t in f1])
    assert all(torch.equal(t, torch.zeros_like(t)) for t in d)


def test_bad_mode_rejected() -> None:
    with pytest.raises(ValueError):
        SymmetricFusion(CH, mode="concat")
