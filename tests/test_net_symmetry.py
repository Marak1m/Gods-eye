"""M5: SiameseChangeNet swap invariance (eval, fp32, batch 2), identity features, size, CPU backward."""

from __future__ import annotations

import pytest
import torch

from scd.model.encoder import adapt_stem
from scd.model.fusion import SymmetricFusion
from scd.model.losses import TotalLoss
from scd.model.net import SiameseChangeNet


@pytest.fixture(scope="module")
def net() -> SiameseChangeNet:
    torch.manual_seed(0)
    return SiameseChangeNet(encoder_init="random").eval()


def _pair(size: int = 128, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, 6, size, size, generator=g), torch.rand(2, 6, size, size, generator=g)


def test_net_symmetry(net: SiameseChangeNet) -> None:
    x1, x2 = _pair()
    with torch.no_grad():
        a = net(x1, x2)
        b = net(x2, x1)
    assert a["change"].shape == (2, 1, 128, 128) and a["cls"].shape == (2, 4, 128, 128)
    assert torch.equal(a["change"], b["change"])  # exact (DEVIATIONS C2)
    assert torch.equal(a["cls"], b["cls"])
    assert a["sem_t1"] is not None and a["sem_t1"].shape == (2, 5, 128, 128)
    assert torch.equal(a["sem_t1"], b["sem_t2"]) and torch.equal(a["sem_t2"], b["sem_t1"])


@pytest.mark.parametrize(
    "fusion,head_mode,skip", [("absdiff", "factorized", False), ("absdiff_prod", "softmax5", True)]
)
def test_net_symmetry_variants(fusion: str, head_mode: str, skip: bool) -> None:
    torch.manual_seed(1)
    m = SiameseChangeNet(
        encoder_init="random", fusion=fusion, head_mode=head_mode, full_res_skip=skip, aux_semantic=False
    ).eval()
    x1, x2 = _pair(64, seed=2)
    with torch.no_grad():
        a, b = m(x1, x2), m(x2, x1)
    assert torch.equal(a["change"], b["change"]) and torch.equal(a["cls"], b["cls"])
    assert a["sem_t1"] is None


def test_identical_inputs_zero_difference_features(net: SiameseChangeNet) -> None:
    x, _ = _pair(64)
    with torch.no_grad():
        f1, f2 = net.encode(x, x.clone())
        d = SymmetricFusion.diff_half(f1, f2)
        out = net(x, x.clone())
    assert all(torch.equal(t, torch.zeros_like(t)) for t in d)
    assert torch.isfinite(out["change"]).all()


def test_param_count_in_range(net: SiameseChangeNet) -> None:
    n = net.param_count()
    assert 30_000_000 <= n <= 50_000_000, n  # DEVIATIONS C7 (handoff says about 45 M)
    groups = net.param_groups(1e-4, 3e-4, 1e-4)
    assert len(groups) == 2 and sum(p.numel() for g in groups for p in g["params"]) == n


def test_adapt_stem_rule() -> None:
    w = torch.randn(96, 3, 4, 4)
    w6 = adapt_stem(w)
    assert w6.shape == (96, 6, 4, 4)
    assert torch.equal(w6[:, 0], w[:, 2]) and torch.equal(w6[:, 1], w[:, 1]) and torch.equal(w6[:, 2], w[:, 0])
    assert torch.allclose(w6[:, 3], w.mean(1)) and torch.equal(w6[:, 3], w6[:, 5])


def test_training_mode_stacks_and_backward_cpu() -> None:
    torch.manual_seed(3)
    m = SiameseChangeNet(encoder_init="random").train()
    x1, x2 = _pair(256, seed=4)
    f1, f2 = m.encode(x1, x2)
    assert f1[0].shape == (2, 96, 64, 64) and len(f1) == 4 and f1[3].shape[1] == 768
    out = m(x1, x2)
    g = torch.Generator().manual_seed(5)
    change = torch.randint(0, 5, (2, 256, 256), generator=g)
    change[:, :8] = 255
    batch = {
        "change": change,
        "sem_t1": torch.randint(0, 5, (2, 256, 256), generator=g),
        "sem_t2": torch.randint(0, 5, (2, 256, 256), generator=g),
        "valid": torch.ones(2, 256, 256, dtype=torch.bool),
        "weight": torch.tensor([1.0, 0.5]),
    }
    terms = TotalLoss()(out, batch)
    terms["total"].backward()
    assert torch.isfinite(terms["total"])
    assert {"change", "cls", "sem_t1", "sem_t2", "total"} <= set(terms)
    enc_grad = next(p.grad for p in m.encoder.parameters() if p.grad is not None)
    assert torch.isfinite(enc_grad).all()
    assert m.change_head is not None and next(m.change_head.parameters()).grad is not None
