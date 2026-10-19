"""RGBN resampling: SEN2SR-Lite x4 then 2x2 mean to 5 m, or plain bicubic (HANDOFF.md 2.1, 5.4).

The SEN2SR path is UNVERIFIED on this machine (no model download / GPU; DEVIATIONS). It follows
the API notes in HANDOFF.md 2.1: input (1, 4, H, W) float32 = DN/10000 in band order
B04, B03, B02, B08; `sen2sr.predict_large(model, X, overlap=32)`; output 4x the input size.
SR output has no nodata marker: validity is always derived from the 10 m source (section 14).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scd.constants import RGBN_BANDS, SEN2SR_RGBN_ORDER, UINT16_SCALE
from scd.geo.reproject import resample
from scd.log import get_logger

log = get_logger(__name__)

SEN2SR_URLS = {
    "sen2sr_lite_rgbn": "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
    "sen2sr_lite_main": "https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/main/mlm.json",
}
_RGBN_TO_SEN2SR = [RGBN_BANDS.index(b) for b in SEN2SR_RGBN_ORDER]  # B02,B03,B04,B08 -> B04,B03,B02,B08
_SEN2SR_TO_RGBN = [SEN2SR_RGBN_ORDER.index(b) for b in RGBN_BANDS]


def bicubic_to_5m(arr: np.ndarray, src_res_m: float, dst_res_m: float = 5.0) -> np.ndarray:
    """Plain bicubic resampling from the native resolution to the working grid resolution."""
    return resample(arr, src_res_m / dst_res_m, "bicubic")


def load_sen2sr(model_dir: Path, device: str = "cuda", method: str = "sen2sr_lite_rgbn") -> Any:
    """Download (once) and load the compiled SEN2SR-Lite model via mlstac."""
    import mlstac

    model_dir = Path(model_dir)
    if not (model_dir / "mlm.json").exists():
        log.info("downloading %s to %s", method, model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        mlstac.download(file=SEN2SR_URLS[method], output_dir=str(model_dir))
    return mlstac.load(str(model_dir)).compiled_model(device=device)


def super_resolve_rgbn(rgbn_dn: np.ndarray, model: Any, pool_to_5m: bool = True) -> np.ndarray:
    """(4, H, W) DN in B02,B03,B04,B08 order at 10 m -> float32 DN at 2.5 m (x4) or 5 m (pooled)."""
    import sen2sr
    import torch

    if rgbn_dn.shape[0] != 4:
        raise ValueError(f"expected 4 RGBN bands, got {rgbn_dn.shape}")
    x = torch.from_numpy(np.ascontiguousarray(rgbn_dn[_RGBN_TO_SEN2SR]).astype(np.float32) / UINT16_SCALE)[None]
    device = next(model.parameters()).device if hasattr(model, "parameters") else "cpu"
    with torch.no_grad():
        y = sen2sr.predict_large(model, x.to(device), overlap=32)
    out = y[0].detach().float().cpu().numpy()[_SEN2SR_TO_RGBN] * UINT16_SCALE
    if pool_to_5m:
        out = resample(out, 0.5, "mean")
    return out.astype(np.float32)
