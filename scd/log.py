"""Rich logging for the package, scripts and notebook."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

if TYPE_CHECKING:
    from scd.config import Config

console = Console()
_CONFIGURED = False


def setup(level: str = "INFO") -> None:
    """Install a RichHandler on the root logger (idempotent)."""
    global _CONFIGURED
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False)],
        force=True,
    )
    for noisy in ("rasterio", "fiona", "matplotlib", "urllib3", "botocore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        setup(os.environ.get("SCD_LOG_LEVEL", "INFO"))
    return logging.getLogger(name)


def banner(cfg: Config | None = None) -> None:
    """Print environment and resolved paths (notebook section 0)."""
    from scd import __version__

    table = Table(title=f"ps10-scd {__version__}", show_header=False)
    table.add_row("python", sys.version.split()[0])
    table.add_row("platform", platform.platform())
    try:
        import torch

        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu"
        table.add_row("torch", f"{torch.__version__} ({dev})")
    except Exception as e:  # pragma: no cover
        table.add_row("torch", f"unavailable ({e})")
    for var in ("SCD_DATA", "SCD_RUNS", "CDSE_USER", "PC_SDK_SUBSCRIPTION_KEY", "HF_TOKEN"):
        val = os.environ.get(var)
        table.add_row(var, ("set" if "KEY" in var or "TOKEN" in var or "PASS" in var else val) if val else "-")
    if cfg is not None:
        table.add_row("data_root", str(cfg.data_root))
        table.add_row("runs", str(cfg.runs_dir))
        table.add_row("config hash", cfg.hash())
        root = Path(cfg.data_root)
        if root.exists():
            usage = shutil.disk_usage(root)
            table.add_row("free space", f"{usage.free / 1e9:.1f} GB")
    console.print(table)
