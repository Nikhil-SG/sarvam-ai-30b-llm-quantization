"""
Shared visualization utilities.

Provides consistent matplotlib styling, color palette, and save helpers
used by all visualization modules (layer_viz, precision_heatmap, pareto, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.core.logger import get_logger

logger = get_logger(__name__)


# ── Color Palette ────────────────────────────────────────────────────────

QUANTIZATION_COLORS: Dict[str, str] = {
    "bf16": "#4C72B0",
    "int8": "#55A868",
    "fp8":  "#C44E52",
    "nf4":  "#8172B3",
    "gptq": "#CCB974",
}

IMPORTANCE_COLORS: Dict[str, str] = {
    "HIGH":   "#2ecc71",  # green
    "MEDIUM": "#f39c12",  # amber
    "LOW":    "#e74c3c",  # red
}


def get_method_color(method: str) -> str:
    """Return the standard color for a quantization method."""
    return QUANTIZATION_COLORS.get(method.lower(), "#888888")


def get_importance_color(importance: str) -> str:
    """Return the standard color for an importance tier."""
    return IMPORTANCE_COLORS.get(importance.upper(), "#888888")


# ── Plot Style ───────────────────────────────────────────────────────────

def setup_plot_style(style: str = "seaborn-v0_8-whitegrid") -> None:
    """Apply consistent matplotlib style across all figures."""
    import matplotlib.pyplot as plt
    try:
        plt.style.use(style)
    except OSError:
        plt.style.use("seaborn-v0_8-whitegrid")

    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
    })


# ── Save Helper ──────────────────────────────────────────────────────────

def save_figure(
    fig,
    path: Path,
    dpi: int = 150,
    close: bool = True,
) -> Path:
    """Save a matplotlib figure with tight layout and proper DPI."""
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    logger.info(f"  Saved figure: {path}")

    if close:
        plt.close(fig)

    return path


# ── Figure Creation Helper ───────────────────────────────────────────────

def create_figure(
    figsize: Tuple[float, float] = (14, 8),
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Create a figure with standard sizing and optional title."""
    import matplotlib.pyplot as plt

    setup_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    return fig, ax
