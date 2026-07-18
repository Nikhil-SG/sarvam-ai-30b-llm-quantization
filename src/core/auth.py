"""
HuggingFace authentication & storage configuration.

Handles three concerns from a single module:
  1. **HF_HOME / cache redirect** – so nothing writes to /home.
    2. **Token resolution** – env var → cli login → config fallback.
  3. **Model path resolution** – local dir on /data → HF Hub ID.
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from src.core.logger import get_logger

logger = get_logger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(raw_path: str) -> Path:
    p = Path(raw_path)
    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (_repo_root() / p).resolve()

    root = _repo_root().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Unsafe path outside repository: {raw_path} -> {resolved}"
        ) from exc
    return resolved


def _migrate_legacy_model_dirs(target_path: Path) -> Optional[Path]:
    """
    One-time migration helper for old shared model folders.

    If target path does not exist, migrate first available legacy directory:
    - ./Base_model
    - ./Model
    """
    if target_path.exists():
        return target_path

    root = _repo_root()
    legacy_names = ["Base_model", "Model"]

    for name in legacy_names:
        src = (root / name).resolve()
        if not src.exists() or not src.is_dir():
            continue

        try:
            src.relative_to(root.resolve())
        except ValueError:
            logger.warning(f"Skipping unsafe legacy model path: {src}")
            continue

        target_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.warning(
                f"Migrating legacy model directory: {src} -> {target_path}"
            )
            shutil.move(str(src), str(target_path))
            logger.info(f"Legacy model migration complete: {target_path}")
            return target_path
        except Exception as exc:
            logger.warning(
                f"Legacy migration failed for {src}: {exc}. Using legacy path for this run."
            )
            return src

    return None


# Storage — redirect all HF caching to /data before any download happens
# ---------------------------------------------------------------------------
def configure_hf_home(config) -> None:
    """
    Set ``HF_HOME`` and ``TRANSFORMERS_CACHE`` env vars from
    ``config.storage.hf_home`` so that *every* library (transformers,
    datasets, tokenizers, huggingface_hub) writes to the configured location.

    Resolves relative paths (e.g., ./hf_cache) to absolute paths.
    Safe to call multiple times — only acts once.
    """
    hf_home = getattr(getattr(config, "storage", None), "hf_home", None)
    if not hf_home:
        return

    # Convert to a repository-contained absolute path
    hf_home_path = _resolve_repo_path(hf_home)
    hf_home_str = str(hf_home_path)
    hf_home_path.mkdir(parents=True, exist_ok=True)

    for var in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
        if os.environ.get(var) != hf_home_str:
            os.environ[var] = hf_home_str

    logger.info(f"HF_HOME set to {hf_home_str}")


# Token resolution (priority: env -> cli cache -> config fallback)
# ---------------------------------------------------------------------------
def resolve_hf_token(config) -> Optional[str]:
    """
    Return a valid HF token or ``None``.

    Args:
        config: Loaded ``Config`` object (needs ``config.model``).

    Returns:
        Token string, or None if no credentials found anywhere.
    """
    # Always configure storage first
    configure_hf_home(config)

    # ── 1. Environment variable ─────────────────────────────────────────
    token = os.environ.get("HF_TOKEN")
    if token:
        logger.debug("HF token: $HF_TOKEN env var")
        return token

    # ── 2. huggingface-cli login cache ──────────────────────────────────
    try:
        from huggingface_hub import HfFolder

        token = HfFolder.get_token()
        if token:
            logger.debug("HF token: cli login cache")
            return token
    except Exception:
        pass

    # ── 3. config.yaml fallback (backward compatibility) ───────────────
    token = getattr(config.model, "hf_token", None)
    if token:
        logger.warning("HF token loaded from config.yaml fallback; prefer HF_TOKEN env var")
        return token

    logger.warning(
        "No HF token found ($HF_TOKEN / huggingface-cli login / config.yaml fallback)"
    )
    return None


# Model path resolution (local dir on /data -> HF cache -> HF Hub model_id)
# ---------------------------------------------------------------------------
def _find_cached_snapshot(model_id: str, config) -> Optional[str]:
    """
    Scan the HuggingFace cache directory for an already-downloaded snapshot
    of ``model_id``.  Returns the snapshot path if found, else ``None``.

    Checks both ``cache_dir`` (from_pretrained style) and ``HF_HOME/hub``
    (env-var style) cache layouts.
    """
    import os

    model_slug = model_id.replace("/", "--")
    candidate_dirs = []

    # 1. cache_dir from config (used by from_pretrained directly)
    cache_dir = getattr(getattr(config, "model", None), "cache_dir", None)
    if cache_dir:
        candidate_dirs.append(_resolve_repo_path(cache_dir))

    # 2. HF_HOME/hub (huggingface_hub env-var layout)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidate_dirs.append(Path(hf_home) / "hub")
        candidate_dirs.append(Path(hf_home))  # some downloads land here

    for base in candidate_dirs:
        model_dir = base / f"models--{model_slug}"
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.is_dir():
            # Pick the most recently modified snapshot
            snaps = sorted(
                [s for s in snapshots_dir.iterdir() if s.is_dir()],
                key=lambda s: s.stat().st_mtime,
                reverse=True,
            )
            if snaps:
                snap = str(snaps[0])
                logger.info(f"Found cached snapshot: {snap}")
                return snap

    return None


def resolve_model_path(config) -> str:
    """
    Return the effective model path to pass to ``from_pretrained()``.

    Priority:
      1. ``storage.local_model_path`` (if set and the directory exists)
         Resolves relative paths (e.g., ./Base_model) to absolute paths.
      2. Locally cached snapshot inside ``model.cache_dir`` or ``HF_HOME``
         (avoids re-downloading a model that was already fetched).
      3. ``model.model_id`` (HuggingFace Hub identifier — triggers download).

    Returns:
        Absolute local path **or** HF model ID string.
    """
    # Always configure HF_HOME first so cache lookups work
    configure_hf_home(config)

    # ── 1. Explicit local directory ─────────────────────────────────────
    local = getattr(getattr(config, "storage", None), "local_model_path", None)
    if local:
        p = _resolve_repo_path(local)

        # One-time compatibility migration from older directory names.
        migrated = _migrate_legacy_model_dirs(p)
        if migrated is not None:
            p = migrated.resolve()

        if p.is_dir():
            # If local_model_path points to a full model repository, use it directly.
            if (p / "config.json").is_file():
                logger.info(f"Model source: local ({p})")
                return str(p)

            # If local_model_path points to a cache root (e.g., ./Model),
            # resolve the latest snapshot for this model under it.
            model_id = config.model.model_id
            model_slug = model_id.replace("/", "--")
            cache_roots = [p, p / "hub"]
            for root in cache_roots:
                snaps_dir = root / f"models--{model_slug}" / "snapshots"
                if not snaps_dir.is_dir():
                    continue
                snaps = sorted(
                    [s for s in snaps_dir.iterdir() if s.is_dir()],
                    key=lambda s: s.stat().st_mtime,
                    reverse=True,
                )
                if snaps:
                    snap = str(snaps[0])
                    logger.info(f"Model source: local cached snapshot ({snap})")
                    return snap

            logger.warning(
                f"Local model path exists but is not a model repository: {p} — checking HF cache"
            )
        else:
            logger.warning(
                f"Local model path not found: {p} — checking HF cache"
            )

    # ── 2. HF cache snapshot (already downloaded) ───────────────────────
    model_id = config.model.model_id
    snap = _find_cached_snapshot(model_id, config)
    if snap:
        logger.info(f"Model source: cached snapshot ({snap})")
        return snap

    # ── 3. Fall back to HF Hub ID (will trigger download) ───────────────
    logger.info(f"Model source: HF Hub ({model_id})")
    return model_id
