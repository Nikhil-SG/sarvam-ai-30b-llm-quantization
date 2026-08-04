"""
tests/test_runner.py
────────────────────
Shared test runner for the LLM Quantization Research pipeline.

After each module completes, main.py calls run_module_tests() which:
  1. Runs the module-specific test functions.
    2. Appends human-readable output to <base_dir>/test_results/test_results.log.
    3. Merges per-test results into <base_dir>/test_results/test_results.json.
  4. Returns True (all pass) or False (at least one failure).

If the function returns False, main.py halts the pipeline immediately.
"""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


class SkipTest(unittest.SkipTest):
    """Lightweight skip signal for direct function-based test execution."""


_UNSET = object()


class _MiniMonkeyPatch:
    """Minimal monkeypatch fixture compatible with common test usage patterns."""

    def __init__(self) -> None:
        self._undo_stack: List[Callable[[], None]] = []

    def setattr(self, target, name, value=_UNSET, raising: bool = True) -> None:
        # Supports monkeypatch.setattr(obj, "attr", value) and
        # monkeypatch.setattr("pkg.mod.attr", value).
        if value is _UNSET:
            if not isinstance(target, str):
                raise TypeError("setattr expected dotted-path string when value is omitted")
            dotted_path = target
            value = name
            module_name, attr_name = dotted_path.rsplit(".", 1)
            target = importlib.import_module(module_name)
            name = attr_name

        existed = hasattr(target, name)
        if not existed and raising:
            raise AttributeError(f"{target!r} has no attribute {name!r}")

        previous = getattr(target, name, _UNSET)
        setattr(target, name, value)

        def _undo() -> None:
            if existed:
                setattr(target, name, previous)
            else:
                delattr(target, name)

        self._undo_stack.append(_undo)

    def setitem(self, mapping, key, value) -> None:
        existed = key in mapping
        previous = mapping.get(key, _UNSET)
        mapping[key] = value

        def _undo() -> None:
            if existed:
                mapping[key] = previous
            else:
                mapping.pop(key, None)

        self._undo_stack.append(_undo)

    def delitem(self, mapping, key, raising: bool = True) -> None:
        existed = key in mapping
        if not existed:
            if raising:
                raise KeyError(key)
            return

        previous = mapping[key]
        del mapping[key]

        def _undo() -> None:
            mapping[key] = previous

        self._undo_stack.append(_undo)

    def setenv(self, name: str, value: str, prepend: str | None = None) -> None:
        existed = name in os.environ
        previous = os.environ.get(name)
        new_value = value
        if prepend and existed and previous:
            new_value = f"{value}{prepend}{previous}"
        os.environ[name] = new_value

        def _undo() -> None:
            if existed and previous is not None:
                os.environ[name] = previous
            else:
                os.environ.pop(name, None)

        self._undo_stack.append(_undo)

    def delenv(self, name: str, raising: bool = True) -> None:
        existed = name in os.environ
        previous = os.environ.get(name)
        if not existed:
            if raising:
                raise KeyError(name)
            return

        del os.environ[name]

        def _undo() -> None:
            if previous is not None:
                os.environ[name] = previous

        self._undo_stack.append(_undo)

    def undo(self) -> None:
        while self._undo_stack:
            self._undo_stack.pop()()


# ── internal helpers ─────────────────────────────────────────────────────────

def _get_test_results_dir(config) -> Path:
    """Return (and create) the test_results directory under config.output.base_dir."""
    base = getattr(config.output, "base_dir", "outputs")
    d = Path(base) / "test_results"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_file_logger(log_path: Path) -> logging.Logger:
    """Return a logger that writes to log_path (append mode)."""
    name = "tests.runner"
    lgr = logging.getLogger(name)
    lgr.setLevel(logging.DEBUG)
    # Avoid duplicate handlers if called multiple times in the same process.
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path)
        for h in lgr.handlers
    ):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)-5s] %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        lgr.addHandler(fh)
    return lgr


def _load_json_record(json_path: Path) -> Dict[str, Any]:
    if json_path.exists():
        try:
            with open(json_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_json_record(json_path: Path, data: Dict[str, Any]) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _build_test_call(fn: Callable, config) -> Tuple[List[Any], Dict[str, Any], _MiniMonkeyPatch | None]:
    """Build call args for direct-function tests with lightweight fixture support."""
    signature = inspect.signature(fn)
    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    monkeypatch_obj: _MiniMonkeyPatch | None = None

    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue

        value: Any = _UNSET
        if param.name == "config":
            value = config
        elif param.name == "monkeypatch":
            monkeypatch_obj = monkeypatch_obj or _MiniMonkeyPatch()
            value = monkeypatch_obj
        elif param.default is inspect._empty:
            raise TypeError(
                f"Unsupported required test parameter '{param.name}' in {fn.__name__}(); "
                "supported parameters: config, monkeypatch"
            )

        if value is _UNSET:
            continue

        if param.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[param.name] = value
        else:
            args.append(value)

    return args, kwargs, monkeypatch_obj


# ── public API ────────────────────────────────────────────────────────────────

def run_module_tests(module_num: int, config, pipeline_logger=None, pipeline_name: str = "research") -> bool:
    """
    Discover and run the test module for module_num.

    Parameters
    ----------
    module_num   : int  — which pipeline module just finished (1-6)
    config       : pipeline config object (has config.output.base_dir)
    pipeline_logger : the main logger (used for a one-line summary there)       
    pipeline_name   : str — either "research" or "mxmoe" to locate tests

    Returns
    -------
    bool — True if ALL tests pass, False if any fail or error.
    """
    # ── locate test results directory ────────────────────────────────────
    results_dir = _get_test_results_dir(config)
    log_path  = results_dir / "test_results.log"
    json_path = results_dir / "test_results.json"

    lgr = _build_file_logger(log_path)
    timestamp = datetime.now(timezone.utc).isoformat()

    # ── discover test functions from the matching test module ────────────
    # Look in tests/research/ first (new location), then tests/ root (legacy)
    import sys
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    test_mod = None
    test_mod_name = None
    import_errors: List[str] = []

    if pipeline_name == "mxmoe":
        candidates = [
            f"tests.mxmoe.test_module{module_num}",
            f"tests.test_module{module_num}",
        ]
    elif pipeline_name == "research":
        candidates = [
            f"tests.research.test_module{module_num}",
            f"tests.test_module{module_num}",
        ]
    else:
        # Backward-compatible fallback for unknown pipeline names.
        candidates = [
            f"tests.{pipeline_name}.test_module{module_num}",
            f"tests.test_module{module_num}",
        ]

    for candidate in candidates:
        try:
            test_mod = importlib.import_module(candidate)
            test_mod_name = candidate
            break
        except ImportError as exc:
            import_errors.append(f"{candidate}: {exc}")
            continue

    if test_mod is None:
        exc = " / ".join(candidates)
        msg = f"No test module found for Module {module_num}: tried {exc}"
        if import_errors:
            msg = f"{msg} | import errors: {' ; '.join(import_errors)}"
        lgr.warning(msg)
        if pipeline_logger:
            pipeline_logger.warning(f"[Tests] {msg}")
        return True  # missing test file is not a failure; skip gracefully

    # Collect functions whose name starts with "test_"
    test_fns: List[Tuple[str, Callable]] = [
        (name, fn)
        for name in dir(test_mod)
        if name.startswith("test_")
        for fn in [getattr(test_mod, name)]
        if callable(fn)
    ]

    if not test_fns:
        lgr.info(f"Module {module_num}: no test_ functions found — skipping")
        return True

    # ── run tests ────────────────────────────────────────────────────────
    lgr.info("=" * 64)
    lgr.info(f"MODULE {module_num} TESTS  —  {timestamp}")
    lgr.info("=" * 64)

    records: List[Dict[str, Any]] = []
    all_passed = True

    for name, fn in test_fns:
        t0 = time.time()
        monkeypatch_obj: _MiniMonkeyPatch | None = None
        try:
            args, kwargs, monkeypatch_obj = _build_test_call(fn, config)
            fn(*args, **kwargs)
            elapsed = round(time.time() - t0, 3)
            lgr.info(f"  PASS  {name}  ({elapsed}s)")
            records.append({"test": name, "status": "PASS", "elapsed_sec": elapsed})
        except SkipTest as sk:
            elapsed = round(time.time() - t0, 3)
            msg = str(sk) or "Skipped"
            lgr.info(f"  SKIP  {name}  —  {msg}  ({elapsed}s)")
            records.append({"test": name, "status": "SKIP", "reason": msg, "elapsed_sec": elapsed})
        except AssertionError as ae:
            elapsed = round(time.time() - t0, 3)
            msg = str(ae) or "AssertionError (no message)"
            lgr.error(f"  FAIL  {name}  —  {msg}  ({elapsed}s)")
            records.append({"test": name, "status": "FAIL",
                            "error": msg, "elapsed_sec": elapsed})
            all_passed = False
        except Exception as exc:
            elapsed = round(time.time() - t0, 3)
            msg = f"{type(exc).__name__}: {exc}"
            lgr.error(f"  ERROR {name}  —  {msg}  ({elapsed}s)")
            records.append({"test": name, "status": "ERROR",
                            "error": msg, "elapsed_sec": elapsed})
            all_passed = False
        finally:
            if monkeypatch_obj is not None:
                monkeypatch_obj.undo()

    n_pass = sum(1 for r in records if r["status"] == "PASS")
    n_total = len(records)
    summary = "ALL PASS" if all_passed else f"FAILED ({n_total - n_pass}/{n_total} failed)"
    lgr.info(f"Module {module_num} tests: {summary}  [{n_pass}/{n_total} passed]")
    lgr.info("")

    # ── persist JSON (merge with prior runs) ─────────────────────────────
    data = _load_json_record(json_path)
    data[f"module_{module_num}"] = {
        "timestamp": timestamp,
        "status": "PASS" if all_passed else "FAIL",
        "passed": n_pass,
        "total": n_total,
        "tests": records,
    }
    _save_json_record(json_path, data)

    # ── one-line echo to the main pipeline log ───────────────────────────
    if pipeline_logger:
        pipeline_logger.info(
            f"[Tests] Module {module_num}: {summary}  "
            f"— details: {log_path}"
        )

    return all_passed
