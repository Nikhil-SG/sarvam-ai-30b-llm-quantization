"""
Shared calibration data loading.

Extracts the calibration dataset preparation logic that was copy-pasted
in ``compressor.py``, ``fp8_quantizer.py``, ``gptq_quantizer.py``, and
``int8_quantizer.py`` into a single reusable function.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
from src.core.logger import get_logger

logger = get_logger(__name__)

_LOCAL_FILE_EXTENSIONS = (".parquet", ".json", ".csv", ".jsonl", ".arrow")


_TEXT_FIELD_FALLBACKS = (
    "text",
    "content",
    "body",
    "document",
    "article",
    "sentence",
)


def _iter_text_values(
    sample: Dict[str, Any],
    preferred_text_column: Optional[str] = None,
) -> Iterable[str]:
    """Yield candidate text values from a dataset row."""
    if preferred_text_column:
        value = sample.get(preferred_text_column)
        if isinstance(value, str) and value.strip():
            yield value.strip()

    messages = sample.get("messages")
    if isinstance(messages, list):
        joined = " ".join(
            msg.get("content", "")
            for msg in messages
            if isinstance(msg, dict) and msg.get("content")
        ).strip()
        if joined:
            yield joined

    for key in _TEXT_FIELD_FALLBACKS:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            yield value.strip()


def load_calibration_texts(
    dataset_name: str = "wikitext",
    dataset_config: Optional[str] = "wikitext-2-raw-v1",
    split: str = "train",
    num_samples: int = 128,
    seed: int = 42,
    text_column: Optional[str] = "text",
    streaming: bool = False,
    min_text_length: int = 200,
    fallback_datasets: Optional[List[Dict[str, Optional[str]]]] = None,
    hf_token: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """
    Load calibration texts with a deterministic fallback chain.

    Returns:
        Tuple[List[str], List[str]]: collected texts and dataset names that
        contributed samples.
    """
    from datasets import load_dataset

    candidates: List[Dict[str, Optional[str]]] = [{
        "name": dataset_name,
        "config": dataset_config,
        "split": split,
    }]

    if fallback_datasets:
        candidates.extend(fallback_datasets)
    else:
        candidates.append({
            "name": "wikitext",
            "config": "wikitext-2-raw-v1",
            "split": "train",
        })

    deduped: List[Dict[str, Optional[str]]] = []
    seen = set()
    for item in candidates:
        name = item.get("name")
        config = item.get("config")
        item_split = item.get("split") or "train"
        key = (name, config, item_split)
        if not name or key in seen:
            continue
        seen.add(key)
        deduped.append({"name": name, "config": config, "split": item_split})

    logger.info(
        "Loading calibration texts with fallback chain: %s",
        " -> ".join(f"{c['name']}[{c['split']}]" for c in deduped),
    )

    collected: List[str] = []
    source_hits: List[str] = []
    errors: List[str] = []

    for candidate in deduped:
        name = candidate["name"]
        config = candidate["config"]
        item_split = candidate["split"] or "train"

        try:
            load_kwargs: Dict[str, Any] = {
                "split": item_split,
                "streaming": streaming,
            }
            if hf_token:
                load_kwargs["token"] = hf_token

            logger.info("Trying calibration dataset: %s", name)

            # ── Local file / directory / glob support ───────────────────
            # Supports three local data modes:
            #   1. Single file:  "dataset/sangraha_verified/hin.parquet"
            #   2. Directory:    "dataset/sangraha_verified"
            #                    (loads ALL .parquet files inside)
            #   3. Glob pattern: "dataset/sangraha_verified/*.parquet"
            import glob as _glob

            _is_local_file = (
                any(name.endswith(ext) for ext in _LOCAL_FILE_EXTENSIONS)
                or os.path.isfile(name)
            )
            _is_local_dir = os.path.isdir(name)
            _is_glob = ("*" in name or "?" in name)

            if _is_local_dir:
                # Directory mode → load all parquet files inside
                parquet_files = sorted(_glob.glob(os.path.join(name, "*.parquet")))
                if not parquet_files:
                    logger.warning("Directory %s has no .parquet files, skipping", name)
                    continue
                logger.info(
                    "Loading from local directory: %s (%d parquet files: %s)",
                    name, len(parquet_files),
                    ", ".join(os.path.basename(f) for f in parquet_files),
                )
                ds = load_dataset("parquet", data_files=parquet_files, split="train")
            elif _is_glob:
                # Glob mode → expand pattern
                parquet_files = sorted(_glob.glob(name))
                if not parquet_files:
                    logger.warning("Glob pattern %s matched no files, skipping", name)
                    continue
                logger.info(
                    "Loading from glob pattern: %s (%d files)", name, len(parquet_files),
                )
                ds = load_dataset("parquet", data_files=parquet_files, split="train")
            elif _is_local_file and os.path.exists(name):
                fmt = "parquet" if name.endswith(".parquet") else "json"
                logger.info("Loading from local file: %s (format=%s)", name, fmt)
                ds = load_dataset(fmt, data_files=name, split="train")
            elif config is not None:
                ds = load_dataset(name, config, **load_kwargs)
            else:
                ds = load_dataset(name, **load_kwargs)

            if hasattr(ds, "shuffle") and not streaming:
                ds = ds.shuffle(seed=seed)

            added_here = 0
            for sample in ds:
                if len(collected) >= num_samples:
                    break

                for raw_text in _iter_text_values(sample, preferred_text_column=text_column):
                    normalized = " ".join(raw_text.split())
                    if len(normalized) < min_text_length:
                        continue
                    collected.append(normalized)
                    added_here += 1
                    break

            if added_here > 0:
                source_hits.append(name)
                logger.info("Collected %d samples from %s", added_here, name)
            else:
                logger.warning("Dataset %s loaded but yielded no usable samples", name)

            if len(collected) >= num_samples:
                break

        except Exception as exc:
            msg = f"{name}: {exc}"
            errors.append(msg)
            logger.warning("Could not load %s: %s", name, exc)

    if not collected:
        tried = " -> ".join(f"{c['name']}[{c['split']}]" for c in deduped)
        details = "; ".join(errors) if errors else "no dataset errors captured"
        raise RuntimeError(
            f"No calibration samples could be collected. Tried: {tried}. Details: {details}"
        )

    if len(collected) < num_samples:
        repeats = (num_samples + len(collected) - 1) // len(collected)
        collected = (collected * repeats)[:num_samples]
    else:
        collected = collected[:num_samples]

    return collected, source_hits


def load_calibration_data(
    tokenizer,
    dataset_name: str = "wikitext",
    dataset_config: Optional[str] = "wikitext-2-raw-v1",
    split: str = "train",
    num_samples: int = 128,
    seq_length: int = 2048,
    seed: int = 42,
    text_column: str = "text",
    streaming: bool = False,
    min_text_length: int = 200,
    fallback_datasets: Optional[List[Dict[str, Optional[str]]]] = None,
    hf_token: Optional[str] = None,
) -> List[Dict[str, torch.Tensor]]:
    """
    Load and tokenize calibration data for quantization.

    Returns a list of dicts, each with 'input_ids' and 'attention_mask' tensors.
    This is the common format expected by llm-compressor and auto-gptq
    calibration routines.

    Args:
        tokenizer: HuggingFace tokenizer (must have pad_token_id set).
        dataset_name: HuggingFace dataset name.
        dataset_config: Dataset configuration (e.g., 'wikitext-2-raw-v1').
        split: Dataset split to use.
        num_samples: Number of calibration samples to collect.
        seq_length: Max sequence length for tokenization.
        seed: Random seed for reproducibility.
        text_column: Name of the text column in the dataset.
        streaming: If True, stream the dataset instead of loading all at once.
        min_text_length: Minimum text length (characters) to include a sample.

    Returns:
        List of dicts with 'input_ids' and 'attention_mask' tensors.
    """
    logger.info(
        f"Loading calibration data: {dataset_name}/{dataset_config} "
        f"({num_samples} samples, seq_len={seq_length})"
    )

    texts, source_hits = load_calibration_texts(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        split=split,
        num_samples=num_samples,
        seed=seed,
        text_column=text_column,
        streaming=streaming,
        min_text_length=min_text_length,
        fallback_datasets=fallback_datasets,
        hf_token=hf_token,
    )

    if source_hits:
        logger.info("Calibration text sources: %s", ", ".join(source_hits))

    # Tokenize
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    calibration_data: List[Dict[str, torch.Tensor]] = []
    for text in texts:
        tokens = tokenizer(
            text,
            return_tensors="pt",
            max_length=seq_length,
            truncation=True,
            padding="max_length",
        )
        calibration_data.append({
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        })

    logger.info(f"  Tokenized {len(calibration_data)} sequences (seq_len={seq_length})")
    return calibration_data
