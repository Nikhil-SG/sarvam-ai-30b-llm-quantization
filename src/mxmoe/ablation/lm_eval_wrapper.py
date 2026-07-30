#!/usr/bin/env python3
"""
DEPRECATED — lm-eval wrapper is no longer needed.

Module 3 evaluation now runs in-process via ablation_study.py which loads
the model directly with ``device_map="auto"`` and calls
``lm_eval.simple_evaluate()`` through the research pipeline's
``PerplexityEvaluator`` and ``BenchmarkRunner``.

The model is loaded in its native quantized format (FP8 + GPTQ compressed-
tensors config_groups). No decompression, no subprocess, no vLLM needed.

This file is kept as a stub to avoid import errors in any references.
"""
raise DeprecationWarning(
    "lm_eval_wrapper is deprecated. Use ablation_study.EvaluationRunner instead. "
    "It loads the model in-process in native quantized format and evaluates directly."
)
