# Module 3 Research Summary

Status: ✓ COMPLETED
Cache tags: fp8, bf16, nf4, gptq, int8
MSE quantizers: int8, fp8, nf4, gptq
Outlier tags: bf16, int8, fp8, nf4, gptq
Prompt count: 4

## Key Findings

- Lowest average weight error is INT8 (6.179e-08), while the highest is GPTQ (1.119e-04).
- The largest single MSE hotspot appears in GPTQ at layer_17_attention.dense (1.443e-04).
- Highest average activation outlier rate is BF16 at 0.1049% across 4 prompts.
- BF16 baseline shows its strongest activation hotspot at model.layers.18.mlp.shared_experts.down_proj (0.2907% outliers), which is the reference point for quantized comparisons.

## MSE Ranking

| Quantizer | Mean MSE | Worst Layer | Worst MSE |
| --- | ---: | --- | ---: |
| INT8 | 6.179e-08 | layer_17_attention.dense | 9.576e-08 |
| FP8 | 4.355e-07 | layer_17_attention.dense | 6.045e-07 |
| NF4 | 5.377e-06 | layer_17_attention.dense | 7.245e-06 |
| GPTQ | 1.119e-04 | layer_17_attention.dense | 1.443e-04 |

## Outlier Ranking

| Tag | Mean Outlier % | Highest-Outlier Layer | Highest-Outlier % |
| --- | ---: | --- | ---: |
| BF16 | 0.1049 | model.layers.18.mlp.shared_experts.down_proj | 0.2907 |
| FP8 | 0.0996 | model.layers.0.attention.dense | 0.2201 |
| INT8 | 0.0987 | model.layers.18.mlp.shared_experts.down_proj | 0.2262 |
| GPTQ | 0.0910 | model.layers.18.mlp.shared_experts.down_proj | 0.2363 |
| NF4 | 0.0826 | model.layers.0.attention.dense | 0.2096 |

## Interpretation

This report combines weight-space distortion and activation-space instability so the quantization tradeoff can be read as both an engineering and model-behavior story.
