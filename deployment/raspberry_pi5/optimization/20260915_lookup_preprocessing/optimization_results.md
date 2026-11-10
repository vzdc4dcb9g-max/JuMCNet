# Further Optimization of Method 3

Means ± sample standard deviations across five existing training seeds; 4 threads, disabled spinning, FP32 precision, model files, and input size remain unchanged.

| Configuration | Accuracy (%) | Macro-F1 (%) | Inference only (ms/image) | Preprocessing (ms/image) | Total latency (ms/image) | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Method 3 + original preprocessing | 94.49 ± 0.22 | 94.54 ± 0.23 | 28.28 ± 0.18 | 5.63 ± 0.05 | 34.76 ± 0.16 | 105.18 ± 0.22 |
| Method 3 + lookup-table preprocessing | 94.49 ± 0.22 | 94.54 ± 0.23 | 28.23 ± 0.13 | 4.72 ± 0.01 | 33.68 ± 0.14 | 103.57 ± 0.27 |

Lookup-table preprocessing precomputes the original FP32 normalization result for each channel value from 0–255, replacing repeated per-pixel floating-point operations. RGB decoding, PIL BICUBIC resizing, center cropping, and output layout follow the original rules. The processed inputs are byte-identical for all 1155 validation images and 2325 test images.

Per-image predictions and outputs from all formal tests match the original records. Because the model graph and weights are identical, small differences in inference-only latency should be treated as measurement variation, not acceleration of model computation. Preprocessing and total latency are the primary evaluation measures.

- Preprocessing latency decreases by 0.906 ms (16.10%).
- Total latency decreases by 1.088 ms (3.13%).

## Preliminary Validation of the Operator-Fusion Candidate

Additional fusion of residual addition and normalization reduces inference-only latency from 28.120 to 28.034 ms in preliminary validation, a reduction of 0.304%. This is below the predefined preliminary threshold of 0.5%, so the candidate does not enter the formal five-seed test or replace the Method 3 model. This value is a preliminary validation result, not a formal five-seed result.

This round evaluates only JuMCNet's own optimization. No models were trained, CPU thread settings or frequency policies changed, or comparison models tested. Peak RSS for the old and new scripts is compared only between the two configurations in this round. Application-level power consumption and a camera pipeline were not measured.
