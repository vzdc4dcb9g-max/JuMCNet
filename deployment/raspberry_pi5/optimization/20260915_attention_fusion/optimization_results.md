# JuMCNet: FP32 Deployment Optimization Results

Means ± sample standard deviations across five training seeds; the original implementation was measured again in the same batch.

| Implementation | Accuracy (%) | Macro-F1 (%) | ONNX (MiB) | Inference only (ms/image) | Total latency including preprocessing (ms/image) | Peak process memory (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| Original implementation (remeasured) | 94.49 ± 0.22 | 94.54 ± 0.23 | 4.82 | 31.18 ± 0.16 | 37.66 ± 0.21 | 107.86 ± 0.28 |
| Method 2: Input/output buffer reuse | 94.49 ± 0.22 | 94.54 ± 0.23 | 4.82 | 31.55 ± 0.32 | 37.84 ± 0.21 | 108.21 ± 0.33 |
| Method 3: Attention operator fusion | 94.49 ± 0.22 | 94.54 ± 0.23 | 4.75 | 28.36 ± 0.19 | 34.87 ± 0.25 | 105.17 ± 0.64 |
| Methods 2+3: Buffer reuse and fusion | 94.49 ± 0.22 | 94.54 ± 0.23 | 4.75 | 28.46 ± 0.08 | 34.90 ± 0.07 | 105.78 ± 0.01 |

## Changes Relative to the Original Implementation

- Method 2: Input/output buffer reuse: inference-only latency reduction of -0.368 ms (-1.18%); total latency reduction of -0.177 ms (-0.47%).
- Method 3: Attention operator fusion: inference-only latency reduction of 2.816 ms (9.03%); total latency reduction of 2.793 ms (7.42%).
- Methods 2+3: Buffer reuse and fusion: inference-only latency reduction of 2.726 ms (8.74%); total latency reduction of 2.757 ms (7.32%).

Per-image predictions in all 20 groups matched the original FP32 records; Accuracy, Macro-F1, and confusion matrices were unchanged. Numerical output differences were within the predefined absolute and relative tolerances of 2e-4.

Threads, CPU frequency policy, FP32 precision, weights, image preprocessing, and timing boundaries retained their original settings. Method 2 includes input-copy overhead; Method 3 fuses nine standard self-attention blocks into ONNX Runtime com.microsoft.Attention operators. Comparison models were not tested.

The four configurations used a common comparison script whose resident reference arrays differed from those in the earlier script for 55 groups. Peak RSS should therefore only be compared among these four configurations; differences from the process peaks in the earlier table cannot be attributed directly to model optimization. Fewer graph nodes do not imply proportional reductions in mathematical computation or parameter count.

These results evaluate deployment optimizations of our own model.
