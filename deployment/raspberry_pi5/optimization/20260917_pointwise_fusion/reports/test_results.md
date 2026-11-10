# JuMCNet: Full Five-Seed Testing of Fusion at the First Two 1×1 Convolution–SiLU Pairs

The verified FP32 implementation is retained without retraining or changes to weights, input preprocessing, or thread settings. Each seed is fully evaluated on the same 2,325 Jujube12000 test images, yielding 11,625 candidate predictions in total; the number of independent images remains 2,325.

| Seed | Accuracy (%) | Macro-F1 (%) | Previous inference only ms | New fused inference only ms | Previous total latency ms | New fused total latency ms |
|---|---:|---:|---:|---:|---:|---:|
| 1111 | 94.6667 | 94.7323 | 27.008 | 24.604 | 32.121 | 29.591 |
| 2222 | 94.1935 | 94.2334 | 27.058 | 24.851 | 32.095 | 29.587 |
| 3333 | 94.6237 | 94.6618 | 26.973 | 24.425 | 32.067 | 29.376 |
| 4444 | 94.3226 | 94.3706 | 27.434 | 24.764 | 32.533 | 29.807 |
| 5555 | 94.6667 | 94.7175 | 26.920 | 24.441 | 31.924 | 29.435 |

| Metric (five-seed mean ± sample standard deviation) | Previous optimized version in the same batch | Additional 1×1 fusion |
|---|---:|---:|
| Accuracy (%) | 94.4946 ± 0.2214 | 94.4946 ± 0.2214 |
| Macro-F1 (%) | 94.5431 ± 0.2269 | 94.5431 ± 0.2269 |
| Inference-only latency (ms) | 27.0787 ± 0.2050 | 24.6170 ± 0.1898 |
| Total latency (ms) | 32.1481 ± 0.2286 | 29.5591 ± 0.1674 |
| Preprocessing latency (ms) | 4.4493 ± 0.0809 | 4.3979 ± 0.0101 |
| Whole-process peak RSS (MiB) | 119.5281 ± 0.2741 | 119.4344 ± 0.4258 |
| ONNX file (MiB) | 4.7473 ± 0.0000 | 4.7465 ± 0.0000 |

Relative to the reference measured in the same batch, inference-only latency decreases by 9.09% and total latency by 8.05%. Per-image classes match for all five seeds, and Accuracy, Macro-F1, and confusion matrices are identical. Logits have small floating-point differences, with a maximum absolute difference of 1.6450882e-05; all pass the predefined atol=rtol=2e-4 tolerance.

Timing: each fresh process loads one model, with four ORT inference threads, one inter-op thread, and spinning disabled. After 100 warmup iterations, fixed validation tensors are used for 500 inference-only measurements, and the full pipeline is timed image by image on the entire test set; test order alternates by seed. Inference-only timing excludes loading and preprocessing. Total latency includes image reading and decoding, resizing and cropping, lookup-table normalization and tensor layout conversion, inference, and classification postprocessing; it excludes camera acquisition and network transfer. Differences use the reference from this batch and cannot be attributed rigorously by comparison with historical latency from other batches.

RSS is the peak of the entire process, not model-only memory. Native-operator dynamic libraries are required in addition to the ONNX file. Device records at the start and end report no undervoltage or throttling. After each seed is tested, the local archive is verified before the two models and reference arrays in the Raspberry Pi's active directory for this round are deleted; shared test images and the runtime environment are retained.
