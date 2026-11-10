# JuMCNet FP32 Deployment Optimization

Only the five existing JuMCNet training seeds, 1111, 2222, 3333, 4444, and 5555, were tested, using the same weights as in the original evaluation. No training, pruning, quantization, input-size reduction, or classification-threshold changes were performed.

## Four Implementations

1. Original implementation: the original FP32 ONNX model executed with `session.run`, remeasured as the reference for this batch.
2. Method 2: the same original ONNX model with preallocated input/output buffers reused through ONNX Runtime I/O Binding; timing includes input copying.
3. Method 3: nine standard unmasked self-attention subgraphs are replaced with `com.microsoft.Attention`, retaining the original Q/K/V weights and biases, attention-head counts, scaling factors, and output projections. The original invocation method is used.
4. Methods 2+3: the attention-fused model combined with buffer reuse.

All weight tensors are byte-identical before and after fusion. The static ONNX graph decreases from 637 to 349 nodes, with no change in parameter count; fewer nodes do not imply a proportional reduction in FLOPs. The fused model targets this ONNX Runtime CPU environment; compatibility with every ONNX inference framework is not claimed.

Operator profiling is used for diagnosis, and latency measured with profiling enabled is excluded from performance results. The four implementations are evaluated in the order saved in advance in `test_plan.json`; execution configurations and seeds are not selected based on classification accuracy.

## Fixed Conditions and Acceptance Criteria

- Raspberry Pi 5, CPUExecutionProvider, FP32, batch=1, 224×224.
- intra-op=4, inter-op=1, sequential execution, all graph optimizations enabled, thread spinning disabled, and the original system ondemand frequency governor retained.
- The same complete test set of 2325 images is used, with RGB conversion, BICUBIC resizing of the shorter edge to 256, center cropping to 224, and ImageNet normalization; preprocessing code is unchanged.
- Each configuration runs in a separate process after at least 15 seconds of cooling and a temperature of no more than 50°C, followed by 100 warmup iterations.
- Each configuration records predictions, logits, and total latency from image loading to class output for all 2325 images; inference-only timing separately cycles through 24 preloaded inputs for 3×500 iterations.
- Per-image predictions must match the original Raspberry Pi FP32 records; Accuracy, Macro-F1, and confusion matrices must be identical. Both atol and rtol for logits are 2e-4, and the maximum absolute error and bitwise equality are recorded.
- Undervoltage and throttling flags are checked throughout. Power/energy consumption, camera operation, and production-line operation are not measured.
- Peak RSS is the high-water mark of the entire independent Python process. The shared script for these four configurations removes some resident reference arrays; memory should therefore only be compared among these four configurations, and the entire RSS difference from the earlier table of 55 groups cannot be attributed to the model.

## Storage and Cleanup

The Raspberry Pi retains only the original and fused models for the current seed, totaling approximately 10 MiB. A single copy of the test images is reused, and at least 2 GiB of free space is maintained. The seed's temporary directory is cleared only after its records are transferred to the computer and their SHA-256 hashes are verified.

`runs/seed<number>/<implementation>/` contains per-image predictions, logits, 1500 timing measurements, temperature and frequency logs, summary JSON files, and transfer-verification records. An independent script produces `verification.json` by recalculating classification metrics and latency statistics rather than relying directly on device-generated summaries.

These experiments compare execution optimizations of this model. If the optimized results are used for the main comparison between different models, a separate comparison should provide equivalent optimization opportunities and measurement conditions.

Official operator definition: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftattention
Official I/O Binding documentation: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html
