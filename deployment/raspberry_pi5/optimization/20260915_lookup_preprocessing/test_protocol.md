# Operator and Preprocessing Optimization of Method 3

This round examines only JuMCNet deployment execution. The weights are the existing checkpoints for seeds 1111, 2222, 3333, 4444, and 5555; no training or model selection based on test scores is performed.

## Preliminary Validation

From the fixed validation split, 8 images per class, 48 in total, are selected by SHA-256 order. Images are not selected based on predictions and do not overlap with the test set.

1. The attention-fused graph is profiled operator by operator. The first 12 startup runs are excluded and the following 48 runs are summarized; profiling results are not reported as formal latency measurements.
2. The 18 residual additions and their subsequent normalization operations are replaced with SkipLayerNormalization. Its fourth output retains the residual sum needed by later branches. All weight tensors remain byte-identical. The original runtime already fuses 3 of these locations, so the additional fusion covers 15 locations.
3. The candidate and Method 3 are tested in alternating order on validation samples. Additional graph fusion improves inference-only latency by approximately 0.3%, below the predefined 0.5% threshold for full testing, so Method 3 is not replaced. This candidate undergoes preliminary validation only and is not described as a formal five-seed deployment result.
4. Original, in-place, and lookup-table normalization are compared. The lookup-table version is fastest in preliminary screening. For each channel value from 0–255, table entries use the original FP32 sequence of division, subtraction, and division; resizing and cropping rules are unchanged. Inputs for all 1155 validation images and 2325 test images are byte-identical.

## Formal Comparison

For each seed, original and lookup-table preprocessing are compared using the same Method 3 model, giving 10 groups. Four CPU threads, inter-op=1, disabled spinning, sequential execution, FP32, batch=1, 224×224, CPUExecutionProvider, and the ondemand frequency governor remain unchanged.

Each separate process starts after at least 15 seconds of cooling and a temperature of no more than 50°C, performs 100 warmup iterations, and evaluates all 2325 test images. Inference-only timing uses the same 24 preloaded inputs as in the preceding round for 3×500 iterations. Inference-only timing excludes image loading; total latency includes loading, decoding, preprocessing, inference, softmax, and class output. Per-image predictions, Accuracy, Macro-F1, and confusion matrices must match, and output differences are checked. Performance is reported as means and sample standard deviations across five training seeds.

Both configurations use the same model file, so inference-only differences cannot be interpreted as model-operator acceleration. Preprocessing and total latency are the primary results of this formal comparison. Peak RSS is compared only between these two configurations under the common script used in this round.

## Storage

Raw logs, per-image predictions, logits, 1500 timing measurements, and temperature records are stored in runs. After each seed finishes, records are transferred and their hashes verified before the temporary Raspberry Pi model is cleared. Only one Method 3 model of approximately 4.75 MiB is transferred at a time.

Comparison models are not modified in this round.
