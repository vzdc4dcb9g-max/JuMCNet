# JuMCNet: Five-Seed Verification of Equivalent Crop-Aware Resizing

The existing weights for 1111, 2222, 3333, 4444, and 5555 are used without retraining. All model graphs use the preceding fusion, feature-layout, and tiling optimizations, together with the FC1 and DW8 implementations.

- `baseline`: the latest combined inference graph, retaining Pillow resizing and center cropping, native lookup-table normalization, and tensor layout conversion.
- `resize`: the same model graph and weights, with equivalent bicubic resizing that computes only the final crop region and its interpolation neighborhood.

Each configuration for each seed is tested twice in independent processes, in ABBA or BAAB order. Each run uses 100 warmup iterations and 48 fixed validation inputs for 500 inference-only measurements, then evaluates all 2,325 fixed test images. Full-pipeline latency includes file reading, decoding, preprocessing, model inference, and class output; it excludes model loading, networking, camera acquisition, and writing results to disk. The file cache may already be warm.

## Data Locations

`test_plan.json` stores the test plan and acceptance criteria; `reports/weight_audit.json` stores model-parameter verification results.

`results/seed*/baseline_1`, `baseline_2`, `resize_1`, and `resize_2` store results, per-image timings, predictions, and logits from four independent processes. `baseline_1/input_equivalence.json` records pixel and FP32-input checks for the seed's complete test set. `paired_consistency.json` compares logits across the four runs.

After completion, `results/seed*/baseline/result.json` and `resize/result.json` aggregate the two measurements. For each configuration, the two measurements are first averaged within each seed, then the mean and sample standard deviation across the five seeds are calculated. Repeated runs do not increase the number of independent test images.

`reports/summary.json` stores the five-seed summary and changes from crop-aware resizing relative to the reference in the same batch; `reports/all_runs.csv` retains mean timings for each independent run; `reports/paired_seed_results.csv` stores paired results within each seed. `reports/independent_result_audit.json` records recalculated classification metrics and checks of the original files.

`models` stores the five final model sets; `scripts` stores preprocessing implementations, runtime wrappers, and dynamic libraries; `reference_sources` stores operator source code and compilation-environment references. The reference directory also contains files from preceding experiments; the libraries actually used in this experiment are specified in `scripts/runtime.py` and `reports/runtime_frozen.json`.

## Verification and Interpretation

Model parameters are compared byte by byte, and image files are verified with SHA-256. RGB pixels, FP32 inputs, and logits must be elementwise identical before and after crop-aware resizing. Comparisons of logits from the latest graph against the original FP32 graph and the preceding five-seed graphs use atol=rtol=2e-4, with identical predicted classes required.

Both preprocessing methods use the same model graph, so inference-only timing differences cannot be attributed to crop-aware resizing. Adoption requires a decrease in mean total latency across the five seeds and decreases in the paired means of at least four seeds; no statistical-significance claim is made. Process peak RSS includes Python, runtime libraries, and test arrays, and the two peaks are averaged first; it does not represent model-only memory.

The Raspberry Pi temporarily stores only the current seed's models and verification references. After each seed finishes, results are transferred back and their hashes checked before that seed's temporary inputs are deleted.
