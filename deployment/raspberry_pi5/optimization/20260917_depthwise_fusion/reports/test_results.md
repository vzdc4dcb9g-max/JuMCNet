# JuMCNet on Raspberry Pi 5: MNN FP32 and Convolution–SiLU Fusion Results

These tests build on the verified attention fusion and native lookup-table preprocessing. Full-model weights remain unchanged, with FP32 precision, single-image input, 4 inference threads, and an unchanged CPU governor.

## Formal Five-Seed Results

| Configuration | Inference only/ms | Total latency including preprocessing/ms | Accuracy/% | Macro-F1/% |
|---|---:|---:|---:|---:|
| Reference for this round | 28.009 ± 0.149 | 33.141 ± 0.164 | 94.495 ± 0.221 | 94.543 ± 0.227 |
| Depthwise convolution–SiLU fusion at 7 locations | 27.176 ± 0.163 | 32.146 ± 0.191 | 94.495 ± 0.221 | 94.543 ± 0.227 |

Inference-only latency decreases by **2.97%**, and total latency including preprocessing decreases by **3.00%**. The table reports the mean ± sample standard deviation of the mean measurements for the five training seeds. Latencies from different historical measurement batches cannot be subtracted to estimate the benefit of this optimization.

Each of the five seeds is evaluated on 2,325 images, yielding 11,625 predictions in total, with per-image classes matching the corresponding baseline. The test set remains the same set of 2,325 independent images. The fused version has a maximum absolute logit difference of 1.4543533e-05, within the predefined tolerance; its outputs are not bitwise identical. Model parameters remain byte-identical, and the original 9 Attention nodes are retained.

## MNN FP32 Screening

On the 48 fixed validation images for seed5555, mean inference-only latency across three rounds is 27.491 ms, and latency including preprocessing is 32.680 ms; the corresponding ORT reference values in the same round are 27.988 and 33.017 ms. Classes match, but the maximum absolute logit difference is 0.00054311752, and 1 of 288 output values exceeds the predefined allclose(atol=rtol=2e-4) tolerance. MNN therefore does not enter full five-seed testing. This does not establish an observed accuracy decrease or justify a claim of unchanged accuracy on the entire test set.

## Paired Results for Each Seed

| Seed | Reference inference only/ms | Fused inference only/ms | Reference total latency/ms | Fused total latency/ms | Accuracy/% | Macro-F1/% |
|---|---:|---:|---:|---:|---:|---:|
| 1111 | 28.062 | 27.257 | 33.247 | 32.300 | 94.667 | 94.732 |
| 2222 | 28.021 | 27.290 | 33.084 | 32.395 | 94.194 | 94.233 |
| 3333 | 27.807 | 27.001 | 32.954 | 31.954 | 94.624 | 94.662 |
| 4444 | 27.946 | 27.333 | 33.054 | 32.009 | 94.323 | 94.371 |
| 5555 | 28.211 | 27.001 | 33.365 | 32.074 | 94.667 | 94.718 |

## Implementation, Verification, and Scope

Fusion covers only the 7 eligible 3×3 depthwise convolution and SiLU pairs. ARM NEON computes and writes the activation immediately after convolution accumulation, without a separate intermediate convolution-output tensor. The implementation uses glibc FP32 vector exp and the existing ORT thread pool; no quantization, FP16, training, or additional CPU threads are introduced. This is local operator fusion, not fusion of all convolutions or compilation and fusion of the entire network.

Timing includes decoding local image files, RGB conversion, resizing, cropping, lookup-table normalization, layout conversion, inference, and softmax/argmax; it excludes model startup, network transfer, and result-file writing. Actual camera acquisition is outside the measurement scope. Only measured latency is reported; power consumption is not inferred.

The official MNN Python tools wrapper was stopped when it attempted to install an optional logging SDK; that SDK was not installed. The local native _tools.mnnconvert was invoked directly instead, without the optional logging wrapper.

Detailed records are in results/seed*/, results/pilots/, and reports/ under the experiment directory; each seed has per-image CSV files, logits, raw timing measurements, and device-state records. New temporary device weights are cleared seed by seed only after the results and models are archived and verified using SHA256.
