# ANE Training — Backpropagation on Apple Neural Engine

Training neural networks directly on Apple's Neural Engine (ANE) via reverse-engineered private APIs. No CoreML training APIs, no Metal, no GPU — pure ANE compute.

## Overview

A from-scratch transformer training implementation (forward + backward pass) running entirely on the ANE in Apple Silicon. The ANE is a 15.8 TFLOPS (M4) inference accelerator that Apple does not expose for training. This project reverse-engineers the `_ANEClient` / `_ANECompiler` private APIs and the MIL (Model Intermediate Language) format to run custom compute graphs — including backpropagation — directly on ANE hardware.

**Results on M4 (single transformer layer, dim=768, seq=512):**
- 9.3 ms/step, 11.2% ANE utilization (1.78 TFLOPS sustained)
- 6 ANE kernel dispatches per training step
- Forward and backward dx passes on ANE, dW gradients on CPU (Accelerate cblas)
- Adam optimizer, gradient accumulation, checkpoint/resume

## Architecture

The training loop uses 6 ANE kernels per step:

| Kernel | Function | Weights |
|--------|----------|---------|
| `kFwdAttn` | RMSNorm + QKV projection + SDPA + output projection | Wq, Wk, Wv, Wo, rms1, mask |
| `kFwdFFN` | RMSNorm + SwiGLU FFN (W1, W3, SiLU, W2) | W1, W2, W3, rms2 |
| `kFFNBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) | W2^T, W1^T, W3^T |
| `kSdpaBwd1` | Wo^T + SDPA backward part 1 (dV, probs, dp) | Wo^T, mask |
| `kSdpaBwd2` | SDPA backward part 2 (softmax grad, dQ, dK) | — |
| `kQKVb` | QKV backward (Wq^T + Wk^T + Wv^T -> dx) | Wq^T, Wk^T, Wv^T |

CPU handles: RMSNorm backward, residual connections, loss computation, dW gradient accumulation (cblas_sgemm), Adam optimizer updates.

### Key Optimizations

- **Channel-first CPU layout** — matches ANE IOSurface `[1,C,1,S]` format, eliminates all transpose overhead
- **vDSP vectorized RMSNorm** — 10x faster than naive (6.7ms -> 0.7ms)
- **GCD async cblas overlap** — dW gradient sgemms run in parallel with ANE evals on a serial dispatch queue
- **Deferred cblas wait** — wait pushed into next step's forward pass for maximum overlap
- **ANE RMSNorm fusion** — RMSNorm folded into forward kernels as MIL ops (reduce_sum + pow + mul)
- **Wo^T fusion** — output projection backward merged into SDPA backward kernel
- **Forward taps** — Q, K, V, attention scores, hidden states exposed via concat outputs, avoiding CPU recompute
- **exec() restart** — bypasses ~119 ANE compile limit per process

## Repository Structure

```
ane/
├── api_exploration.m           # Initial ANE API discovery
├── inmem_basic.m               # In-memory MIL compilation proof-of-concept
├── inmem_bench.m               # ANE dispatch latency benchmarks
├── inmem_peak.m                # Peak TFLOPS measurement (2048x2048 matmul)
├── sram_bench.m                # ANE SRAM bandwidth probing
├── sram_probe.m                # SRAM size/layout exploration
└── training/
    ├── train_large.m           # Main: 12-layer Stories110M training (optimized)
    ├── train.m                 # Minimal training loop (early prototype)
    ├── tiny_train.m            # 2-layer tiny model training
    ├── ane_runtime.h           # ANE private API wrapper (compile, eval, IOSurface)
    ├── ane_mil_gen.h           # MIL program generation helpers
    ├── model.h                 # Model weight initialization and blob builders
    ├── forward.h               # Forward pass MIL generators
    ├── backward.h              # Backward pass MIL generators
    ├── stories_config.h        # Stories110M model config, structs, alloc helpers
    ├── stories_io.h            # IOSurface I/O, NEON fp16 conversion, kernel compile/eval
    ├── stories_mil.h           # MIL program generators for all 6 ANE kernel types
    ├── stories_cpu_ops.h       # vDSP-vectorized RMSNorm, cross-entropy, Adam, embedding
    ├── dashboard.py            # TUI dashboard — loss curves, power graphs, text generation
    ├── tokenize.py             # Extract pretokenized TinyStories data
    ├── test_dashboard.py       # Python unit tests (59 tests, pytest)
    ├── test_ane_causal_attn.m  # Decomposed causal attention test
    ├── test_full_fused.m       # Full fused forward pass test
    ├── test_ane_sdpa5.m        # SDPA 5-op test
    ├── test_conv_attn3.m       # Convolution-based attention test
    ├── test_fused_qkv.m        # Fused QKV projection test
    ├── test_fused_bwd.m        # Backward pass kernel test
    ├── test_ane_advanced.m     # Advanced ANE API probe (SharedEvents, VirtualClient)
    ├── test_perf_stats.m       # ANE performance statistics probe
    ├── test_weight_reload.m    # Weight reload without recompilation test
    ├── test_qos_sweep.m        # QoS sweep for latency/frequency analysis
    └── Makefile
```

## Getting Started

### Requirements

- macOS 15+ on Apple Silicon (tested on M4)
- Xcode Command Line Tools (`xcode-select --install`)
- Python 3.9+ with `numpy`, `blessed`, and optionally `psutil`

### Build and Train

```bash
cd training

# Extract tokenized training data (needs ~/tiny_stories_data_pretokenized.zip)
python3 tokenize.py

# Build and run
make train_large
./train_large                     # start fresh
./train_large --resume            # resume from checkpoint
./train_large --resume --steps 50000
```

### Dashboard

```bash
pip install blessed psutil numpy
python3 dashboard.py              # spawns train_large and monitors it
python3 dashboard.py --resume     # resume training
python3 dashboard.py --infinite   # train indefinitely
python3 dashboard.py --no-powermetrics  # skip sudo powermetrics
```

Dashboard keybindings: `q` quit, `r` restart with resume, `g` force text generation, `p` toggle auto-scroll, arrows to scroll logs.

### Run Tests

```bash
# Python tests (runs anywhere)
pip install pytest
cd training && python3 -m pytest test_dashboard.py -v

# ANE probe tests (macOS Apple Silicon only)
make probes
./test_weight_reload
./test_perf_stats
./test_qos_sweep
./test_ane_advanced
```

No external dependencies for the Objective-C code. Uses only system frameworks + private ANE APIs resolved at runtime via `objc_msgSend`.

## How It Works

1. **MIL generation** — Objective-C code constructs MIL program text at runtime, specifying convolutions (for linear layers), matmul (for attention), softmax, and element-wise ops
2. **In-memory compilation** — `_ANEInMemoryModelDescriptor` compiles MIL text + weight blobs directly to ANE programs, no disk `.mlmodelc` needed
3. **IOSurface I/O** — Input/output tensors passed via IOSurface shared memory in `[1, channels, 1, spatial]` format (fp16)
4. **Weight embedding** — Weights baked into ANE programs as BLOBFILE constants; recompiled each batch when weights change
5. **Gradient flow** — Forward taps expose intermediates needed for backward; backward kernels compute dx (input gradients) on ANE; dW (weight gradients) computed on CPU via cblas

## Performance History

| Optimization | ms/step | ANE utilization |
|---|---|---|
| Baseline (vDSP transpose) | 33.5 | 3.1% |
| Channel-first layout | 20.3 | 5.2% |
| vDSP vectorized RMSNorm | 14.2 | 7.4% |
| GCD async cblas overlap | 11.4 | 9.2% |
| ANE RMSNorm fusion | 11.4 | 9.2% |
| Wo^T fusion (7->6 kernels) | 11.4 | 9.2% |
| Deferred cblas wait | **9.3** | **11.2%** |

## Limitations

- **SDPA causal masking** — ANE hardware ignores `attn_mask` in SDPA ops; causal attention is decomposed into separate Q@K^T (ANE) -> mask+softmax (ANE via add+softmax) -> scores@V (ANE)
- **~119 compile limit** — ANE compiler leaks resources; worked around via `exec()` restart with checkpoint
- **Single-layer prototype** — The `train.m` prototype trains one layer; the full `train_large.m` handles 12 layers with compile budget management
- **Data** — Uses pretokenized TinyStories (20M tokens, uint16 BPE)

## Disclaimer

This project is independent research into Apple Neural Engine architecture. It uses undocumented APIs discovered through runtime introspection for research and educational purposes under fair use and interoperability provisions (see *Sega v. Accolade*, 1992; DMCA 1201(f)). No Apple proprietary code or binaries are included in this repository. This project is not affiliated with or endorsed by Apple Inc. Use at your own risk.

## License

MIT — see [LICENSE](LICENSE)
