# Stories110M — ANE Training + Inference

Training and inference for a 109M-parameter Llama2-architecture transformer (Stories110M) on Apple's Neural Engine using private ANE APIs.

## Model

| Parameter | Value |
|-----------|-------|
| Architecture | Llama2 (RMSNorm + SwiGLU + RoPE + GQA) |
| Dimensions | dim=768, hidden=2048, heads=12, seq=256 |
| Layers | 12 |
| Vocabulary | 32,000 (BPE, llama2.c tokenizer) |
| Parameters | 109.53M (84.95M transformer + 24.58M embedding) |
| ANE kernels | 72 per compile (60 weight-bearing + 12 weight-free sdpaBwd2) |

## Performance

Per-step timing breakdown on M4:

| Component | Time (ms/step) |
|-----------|---------------|
| ANE eval | 9.6 |
| IO (fp16 conversion) | 4.1 |
| Classifier (cblas) | 9.1 |
| Cross-entropy + residuals | 14.4 |
| RMSNorm | 0.1 |
| **Total** | **107 ms/step** |

## Files

| File | Description |
|------|-------------|
| `train_large.m` | Main training loop — 12-layer forward/backward, checkpoint, exec() restart |
| `inference.m` | ANE-accelerated autoregressive inference with KV-cache |
| `stories_config.h` | Model config, structs, memory allocation helpers |
| `stories_io.h` | IOSurface I/O, NEON fp16 conversion, kernel compile/eval wrappers |
| `stories_mil.h` | MIL program generators for all 6 ANE kernel types |
| `stories_cpu_ops.h` | vDSP-vectorized RMSNorm, cross-entropy loss, Adam optimizer, embedding ops |
| `dashboard.py` | TUI dashboard — loss curves, power/CPU/memory graphs, live text generation |
| `tokenize.py` | Extract pretokenized TinyStories data from zip |
| `test_dashboard.py` | Python unit tests for dashboard functions (59 tests) |
| `test_*.m` | ANE kernel tests and hardware probes (10 files) |
| `Makefile` | Build targets for training and probes |

## Training Pipeline

1. **Forward pass**: Each layer runs `fwdAttn` (QKV + SDPA + Wo) and `fwdFFN` (W1 + SiLU(W3) + W2) on ANE via MIL-compiled kernels. Final RMSNorm + classifier matmul on CPU (cblas).

2. **Backward pass**: Reverse layer order. `ffnBwd`, `sdpaBwd1`, `sdpaBwd2`, `qkvBwd` on ANE. Weight gradients (dW) via async `cblas_sgemm` on CPU. RMSNorm backward via vDSP.

3. **Compile budget**: ANE has a ~119 compile limit per process. With 72 kernels per batch, we run 10 accumulation steps then `exec()` restart with checkpoint resume.

4. **Data**: Real TinyStories text (20M tokens), mmap'd uint16 token IDs, random position sampling per step.

## Usage

### Training

```bash
# 1. Extract tokenized data (needs ~/tiny_stories_data_pretokenized.zip)
python3 tokenize.py

# 2. Build and train
make train_large
./train_large                     # start fresh
./train_large --resume            # resume from checkpoint

# 3. Monitor with dashboard
pip install blessed psutil numpy
python3 dashboard.py --resume     # spawns train_large, shows TUI
python3 dashboard.py --infinite   # train indefinitely
python3 dashboard.py --no-powermetrics  # skip sudo for power monitoring
```

### Inference

```bash
make inference
./inference                          # ANE-accelerated, 256 tokens
./inference --tokens 512 --temp 0.6  # longer, less random
./inference --temp 0                 # greedy decoding
./inference --top-p 0.95             # nucleus sampling
./inference --cpu                    # CPU-only (Accelerate BLAS) for comparison
./inference --ckpt ane_stories110M_ckpt.bin  # use trained checkpoint
```

The inference engine compiles 49 ANE kernels once at startup (4 per layer + 1 classifier):
- **Fused QKV**: single ANE dispatch for Q, K, V projections
- **Fused FFN up**: W1 + W3 + SiLU + gate in one kernel
- **Wo, W2**: separate projection kernels
- KV-cache on CPU for O(1) per-token attention
- CPU handles RMSNorm, RoPE, causal attention, sampling

### Dashboard Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Restart training with --resume |
| `g` | Force text generation from current checkpoint |
| `p` | Toggle auto-scroll on logs |
| Up/Down | Scroll logs |

## Tests

### Python Tests (cross-platform)

```bash
pip install pytest numpy
python3 -m pytest test_dashboard.py -v
```

59 tests covering: `rmsnorm`, `softmax`, `braille_chart`, `parse_line`, `parse_powermetrics_text`, `Tokenizer.decode`, regex patterns, RoPE vectorization correctness, and edge cases.

### ANE Probe Tests (macOS Apple Silicon only)

```bash
make probes
./test_weight_reload    # Weight blob reload without recompilation
./test_perf_stats       # ANE performance statistics API
./test_qos_sweep        # QoS impact on latency/frequency
./test_ane_advanced      # SharedEvents, VirtualClient, ChainingRequest APIs
```

Additional kernel tests (built individually):
- `test_ane_causal_attn.m` — Decomposed causal attention (Q@K^T -> mask+softmax -> scores@V)
- `test_full_fused.m` — Full fused forward pass (QKV convs + matmul + softmax + output)
- `test_ane_sdpa5.m` — Scaled dot-product attention with 5 operations
- `test_conv_attn3.m` — Convolution-based attention with 3 operations
- `test_fused_qkv.m` — Fused QKV projection kernels
- `test_fused_bwd.m` — Backward pass kernels for FFN and SDPA

## Key Techniques

- **NEON vectorized fp16<->fp32**: ARM NEON intrinsics for fast IOSurface data transfer
- **vDSP cross-entropy**: `vDSP_mtrans` + `vvexpf` + `vDSP_sve` — 8x faster than scalar
- **Async weight gradients**: `cblas_sgemm` dispatched to background queue, overlapped with ANE
- **Vectorized RoPE** (dashboard): NumPy `np.outer` + `np.cos`/`np.sin` replacing nested Python loops
- **SDPA causal mask workaround**: ANE hardware ignores `attn_mask`, so attention is decomposed into Q@K^T (ANE conv) -> mask+softmax (ANE) -> scores@V (ANE conv)
