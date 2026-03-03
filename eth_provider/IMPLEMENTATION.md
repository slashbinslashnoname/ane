# ANE Compute Provider — Implementation Spec

MVP implementation: Secure Enclave attestation + Proof of Sampling + x402 payments. Architecture-agnostic, sandboxed execution, multi-tenant. Supports any model architecture — decoder-only, encoder-only, encoder-decoder, MoE, vision, and diffusion.

---

## 1. Design Principles

1. **Architecture-agnostic.** The provider accepts any model architecture — decoder-only transformers (Llama, GPT-2, Mamba), encoder-only (BERT), encoder-decoder (T5, BART), mixture-of-experts (Mixtral, DBRX), vision transformers (ViT, CLIP), and diffusion models (Stable Diffusion). Models are opaque weight blobs identified by hash, routed to the correct inference template by architecture tag.
2. **Isolated.** Each job runs in a sandboxed subprocess with its own memory, temp directory, and ANE kernel set. A crash or malicious model cannot affect other jobs or the host.
3. **Stateless between jobs.** No job state persists after completion. KV-cache, activations, compiled kernels — all freed. The provider is a pure function: `(model, input) → (output, proof)`.
4. **Minimal trust surface.** The provider binary is open-source and deterministically compiled. The Secure Enclave signs every result. The provider operator cannot tamper with outputs without invalidating the SEP attestation.
5. **Extensible.** New architectures are added by dropping in a new inference template + weight adapter. No changes to the gateway, sandbox, proof system, or payment layer.

---

## 2. System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      ANE Provider Node                        │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Gateway (Python)                      │ │
│  │                                                         │ │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐ │ │
│  │  │ x402     │  │ Job      │  │ Proof Assembler       │ │ │
│  │  │ Endpoint │─►│ Scheduler│─►│ (SEP sign + PoSP)     │ │ │
│  │  └──────────┘  └────┬─────┘  └───────────────────────┘ │ │
│  │                     │                                    │ │
│  └─────────────────────┼────────────────────────────────────┘ │
│                        │                                      │
│  ┌─────────────────────▼────────────────────────────────────┐ │
│  │               Sandbox Layer                               │ │
│  │                                                           │ │
│  │   ┌─────────┐  ┌─────────┐  ┌─────────┐                │ │
│  │   │ Job 1   │  │ Job 2   │  │ Job 3   │  ...           │ │
│  │   │         │  │         │  │         │                 │ │
│  │   │ model A │  │ model B │  │ model A │                 │ │
│  │   │ tmpdir/ │  │ tmpdir/ │  │ tmpdir/ │                 │ │
│  │   │ ANE krnl│  │ ANE krnl│  │ ANE krnl│                 │ │
│  │   └────┬────┘  └────┬────┘  └────┬────┘                │ │
│  │        │             │            │                      │ │
│  └────────┼─────────────┼────────────┼──────────────────────┘ │
│           │             │            │                         │
│  ┌────────▼─────────────▼────────────▼──────────────────────┐ │
│  │                ANE Hardware                               │ │
│  │   _ANEInMemoryModel compile / eval via private API        │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Secure Enclave Processor                                 │ │
│  │  (hardware root of trust, per-result signing)             │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                               │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  Model Store                                              │ │
│  │  ~/.ane_provider/models/<hash>/                            │ │
│  │    weights.bin   (original weights, immutable)            │ │
│  │    manifest.json (dims, hash, format)                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Model Abstraction Layer

The current codebase hardcodes Stories110M dimensions everywhere (`#define DIM 768`, etc.). The provider must support any model architecture. Instead of rewriting the entire C codebase to be dynamic, we use a **compilation-per-model** strategy with an **architecture registry** that routes each model to the correct inference template.

### 3.1 Architecture Registry

The core abstraction: every model architecture is a `(template, weight_adapter)` pair. The template defines the ANE compute graph. The weight adapter parses the model's native weight format and maps tensors into the template's expected layout.

```python
# Architecture registry — maps architecture tag to (template, adapter)
ARCHITECTURES = {
    # Decoder-only, RMSNorm + SwiGLU + RoPE
    "llama":       ("inference_llama.m.tmpl",    LlamaAdapter),
    "mistral":     ("inference_llama.m.tmpl",    MistralAdapter),
    "phi3":        ("inference_llama.m.tmpl",    Phi3Adapter),
    "gemma":       ("inference_llama.m.tmpl",    GemmaAdapter),
    "qwen":        ("inference_llama.m.tmpl",    QwenAdapter),

    # Decoder-only, LayerNorm + GELU
    "gpt2":        ("inference_gpt.m.tmpl",      GPT2Adapter),
    "gpt-neox":    ("inference_gpt.m.tmpl",      NeoXAdapter),
    "opt":         ("inference_gpt.m.tmpl",      OPTAdapter),
    "pythia":      ("inference_gpt.m.tmpl",      NeoXAdapter),

    # Decoder-only, state-space
    "mamba":       ("inference_mamba.m.tmpl",     MambaAdapter),

    # Encoder-only
    "bert":        ("inference_encoder.m.tmpl",  BERTAdapter),
    "roberta":     ("inference_encoder.m.tmpl",  RoBERTaAdapter),

    # Encoder-decoder
    "t5":          ("inference_encdec.m.tmpl",   T5Adapter),
    "bart":        ("inference_encdec.m.tmpl",   BARTAdapter),
    "flan-t5":     ("inference_encdec.m.tmpl",   T5Adapter),

    # Mixture of experts (dense layers on ANE, routing on CPU)
    "mixtral":     ("inference_moe.m.tmpl",      MixtralAdapter),
    "dbrx":        ("inference_moe.m.tmpl",      DBRXAdapter),

    # Vision
    "vit":         ("inference_vit.m.tmpl",      ViTAdapter),
    "clip":        ("inference_clip.m.tmpl",     CLIPAdapter),

    # Diffusion
    "sd-unet":     ("inference_diffusion.m.tmpl", SDUNetAdapter),
}
```

Adding a new architecture = adding one template file + one adapter class. No changes to gateway, sandbox, proof system, or payments.

### 3.2 Model Manifest

Every model in the store has a manifest. The `architecture` field selects the template. The `config` fields are architecture-specific.

```json
{
  "model_id": "sha256:<hash of weights.bin>",
  "architecture": "llama",
  "format": "safetensors",
  "config": {
    "dim": 4096,
    "hidden_dim": 11008,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 32000,
    "seq_len": 4096,
    "norm_type": "rmsnorm",
    "activation": "swiglu",
    "position_encoding": "rope",
    "rope_theta": 10000.0
  },
  "size_bytes": 13500000000,
  "uploaded_at": "2026-03-03T12:00:00Z"
}
```

For non-transformer architectures, `config` carries different fields:

```json
{
  "architecture": "sd-unet",
  "config": {
    "in_channels": 4,
    "out_channels": 4,
    "block_out_channels": [320, 640, 1280, 1280],
    "attention_head_dim": 8,
    "cross_attention_dim": 768,
    "timestep_embedding_dim": 1280
  }
}
```

```json
{
  "architecture": "mamba",
  "config": {
    "dim": 2560,
    "n_layers": 64,
    "ssm_state_size": 16,
    "ssm_conv_width": 4,
    "expand_factor": 2,
    "vocab_size": 50280
  }
}
```

### 3.3 Weight Adapters

A weight adapter is a Python class that reads a specific weight format and produces a normalized binary blob that the compiled template binary can `mmap`:

```python
class WeightAdapter:
    """Base class. Subclass for each model family."""

    def detect(self, path: str) -> dict:
        """Read model file, return config dict or raise ValueError."""
        raise NotImplementedError

    def export(self, path: str, config: dict, output_path: str):
        """Convert weights to template-expected binary layout.
        Output: contiguous float16 arrays in canonical order."""
        raise NotImplementedError

    @property
    def architecture(self) -> str:
        raise NotImplementedError
```

Example adapter logic for different formats:

```
LlamaAdapter.detect():
  - Read 28-byte header (7 ints): dim, hidden, layers, heads, kv_heads, vocab, seq
  - Return config dict

GPT2Adapter.detect():
  - Read safetensors/HF metadata
  - Extract: n_embd, n_layer, n_head, vocab_size, n_positions
  - Map to: dim=n_embd, hidden=4*n_embd, n_layers=n_layer, etc.
  - Return config dict

BERTAdapter.detect():
  - Read config.json: hidden_size, num_hidden_layers, num_attention_heads, intermediate_size
  - Map to: dim=hidden_size, hidden=intermediate_size, n_layers, etc.
  - Return config dict

SDUNetAdapter.detect():
  - Read diffusion_pytorch_model.safetensors metadata
  - Extract channel dimensions, attention config
  - Return config dict
```

**Weight layout contract**: Every adapter exports weights as a contiguous binary blob with a standard header:

```c
struct WeightHeader {
    uint32_t magic;              // 0x414E4557 ("ANEW")
    uint32_t version;            // 1
    char     architecture[32];   // "llama", "gpt2", "bert", etc.
    uint32_t n_tensors;          // Number of weight tensors
    uint32_t dtype;              // 0=fp32, 1=fp16
    // Followed by tensor offsets table, then contiguous weight data
};
```

### 3.4 Templated Compilation

Rather than making the C code dynamic (fragile, 50+ locations to change), we **generate a model-specific C source file from a template** at model registration time, compile it once, and cache the binary.

```
Model registration flow:

  1. Client uploads model (any format: safetensors, GGUF, llama2c, ONNX, bin)
  2. Auto-detect: try each adapter's detect() until one succeeds
     - Or client specifies architecture explicitly in upload metadata
  3. Adapter exports weights to normalized binary layout
  4. Select template by architecture tag from registry
  5. Generate model-specific C source from template:
     - sed s/%%DIM%%/4096/g; s/%%HIDDEN%%/11008/g; s/%%NORM%%/rmsnorm/g; ...
  6. Compile: xcrun clang -O2 -o inference_<hash> inference_<hash>.m ...
  7. Hash binary: binary_hash = sha256(inference_<hash>)
  8. Store binary + normalized weights + manifest in model store
```

**Why this approach:**
- ANE kernels are compiled with dimensions baked in anyway (MIL text embeds tensor shapes). Dynamic dims don't help at the ANE level.
- A pre-compiled binary per model is faster than parsing config at runtime.
- The binary hash becomes part of the attestation — anyone can reproduce it from the same source + weights.
- No risk of buffer overflows from dynamic allocation mistakes.
- Architecture-specific optimizations (fused kernels, layout choices) live in the template, not in runtime branching.

### 3.5 Template Anatomy

Each template is a complete inference implementation for one architecture family. Templates share common ANE primitives (MIL generation, IOSurface I/O, SEP signing) via shared headers but differ in compute graph structure.

```
Templates and what they implement:

inference_llama.m.tmpl
  Block: RMSNorm → QKV → RoPE → Causal SDPA → Wo → Residual → RMSNorm → SwiGLU FFN → Residual
  ANE kernels: fused_qkv, fused_ffn_up, wo_proj, ffn_down (per layer)
  CPU: RoPE, causal masking, sampling

inference_gpt.m.tmpl
  Block: LayerNorm → QKV → Causal Attn → Wo → Residual → LayerNorm → GELU FFN → Residual
  ANE kernels: qkv_proj, ffn_up_gelu, wo_proj, ffn_down (per layer)
  CPU: positional embeddings (learned, not RoPE), sampling

inference_encoder.m.tmpl
  Block: LayerNorm → QKV → Bidirectional Attn → Wo → Residual → LayerNorm → GELU FFN → Residual
  ANE kernels: same as GPT but NO causal mask
  CPU: [CLS] pooling, classification head

inference_encdec.m.tmpl
  Two stacks: encoder (bidirectional) + decoder (causal + cross-attention)
  ANE kernels: encoder_block, decoder_self_attn, decoder_cross_attn, ffn
  CPU: beam search, length penalty

inference_moe.m.tmpl
  Block: same as llama but FFN replaced by top-k expert routing
  ANE kernels: fused_qkv, wo_proj, expert_ffn (per expert, compiled once, dispatched per-token)
  CPU: router (gating network), expert selection, token-to-expert dispatch

inference_mamba.m.tmpl
  Block: selective state space model (no attention)
  ANE kernels: ssm_conv, ssm_scan (selective scan as conv chain)
  CPU: discretization (ZOH), state update

inference_vit.m.tmpl
  Patch embedding → positional embedding → encoder blocks → classification head
  ANE kernels: patch_embed_conv, encoder_block (bidirectional attn + MLP)
  CPU: patch extraction, classification

inference_clip.m.tmpl
  Dual encoder: ViT (image) + GPT (text) → contrastive similarity
  ANE kernels: image_encoder, text_encoder
  CPU: cosine similarity, contrastive loss

inference_diffusion.m.tmpl
  UNet: downsample → middle → upsample with skip connections + cross-attention
  ANE kernels: resnet_block, cross_attn, downsample_conv, upsample_conv
  CPU: timestep scheduling, noise prediction loop, VAE decode
```

### 3.6 Shared Headers

All templates include the same core headers:

| Header | Contents |
|--------|----------|
| `ane_runtime.h` | ANE private API wrapper (compile, eval, IOSurface) |
| `ane_mil_gen.h` | MIL text generation helpers (conv, matmul, softmax, etc.) |
| `proof_output.h` | Logits hashing, layer checkpoints, proof JSON serialization |
| `sep_sign.h` | Secure Enclave key management and signing |
| `weight_io.h` | Normalized weight blob loading (mmap + header parse) |

Architecture-specific headers:
| Header | Used by |
|--------|---------|
| `rope.h` | Llama, Mistral — RoPE position encoding |
| `layernorm.h` | GPT, BERT, T5 — LayerNorm (vs RMSNorm) |
| `moe_router.h` | Mixtral, DBRX — top-k expert gating |
| `ssm.h` | Mamba — selective state space discretization |
| `unet.h` | Stable Diffusion — UNet skip connections, timestep embedding |

### 3.7 Supported Formats & Auto-Detection

The provider accepts models in any common format. Auto-detection priority:

| Priority | Format | Detection | Architectures |
|----------|--------|-----------|---------------|
| 1 | Safetensors | `.safetensors` extension, JSON header | All (HuggingFace standard) |
| 2 | GGUF | Magic bytes `GGUF` at offset 0 | All (llama.cpp ecosystem) |
| 3 | llama2.c | 28-byte int header, size matches config | Llama family |
| 4 | ONNX | Magic bytes `\x08` (protobuf) | All |
| 5 | PyTorch | Magic bytes `PK` (zip) | All (with config.json) |

For safetensors and GGUF, architecture is auto-detected from metadata. For raw binary formats, the client must specify `architecture` in the upload request.

### 3.8 ANE Primitive Coverage

Every architecture decomposes into the same small set of ANE primitives. This is why one accelerator can run all of them:

| ANE Primitive | MIL Op | Used By |
|---------------|--------|---------|
| Linear projection | `conv(weight, x, pad="valid")` | All (QKV, FFN, embeddings) |
| MatMul (attention) | `matmul(Q, K^T)` | All attention-based models |
| Softmax | `softmax(x)` | All attention-based models |
| RMSNorm | `reduce_sum(x*x) + pow + mul` | Llama, Mistral, Gemma, Qwen |
| LayerNorm | `reduce_mean + reduce_sum + mul + add` | GPT-2, BERT, T5, BART, ViT |
| SiLU/SwiGLU | `sigmoid(x) * x * gate` | Llama, Mistral |
| GELU | `x * 0.5 * (1 + tanh(...))` | GPT-2, BERT, T5, ViT |
| ReLU | `max(x, 0)` | OPT, older models |
| Convolution (spatial) | `conv(weight, x, pad=...)` | Diffusion UNet, Mamba (1D conv) |
| Element-wise add | `add(x, y)` | All (residual connections) |
| Concat | `concat(x, y, dim)` | UNet skip connections, forward taps |

Ten primitives cover every architecture. The template just defines the order and connectivity.

---

## 4. Sandbox & Isolation

Every job runs in an isolated environment. A malicious model or input cannot escape.

### 4.1 Process Isolation

```
Gateway (parent)
  │
  ├── fork+exec: inference_<model_hash>
  │     │
  │     ├── Reads input from stdin (JSON)
  │     ├── Loads weights from read-only path
  │     ├── Compiles ANE kernels in private tmpdir
  │     ├── Runs inference
  │     ├── Writes output to stdout (JSON + proof data)
  │     └── Exits (all ANE kernels auto-unloaded)
  │
  └── Parent collects stdout, assembles proof, returns to client
```

### 4.2 Resource Limits

The gateway enforces per-job limits before exec:

| Resource | Limit | Enforcement |
|----------|-------|-------------|
| Wall time | 60s (inference), 600s (training batch) | `alarm()` / SIGKILL |
| Memory | 4 GB (inference), 16 GB (training) | `setrlimit(RLIMIT_AS)` |
| Disk (tmpdir) | 2 GB | Tmpfs mount with size cap |
| ANE compiles | 100 per process | Existing ANE limit (natural) |
| Network | None | `sandbox-exec` deny-network (macOS) |
| File access | Read-only model dir + private tmpdir | `sandbox-exec` profile |

### 4.3 macOS Sandbox Profile

```scheme
(version 1)
(deny default)

; Allow read from model store
(allow file-read*
  (subpath "/Users/*/ane_provider/models"))

; Allow read-write to job tmpdir
(allow file-read* file-write*
  (subpath "/private/tmp/ane_job_*"))

; Allow ANE framework access
(allow file-read*
  (subpath "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework"))

; Allow IOSurface
(allow iokit-open)
(allow mach-lookup
  (global-name "com.apple.ane.ioservice"))

; Deny network
(deny network*)

; Deny process control
(deny process-exec)
(deny signal)
```

### 4.4 Cleanup

On job completion (success or failure):
1. Kill subprocess if still running
2. `rm -rf /tmp/ane_job_<id>/`
3. ANE kernels auto-unload when process exits (OS reclaims)
4. Zero parent-process state change

---

## 5. Gateway Server

### 5.1 Endpoints

```
POST /v1/inference
  → x402 payment gated
  → Input: { model_id, tokens, max_tokens, temperature, seed, top_p }
  → Output: { tokens, proof_bundle }

POST /v1/training/step
  → x402 payment gated
  → Input: { model_id, data_url, steps, learning_rate }
  → Output: { losses, checkpoint_hash, merkle_root, proof_bundle }

POST /v1/models
  → Upload model weights (multipart/form-data)
  → Optional: architecture hint (auto-detected if omitted)
  → Auto-detects format + architecture, selects template, compiles
  → Output: { model_id, architecture, config, binary_hash }

GET /v1/models
  → List available models with configs and pricing

GET /v1/models/<id>
  → Model manifest + binary hash

GET /v1/health
  → { status, queue_depth, chip, ane_tops, models_loaded }

GET /v1/capabilities
  → { chip, memory, ane_tops, models, architectures, services, pricing }

GET /v1/architectures
  → List supported architecture families + template versions
```

### 5.2 x402 Integration

The gateway acts as an x402 resource server. Payment is verified before compute begins.

```python
@app.route('/v1/inference', methods=['POST'])
def inference():
    # Check for payment header
    payment = request.headers.get('X-PAYMENT')
    if not payment:
        # Return 402 with payment requirements
        return Response(
            status=402,
            headers={
                'X-PAYMENT-REQUIRED': json.dumps({
                    'address': PROVIDER_ADDRESS,
                    'amount': str(estimate_cost(request.json)),
                    'asset': 'USDC',
                    'chain': 'eip155:8453',  # Base
                    'expiry': int(time.time()) + 300,
                    'facilitator': 'https://x402.coinbase.com'
                })
            }
        )

    # Verify payment via facilitator
    if not verify_payment(payment, request.json):
        return Response(status=402, ...)

    # Payment valid — execute job
    result = execute_inference_job(request.json)
    return jsonify(result)
```

### 5.3 Job Queue

```
                  ┌──────────┐
   request ─────►│  Queue    │
                  │ (bounded) │
                  └────┬─────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
     ┌────▼───┐  ┌────▼───┐  ┌────▼───┐
     │Worker 1│  │Worker 2│  │Worker 3│
     │(subproc│  │(subproc│  │(subproc│
     └────────┘  └────────┘  └────────┘
```

- **Queue depth** configurable (default: 8). Requests beyond capacity get 503.
- **Workers** = number of concurrent ANE jobs. Default: 2 (ANE can multiplex but throughput drops with >2 concurrent models).
- **Priority**: Higher-paying jobs dequeue first. Prevents low-value spam from blocking legitimate requests.

---

## 6. Proof Bundle

Every response includes a proof bundle — the evidence package that makes the result verifiable.

### 6.1 Structure

```json
{
  "proof_version": 1,

  "job": {
    "job_id": "uuid",
    "model_id": "sha256:abc...",
    "binary_hash": "sha256:def...",
    "input_hash": "keccak256(input_tokens)",
    "output_hash": "keccak256(output_tokens)",
    "seed": 42,
    "temperature": 0.0,
    "timestamp": 1709510400
  },

  "attestation": {
    "sep_signature": "<base64 Secure Enclave signature over job fields>",
    "cert_chain": ["<device cert>", "<intermediate>", "<Apple root>"],
    "device_id": "<hardware identifier>",
    "chip_family": "M4"
  },

  "reproducibility": {
    "logits_hash": "keccak256(final_logits_fp32)",
    "layer_checkpoints": {
      "6": "keccak256(residual_after_layer_6)"
    },
    "rng_state": "<ChaCha20 state after sampling>"
  }
}
```

### 6.2 What Each Part Proves

| Field | Proves | Verifiable by |
|-------|--------|---------------|
| `sep_signature` | A real Apple device produced this output | Anyone (verify cert chain) |
| `binary_hash` | The inference binary matches published source | Anyone (recompile and compare) |
| `input_hash` / `output_hash` | Commitment to specific I/O | Anyone (hash the data) |
| `logits_hash` | The raw logits before sampling | PoSP validator (re-run model, compare hash) |
| `layer_checkpoints` | Intermediate state at layer N | Spot-check verifier |
| `seed` + `rng_state` | Sampling was deterministic given seed | Anyone (replay sampling) |

### 6.3 SEP Signing Flow

The inference binary calls the Secure Enclave via the Keychain API:

```
1. At first boot, generate a SEP-bound EC key:
   SecKeyCreateRandomKey(kSecAttrTokenIDSecureEnclave, ...)

2. At end of each inference:
   payload = canonical_serialize(job_id, model_id, input_hash, output_hash, ...)
   signature = SecKeyCreateSignature(sep_key, kSecKeyAlgorithmECDSASignatureMessageX962SHA256, payload)

3. Return (signature, certificate_chain) as attestation
```

The private key never leaves the Secure Enclave. The signature proves this specific device produced this specific output.

---

## 7. Proof of Sampling (PoSP)

### 7.1 Roles

| Role | Who | Action |
|------|-----|--------|
| **Asserter** | The provider that computed the job | Submits output + proof bundle |
| **Orchestrator** | On-chain VRF contract | Decides which jobs get challenged, selects validators |
| **Validator** | Another ANE provider (randomly selected) | Re-runs computation, compares result |

### 7.2 Protocol

```
For each completed job:

1. Asserter posts on-chain:
   commit(job_id, output_hash, logits_hash, attestation)

2. Orchestrator draws VRF:
   challenge = VRF(block_hash, job_id) < challenge_threshold

   If not challenged (90-95% of jobs):
     → Finalize immediately, release payment

   If challenged (5-10% of jobs):
     → Select K validators from same chip_family
     → Validators receive (model_id, input_tokens, seed, temperature)

3. Each validator:
   a. Load same model (by model_id hash)
   b. Run inference with same input + seed
   c. Compute logits_hash and output_hash
   d. Submit: commit(job_id, validator_output_hash, validator_logits_hash)

4. Compare:
   If asserter.logits_hash == majority(validator.logits_hash):
     → Asserter correct. Finalize. Validators get reward.
   Else:
     → Asserter wrong. Slash asserter stake. Refund client.
```

### 7.3 Determinism Guarantees

For PoSP to work, the same input must produce the same output across nodes. We enforce this at multiple levels:

**Level 1 — Same binary.** All providers for a given model compile from the same template source. Binary hash published in manifest. Validators reject binary hash mismatch.

**Level 2 — Same MIL programs.** The MIL text is a deterministic function of (template_source, model_dimensions). Same dimensions → same MIL → same ANE program → same compute graph.

**Level 3 — Same chip family.** Validators selected from same chip family (M4 validates M4). Within a family, ANE produces bit-identical fp16 results because:
- Fixed dataflow (no thread scheduling variance)
- Deterministic reduction order (compiled into MIL)
- No atomic operations

**Level 4 — Deterministic sampling.** For temperature > 0, both asserter and validator use `ChaCha20(seed=keccak256(job_id))` as PRNG. Same seed → same random draws → same token sequence.

**Level 5 — Tolerance for cross-family.** When same-family validators aren't available, allow L1-norm tolerance on logits: `||logits_A - logits_B||_1 / vocab_size < 0.01`. Calibrated from empirical cross-family divergence measurements.

### 7.4 Nash Equilibrium Condition

From the [PoSP paper](https://arxiv.org/abs/2405.00295), honesty is the dominant strategy when:

```
p × (stake + reward) > cost_of_compute

Where:
  p = challenge probability (0.05 to 0.10)
  stake = provider's locked stake (e.g., 0.01 ETH)
  reward = job payment (e.g., 0.001 USDC)
  cost_of_compute = electricity + depreciation for one inference (~$0.0001)

Example:
  0.05 × (0.01 ETH + 0.001 USDC) > $0.0001
  0.05 × $30.001 > $0.0001
  $1.50 > $0.0001  ✓  (by a factor of 15,000)
```

Cheating is economically irrational by 4 orders of magnitude.

---

## 8. Model Store

### 8.1 Directory Layout

```
~/.ane_provider/
├── config.yaml                    # Provider configuration
├── keys/
│   └── sep_key_ref.pem            # Reference to SEP key (not the key itself)
├── models/
│   ├── sha256_a1b2c3.../
│   │   ├── weights.bin            # Original weights (read-only)
│   │   ├── manifest.json          # Config, hashes, metadata
│   │   ├── inference              # Compiled binary (model-specific)
│   │   └── tokenizer.bin          # Optional tokenizer
│   └── sha256_d4e5f6.../
│       └── ...
└── logs/
    └── provider.log
```

### 8.2 Model Lifecycle

```
Upload → Detect → Validate → Adapt → Template → Compile → Hash → Register → Serve
  │        │         │         │         │          │        │        │         │
  │     auto-ID   check     export    select     xcrun   sha256   on-chain   ready
  │     arch +    dims +    weights   template   clang   binary   announce
  │     format    size      to ANEW   by arch
  │
  └─ reject if:
     - no adapter recognizes the format
     - architecture not in registry (unknown template)
     - dims unsupported (e.g., dim > 16384)
     - file too large (> 50 GB)
     - weight tensor count doesn't match architecture expectations
     - compilation fails (invalid dimensions for ANE)
```

### 8.3 Cache & Eviction

- **Compiled binaries** cached indefinitely (small: ~200 KB each)
- **ANE kernel cache** managed by macOS (temp directories)
- **Weights** evicted LRU when disk exceeds configured limit (default: 100 GB)
- **Hot models** (served in last hour) never evicted

---

## 9. Configuration

```yaml
# ~/.ane_provider/config.yaml

provider:
  address: "0x..."                   # Ethereum address for payments
  private_key_path: "./keys/eth.key" # For signing on-chain txs
  stake: "0.01"                      # ETH staked on-chain

server:
  host: "0.0.0.0"
  port: 8402
  tls_cert: "./certs/cert.pem"
  tls_key: "./certs/key.pem"
  max_queue: 8
  max_workers: 2

x402:
  facilitator: "https://x402.coinbase.com"
  chain: "eip155:8453"               # Base
  asset: "USDC"

models:
  store_path: "~/.ane_provider/models"
  max_disk_gb: 100
  allowed_architectures: "all"       # Or list: ["llama", "gpt2", "bert", "t5", "mamba", "vit", "sd-unet"]
  max_dim: 16384
  max_layers: 128
  max_vocab: 256000
  max_experts: 16                    # For MoE models

sandbox:
  timeout_inference_s: 60
  timeout_training_s: 600
  max_memory_gb: 4
  network: deny

posp:
  registry_contract: "0x..."
  challenge_rate: 0.05               # 5% of jobs challenged
  validator_reward_bps: 50           # 0.5% of job value

logging:
  level: "info"
  file: "~/.ane_provider/logs/provider.log"
```

---

## 10. Wire Protocol

### 10.1 Inference Request

```
POST /v1/inference HTTP/1.1
Host: provider.example.com:8402
Content-Type: application/json
X-PAYMENT: <x402 payment payload>

{
  "model_id": "sha256:a1b2c3...",
  "tokens": [1, 4532, 817],
  "max_tokens": 256,
  "temperature": 0.0,
  "seed": 42,
  "top_p": 1.0,
  "stream": false,
  "include_logits": false
}
```

### 10.2 Inference Response

```json
{
  "tokens": [291, 1033, 445, 2],
  "text": "Once upon a time...",
  "finish_reason": "eos",
  "usage": {
    "prompt_tokens": 3,
    "completion_tokens": 4,
    "total_tokens": 7,
    "ane_time_ms": 12.4,
    "total_time_ms": 18.7
  },
  "proof": {
    "proof_version": 1,
    "job": { ... },
    "attestation": { ... },
    "reproducibility": { ... }
  }
}
```

### 10.3 IPC with Sandbox Subprocess

Gateway → subprocess communication via stdin/stdout:

```
Gateway writes to stdin:
{
  "command": "inference",
  "weights_path": "/path/to/weights.bin",
  "tokenizer_path": "/path/to/tokenizer.bin",
  "tokens": [1, 4532, 817],
  "max_tokens": 256,
  "temperature": 0.0,
  "seed": 42,
  "top_p": 1.0
}
<EOF>

Subprocess writes to stdout:
{
  "tokens": [291, 1033, 445, 2],
  "logits_hash": "abc...",
  "layer_checkpoints": {"6": "def..."},
  "ane_time_ms": 12.4
}
<EOF>
```

The subprocess reads JSON from stdin, runs inference, writes JSON to stdout, exits. No persistent connection. No shared state.

---

## 11. File Structure

```
eth_provider/
├── DESIGN.md                        # Strategic design document (verification theory)
├── IMPLEMENTATION.md                # This document
│
├── contracts/
│   └── ANEMarketplace.sol           # On-chain: escrow, staking, PoSP, slashing
│
├── gateway/
│   ├── server.py                    # Flask x402 server
│   ├── x402.py                      # x402 payment verification
│   ├── scheduler.py                 # Job queue + worker pool
│   ├── proof.py                     # Proof bundle assembly
│   └── posp.py                      # PoSP orchestrator client
│
├── sandbox/
│   ├── runner.py                    # Subprocess exec with sandbox-exec
│   ├── ane_sandbox.sb               # macOS sandbox profile
│   └── limits.py                    # Resource limit enforcement
│
├── models/
│   ├── store.py                     # Model store management (upload, compile, evict)
│   ├── registry.py                  # Architecture registry (arch → template + adapter)
│   ├── adapters/
│   │   ├── base.py                  # WeightAdapter base class
│   │   ├── llama.py                 # Llama, Mistral, Phi, Gemma, Qwen adapters
│   │   ├── gpt.py                   # GPT-2, GPT-NeoX, OPT, Pythia adapters
│   │   ├── encoder.py              # BERT, RoBERTa adapters
│   │   ├── encdec.py               # T5, BART, Flan-T5 adapters
│   │   ├── moe.py                   # Mixtral, DBRX adapters
│   │   ├── mamba.py                 # Mamba/SSM adapter
│   │   ├── vision.py               # ViT, CLIP adapters
│   │   └── diffusion.py            # Stable Diffusion UNet adapter
│   ├── formats/
│   │   ├── safetensors.py           # Safetensors reader
│   │   ├── gguf.py                  # GGUF reader
│   │   ├── llama2c.py               # llama2.c binary reader
│   │   └── onnx.py                  # ONNX reader
│   └── templates/
│       ├── shared/
│       │   ├── ane_runtime.h         # ANE private API wrapper
│       │   ├── ane_mil_gen.h         # MIL text generation helpers
│       │   ├── proof_output.h        # Logits hashing, proof serialization
│       │   ├── sep_sign.h            # Secure Enclave signing
│       │   └── weight_io.h           # Normalized weight blob loading
│       ├── arch_headers/
│       │   ├── rope.h                # RoPE position encoding (Llama, Mistral)
│       │   ├── layernorm.h           # LayerNorm (GPT, BERT, T5, ViT)
│       │   ├── moe_router.h          # Top-k expert gating (Mixtral, DBRX)
│       │   ├── ssm.h                 # Selective state space ops (Mamba)
│       │   └── unet.h                # UNet skip connections (Diffusion)
│       ├── inference_llama.m.tmpl    # Decoder-only: RMSNorm + SwiGLU + RoPE
│       ├── inference_gpt.m.tmpl      # Decoder-only: LayerNorm + GELU
│       ├── inference_encoder.m.tmpl  # Encoder-only: bidirectional attention
│       ├── inference_encdec.m.tmpl   # Encoder-decoder: cross-attention
│       ├── inference_moe.m.tmpl      # Mixture of experts: routing + expert FFN
│       ├── inference_mamba.m.tmpl    # State-space: selective scan
│       ├── inference_vit.m.tmpl      # Vision transformer: patch embed + encoder
│       ├── inference_clip.m.tmpl     # CLIP: dual encoder + contrastive
│       └── inference_diffusion.m.tmpl # Diffusion UNet: downsample/upsample + cross-attn
│
├── attestation/
│   ├── sep.py                       # Secure Enclave key management
│   └── verify.py                    # Attestation chain verification
│
├── config.py                        # Config loading (YAML)
├── requirements.txt
└── README.md                        # Quick start
```

---

## 12. Threat Model (Implementation-Specific)

| Threat | Mitigation |
|--------|-----------|
| Malicious model weights (crafted to exploit buffer overflow) | Sandbox: deny network, deny process-exec, memory limit. Weight adapter validates tensor shapes against architecture schema before generating C code. Template compilation rejects invalid dims. |
| Unknown architecture used to bypass adapter validation | Architecture tag must map to a registered template. Upload rejected if no adapter recognizes the format. `allowed_architectures` config can restrict to a whitelist. |
| Model too large for device memory | Pre-check: `weight_bytes < available_memory * 0.8`. Reject at upload. Per-architecture memory estimator (weights + KV-cache + activations). |
| Denial of service (flood requests) | Bounded queue (503 on overflow). x402 payment required (costs money to spam). Rate limit per wallet. |
| Provider returns cached stale result | Freshness nonce in every request. SEP attestation includes timestamp. PoSP validators use same nonce. |
| Provider runs weaker model (fewer layers) | Binary hash in attestation. Validators compile same template from same config, compare hash. |
| Provider swaps architecture (claims Llama, runs GPT-2) | Architecture tag in manifest. Binary hash tied to specific template. Validators detect template mismatch. Logits dimensionality check (wrong arch = wrong output shape). |
| Crafted adapter exploits compilation | Adapter runs in gateway (before sandbox) — dimension bounds checked: `0 < dim <= max_dim`, `0 < layers <= max_layers`. Generated C source is deterministic from config. No user-controlled strings in generated code. |
| Sandbox escape | macOS sandbox-exec is kernel-enforced. ANE access is read-only (no code execution on ANE). Process exits after each job. Same profile for all architectures. |
| Side-channel on model weights | Weights are read-only in sandbox. Network denied. No exfiltration path. |
| Compromised gateway (not subprocess) | Gateway never touches model weights or ANE. It only routes JSON. Attestation is signed by subprocess via SEP. |
| Cross-architecture PoSP mismatch | Validators must use same `(architecture, template_version, binary_hash)`. Different architectures naturally produce different binary hashes, preventing cross-architecture validation. |

---

## 13. Implementation Sequence

```
Week 1: Core plumbing (Llama template as first architecture)
  ├── Shared headers: weight_io.h, proof_output.h, sep_sign.h
  ├── inference_llama.m.tmpl (parameterized Llama inference template)
  ├── LlamaAdapter + safetensors reader
  ├── registry.py (architecture → template mapping)
  ├── store.py (model upload, auto-detect, adapt, compile, cache)
  ├── runner.py (subprocess exec with IPC)
  └── Verify: upload Llama model → compile → run inference → get output

Week 2: Multi-architecture + proof
  ├── inference_gpt.m.tmpl (GPT-2/NeoX template, layernorm.h)
  ├── inference_encoder.m.tmpl (BERT template)
  ├── GPT2Adapter, BERTAdapter, GGUF format reader
  ├── Add logits_hash + layer_checkpoints to all template outputs
  ├── sep.py (SEP key gen + signing via Security.framework)
  ├── proof.py (assemble proof bundle)
  └── Verify: upload GPT-2 and BERT models, get valid proofs from both

Week 3: Remaining architectures + x402 + server
  ├── inference_encdec.m.tmpl (T5/BART)
  ├── inference_moe.m.tmpl (Mixtral — moe_router.h)
  ├── inference_vit.m.tmpl, inference_diffusion.m.tmpl
  ├── Remaining adapters (T5, Mixtral, ViT, SD)
  ├── server.py (Flask endpoints)
  ├── x402.py (payment verification via Coinbase facilitator)
  ├── scheduler.py (queue + workers)
  └── Verify: end-to-end paid inference across 3+ architectures

Week 4: PoSP + on-chain
  ├── ANEMarketplace.sol (deploy to Base testnet)
  ├── posp.py (commit results, handle challenges)
  ├── Validator mode (re-run jobs on challenge, architecture-aware matching)
  ├── inference_mamba.m.tmpl (Mamba SSM — last architecture)
  └── Verify: full loop — pay, compute, prove, challenge, settle
```
