# ANE Compute Provider — Implementation Spec

MVP implementation: Secure Enclave attestation + Proof of Sampling + x402 payments. Fully open plugin architecture — zero hardcoded models. Any architecture, any format, any work. Sandboxed execution, multi-tenant.

---

## 1. Design Principles

1. **Zero hardcoded models.** The provider has no built-in knowledge of any specific model or architecture. It discovers capabilities at startup by scanning plugin directories. Templates, adapters, and format readers are all plugins. The core system is a generic pipeline: `upload → detect → adapt → compile → sandbox → prove`.
2. **Plugin-first.** Support for a new architecture = drop files in a directory. No code changes to the gateway, sandbox, scheduler, proof system, or payment layer. No redeployment. No registry edits. The plugin declares what it handles.
3. **Isolated.** Each job runs in a sandboxed subprocess with its own memory, temp directory, and ANE kernel set. A crash or malicious model cannot affect other jobs or the host.
4. **Stateless between jobs.** No job state persists after completion. KV-cache, activations, compiled kernels — all freed. The provider is a pure function: `(model, input) → (output, proof)`.
5. **Minimal trust surface.** The provider binary is open-source and deterministically compiled. The Secure Enclave signs every result. The provider operator cannot tamper with outputs without invalidating the SEP attestation.

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

## 3. Plugin Architecture

The provider has **zero hardcoded knowledge** of any specific model or architecture. Everything is discovered at runtime from plugin directories. The core system is a generic pipeline; all model-specific logic lives in plugins.

### 3.1 Plugin Discovery

At startup, the provider scans three plugin directories and builds its capability set dynamically:

```
~/.ane_provider/
├── plugins/
│   ├── templates/    ← Inference templates (*.m.tmpl + manifest.yaml)
│   ├── adapters/     ← Weight adapters (*.py, auto-imported)
│   └── formats/      ← Format readers (*.py, auto-imported)
```

```python
# At startup — NO hardcoded list anywhere
class PluginLoader:
    def __init__(self, plugin_dir: str):
        self.templates = {}   # pattern_tag → TemplatePlugin
        self.adapters  = []   # [AdapterPlugin, ...]
        self.formats   = []   # [FormatPlugin, ...]

    def scan(self):
        """Walk plugin dirs. Import all .py modules. Parse all .yaml manifests.
        Each plugin self-registers by declaring what it handles."""

        # Templates: scan for *.m.tmpl files, read their sidecar manifest.yaml
        for manifest_path in glob("plugins/templates/*/manifest.yaml"):
            tpl = TemplatePlugin.from_manifest(manifest_path)
            self.templates[tpl.pattern] = tpl

        # Adapters: import all .py files, collect subclasses of AdapterPlugin
        for py_file in glob("plugins/adapters/*.py"):
            module = importlib.import_module(py_file)
            for cls in find_subclasses(module, AdapterPlugin):
                self.adapters.append(cls())

        # Formats: import all .py files, collect subclasses of FormatPlugin
        for py_file in glob("plugins/formats/*.py"):
            module = importlib.import_module(py_file)
            for cls in find_subclasses(module, FormatPlugin):
                self.formats.append(cls())
```

**No code changes to add support for a new model.** Drop files in the plugin directories. Restart (or hot-reload). Done.

### 3.2 Template Plugins

A template plugin is a directory containing an inference template (`.m.tmpl`) and a manifest that declares what it handles. The manifest is the template's self-description — the provider never needs to know what's inside the template.

```
plugins/templates/
├── decoder_rmsnorm_swiglu/
│   ├── manifest.yaml
│   ├── inference.m.tmpl
│   └── deps/                  ← Optional extra headers
│       └── rope.h
├── decoder_layernorm_gelu/
│   ├── manifest.yaml
│   └── inference.m.tmpl
├── encoder_bidirectional/
│   ├── manifest.yaml
│   └── inference.m.tmpl
├── encoder_decoder/
│   ├── manifest.yaml
│   └── inference.m.tmpl
├── moe_topk/
│   ├── manifest.yaml
│   ├── inference.m.tmpl
│   └── deps/
│       └── router.h
├── state_space/
│   ├── manifest.yaml
│   ├── inference.m.tmpl
│   └── deps/
│       └── ssm.h
├── vision_patch/
│   ├── manifest.yaml
│   └── inference.m.tmpl
├── diffusion_unet/
│   ├── manifest.yaml
│   ├── inference.m.tmpl
│   └── deps/
│       └── unet.h
└── ... (any user-added template)
```

**Template manifest** (`manifest.yaml`):

```yaml
# Self-description. The provider reads this, never the template source.
pattern: "decoder_rmsnorm_swiglu"
version: 1
description: "Decoder-only transformer with RMSNorm, SwiGLU activation, RoPE"

# What config fields this template requires (used for validation)
required_config:
  - dim           # Model dimension
  - hidden_dim    # FFN hidden dimension
  - n_layers      # Number of transformer layers
  - n_heads       # Number of attention heads
  - n_kv_heads    # Number of key-value heads (for GQA)
  - vocab_size    # Vocabulary size
  - seq_len       # Maximum sequence length

# Optional config with defaults (template substitutes %%KEY%% → value)
optional_config:
  rope_theta: 10000.0
  norm_eps: 1e-5

# What this template produces
output_type: "token_sequence"        # or "embedding", "classification", "image"
supports_kv_cache: true
supports_batching: false

# Compile command (%%TEMPLATE_SRC%% and %%OUTPUT_BIN%% are injected)
compile: "xcrun clang -O2 -o %%OUTPUT_BIN%% %%TEMPLATE_SRC%% -framework Foundation -framework Accelerate"

# Resource hints (for scheduling & admission control)
resource_hints:
  memory_per_param_bytes: 2          # fp16
  ane_kernels_per_layer: 4
  cpu_ops: ["rope", "sampling"]
```

The provider never parses template C code. It only reads the manifest, substitutes `%%PLACEHOLDER%%` values from the model config into the template source, compiles, and runs.

### 3.3 Adapter Plugins

An adapter plugin knows how to read a specific model family's weight layout and map it to a normalized binary format that a template can consume. Adapters are Python files dropped into `plugins/adapters/`.

```python
class AdapterPlugin:
    """Base class. Each adapter is a .py file in plugins/adapters/."""

    @property
    def name(self) -> str:
        """Unique adapter name, e.g. 'hf_causal_lm'."""
        raise NotImplementedError

    @property
    def pattern(self) -> str:
        """The template pattern this adapter targets, e.g. 'decoder_rmsnorm_swiglu'.
        Must match a template's manifest.yaml pattern field."""
        raise NotImplementedError

    def can_handle(self, metadata: dict, format_hint: str) -> float:
        """Given parsed metadata from a format reader, return confidence 0.0-1.0
        that this adapter can handle the model. 0.0 = cannot, 1.0 = certain.
        Allows multiple adapters to compete; highest confidence wins."""
        raise NotImplementedError

    def extract_config(self, metadata: dict) -> dict:
        """Extract template config fields from model metadata.
        Returns dict matching the template's required_config fields."""
        raise NotImplementedError

    def export_weights(self, source_path: str, metadata: dict,
                       config: dict, output_path: str):
        """Read weights from source file, reorder/cast as needed,
        write to output_path in ANEW normalized layout."""
        raise NotImplementedError
```

**Key design: confidence scoring, not hardcoded matching.** When a model is uploaded, every adapter gets a chance to claim it. The adapter with the highest `can_handle()` score wins. This means:

- Two adapters can handle the same model family differently (community vs. official)
- A generic "HuggingFace CausalLM" adapter can handle any HF model that follows the standard naming convention
- A specialized adapter can override the generic one with higher confidence for models it knows well

```python
# Example: a GENERIC adapter that handles any HuggingFace causal LM
class HFCausalLMAdapter(AdapterPlugin):
    name = "hf_causal_lm"
    pattern = "decoder_rmsnorm_swiglu"  # default, overridden by metadata

    def can_handle(self, metadata, format_hint):
        # If it has 'model_type' in config.json, we can probably handle it
        if "model_type" in metadata:
            return 0.5  # generic confidence
        return 0.0

    def extract_config(self, metadata):
        # Generic HF config.json → template config mapping
        hf = metadata
        return {
            "dim":        hf.get("hidden_size", hf.get("d_model")),
            "hidden_dim": hf.get("intermediate_size", hf.get("d_ff")),
            "n_layers":   hf.get("num_hidden_layers", hf.get("n_layer")),
            "n_heads":    hf.get("num_attention_heads", hf.get("n_head")),
            "n_kv_heads": hf.get("num_key_value_heads",
                           hf.get("num_attention_heads")),
            "vocab_size": hf["vocab_size"],
            "seq_len":    hf.get("max_position_embeddings", 2048),
        }

    @property
    def pattern(self):
        # Could dynamically select pattern based on model metadata
        # e.g., check if it uses RMSNorm vs LayerNorm
        return self._detected_pattern or "decoder_rmsnorm_swiglu"
```

### 3.4 Format Plugins

A format plugin knows how to open a specific file format (safetensors, GGUF, ONNX, etc.) and extract metadata + raw tensor data. Format plugins are also auto-discovered from `plugins/formats/`.

```python
class FormatPlugin:
    """Base class. Each format reader is a .py file in plugins/formats/."""

    @property
    def name(self) -> str:
        """Format name, e.g. 'safetensors'."""
        raise NotImplementedError

    def can_read(self, path: str) -> bool:
        """Check magic bytes / extension. Return True if this format reader
        can parse the file."""
        raise NotImplementedError

    def read_metadata(self, path: str) -> dict:
        """Parse file header / config. Return metadata dict containing
        architecture hints, dimensions, tensor names, etc."""
        raise NotImplementedError

    def iter_tensors(self, path: str) -> Iterator[Tuple[str, np.ndarray]]:
        """Yield (tensor_name, tensor_data) pairs.
        Tensor data as numpy arrays (any dtype, adapter will cast)."""
        raise NotImplementedError
```

Format readers **don't know** about architectures. They just parse files. The adapter layer maps parsed metadata to template configs.

```
Upload pipeline (fully plugin-driven):

  1. File arrives
  2. Try each FormatPlugin.can_read() → first match opens the file
  3. FormatPlugin.read_metadata() → raw metadata dict
  4. Try each AdapterPlugin.can_handle(metadata) → highest confidence wins
  5. AdapterPlugin.extract_config(metadata) → template config dict
  6. AdapterPlugin.pattern → selects which TemplatePlugin to use
  7. Validate config against template's required_config
  8. AdapterPlugin.export_weights() → normalized ANEW binary
  9. Template substitution → model-specific .m source
  10. Compile → binary
  11. Hash → manifest → serve
```

Every step is pluggable. No step has hardcoded knowledge of any specific model.

### 3.5 Normalized Weight Format (ANEW)

The contract between adapters and templates. Adapters write this format; template binaries `mmap` and read it. Architecture-neutral.

```c
struct ANEWHeader {
    uint32_t magic;              // 0x414E4557 ("ANEW")
    uint32_t version;            // 1
    char     pattern[64];        // Template pattern name (e.g., "decoder_rmsnorm_swiglu")
    uint32_t n_tensors;          // Number of weight tensors
    uint32_t dtype;              // 0=fp32, 1=fp16, 2=bf16, 3=int8, 4=int4
    uint64_t data_offset;        // Byte offset to first tensor data
    // Followed by tensor table:
    //   struct { char name[64]; uint64_t offset; uint64_t size_bytes; uint32_t shape[4]; }
    // Then contiguous tensor data
};
```

**Tensor naming convention** — templates define what tensor names they expect (in the manifest or as constants in the template source). Adapters map model-native tensor names to template-expected names:

```yaml
# In template manifest.yaml
tensor_layout:
  - "layers.{i}.attention.wq"      # Shape: [dim, dim]
  - "layers.{i}.attention.wk"      # Shape: [dim, kv_dim]
  - "layers.{i}.attention.wv"      # Shape: [dim, kv_dim]
  - "layers.{i}.attention.wo"      # Shape: [dim, dim]
  - "layers.{i}.ffn.w1"            # Shape: [hidden_dim, dim]
  - "layers.{i}.ffn.w2"            # Shape: [dim, hidden_dim]
  - "layers.{i}.ffn.w3"            # Shape: [hidden_dim, dim]
  - "layers.{i}.norm1"             # Shape: [dim]
  - "layers.{i}.norm2"             # Shape: [dim]
  - "token_embedding"              # Shape: [vocab_size, dim]
  - "output_norm"                  # Shape: [dim]
  - "output_proj"                  # Shape: [vocab_size, dim]
```

The adapter's job: map `model.layers.0.self_attn.q_proj.weight` → `layers.0.attention.wq` (or whatever the source model calls it).

### 3.6 Templated Compilation

Templates are Objective-C source files with `%%PLACEHOLDER%%` tokens. The provider substitutes values from the model config, compiles once, caches the binary.

```
Substitution rules:
  - %%KEY%% → config[key]         for required_config and optional_config keys
  - %%NLAYERS%% → config[n_layers]
  - %%DIM%% → config[dim]
  - etc.

The template source uses these as #define values:
  #define DIM %%dim%%
  #define HIDDEN %%hidden_dim%%
  #define HEADS %%n_heads%%
  // ... template handles the rest
```

**Why compile-per-model instead of dynamic:**
- ANE kernels bake tensor shapes into MIL programs. Dynamic dimensions don't help at the hardware level.
- A pre-compiled binary per model is faster than parsing config at runtime.
- The binary hash becomes part of the attestation — anyone can reproduce it from the same template + config.
- No buffer overflow risk from dynamic allocation mistakes.
- Architecture-specific optimizations (fused kernels, layout choices) live in the template, not in runtime branching.

### 3.7 Shared Headers

All templates include the same core headers via `#include`. These are the only non-plugin components:

| Header | Contents |
|--------|----------|
| `ane_runtime.h` | ANE private API wrapper (compile, eval, IOSurface) |
| `ane_mil_gen.h` | MIL text generation helpers (conv, matmul, softmax, etc.) |
| `proof_output.h` | Logits hashing, layer checkpoints, proof JSON serialization |
| `sep_sign.h` | Secure Enclave key management and signing |
| `weight_io.h` | ANEW normalized weight blob loading (mmap + header parse) |

Templates can also include their own headers from their `deps/` directory (e.g., `rope.h`, `ssm.h`, `router.h`). These travel with the template plugin — not managed by the core system.

### 3.8 ANE Primitive Coverage

Any computation that decomposes into these MIL operations can run on ANE. This is the hardware's instruction set — not a list of supported models:

| ANE Primitive | MIL Op | Covers |
|---------------|--------|--------|
| Linear projection | `conv(weight, x, pad="valid")` | Any learned linear layer |
| MatMul | `matmul(A, B)` | Attention scores, similarity, any bilinear op |
| Softmax | `softmax(x, dim)` | Attention weights, classification, any probability distribution |
| Reduce | `reduce_sum`, `reduce_mean`, `reduce_max` | Any normalization, pooling, aggregation |
| Element-wise unary | `sigmoid`, `tanh`, `exp`, `pow`, `sqrt`, `neg` | Any activation or normalization |
| Element-wise binary | `add`, `mul`, `sub`, `div`, `max` | Residuals, gating, any element-wise combination |
| Convolution (N-D) | `conv(weight, x, pad, stride, dilation)` | Spatial convolutions, 1D causal convs, patch embedding |
| Concat / Split | `concat(tensors, dim)`, `split(x, sizes, dim)` | Skip connections, multi-head split, expert routing |
| Reshape / Transpose | `reshape`, `transpose` | Layout transformations between ops |
| Gather / Scatter | `gather(x, indices)` | Embedding lookup, expert selection |

These 10 primitive categories cover **any** differentiable computation graph. If you can express your model as a DAG of these operations, it runs on ANE. The template's job is to compose them in the right order for a given architecture pattern.

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
  → Optional: pattern + adapter hints (auto-detected if omitted)
  → Plugin pipeline: format detect → adapter match → compile
  → Output: { model_id, pattern, config, binary_hash }

GET /v1/models
  → List available models with configs and pricing

GET /v1/models/<id>
  → Model manifest + binary hash

GET /v1/health
  → { status, queue_depth, chip, ane_tops, models_loaded }

GET /v1/capabilities
  → { chip, memory, ane_tops, models, patterns, formats, pricing }

GET /v1/plugins
  → List installed plugins: templates (with pattern + required_config),
    adapters (with name + pattern), format readers (with name + extensions)

POST /v1/plugins/templates
  → Upload a new template plugin (tar.gz: manifest.yaml + .m.tmpl + deps/)
  → Hot-reloads into registry without restart

POST /v1/plugins/adapters
  → Upload a new adapter plugin (.py file)
  → Hot-reloads into registry without restart
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
    "pattern": "decoder_rmsnorm_swiglu",
    "template_version": 1,
    "binary_hash": "sha256:def...",
    "input_hash": "keccak256(canonical_input)",
    "output_hash": "keccak256(canonical_output)",
    "params_hash": "keccak256(canonical_params)",
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

**Level 1 — Same binary.** All providers for a given model compile from the same template plugin + config. Binary hash published in manifest. Validators reject binary hash mismatch. Template version pinned in proof bundle.

**Level 2 — Same MIL programs.** The MIL text is a deterministic function of (template_source, config_values). Same config → same MIL → same ANE program → same compute graph. No runtime branching.

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
├── plugins/
│   ├── templates/                 # Template plugins (auto-discovered)
│   │   ├── decoder_rmsnorm_swiglu/
│   │   │   ├── manifest.yaml
│   │   │   ├── inference.m.tmpl
│   │   │   └── deps/
│   │   ├── ... (any number of templates)
│   ├── adapters/                  # Adapter plugins (auto-discovered)
│   │   ├── hf_causal_lm.py
│   │   ├── gguf_generic.py
│   │   ├── ... (any number of adapters)
│   └── formats/                   # Format reader plugins (auto-discovered)
│       ├── safetensors.py
│       ├── gguf.py
│       ├── ... (any number of format readers)
├── models/
│   ├── sha256_a1b2c3.../
│   │   ├── weights.anew           # Normalized weights (ANEW format, read-only)
│   │   ├── manifest.json          # Config, hashes, pattern, adapter used
│   │   ├── inference              # Compiled binary
│   │   └── tokenizer.bin          # Optional tokenizer
│   └── sha256_d4e5f6.../
│       └── ...
└── logs/
    └── provider.log
```

### 8.2 Model Lifecycle

```
Upload → Format → Adapter → Validate → Export → Template → Compile → Hash → Serve
  │        │        │          │         │         │          │        │       │
  │     plugin   plugin     check     write     plugin     xcrun   sha256   ready
  │     scan     score +    config    ANEW      subst +   clang   binary
  │     detect   select     vs tmpl   blob      select
  │
  └─ reject if:
     - no format plugin can read the file
     - no adapter claims the model (all confidence = 0.0)
     - config fields don't satisfy template's required_config
     - no template installed for the adapter's target pattern
     - dims exceed provider resource limits
     - file too large (configurable limit)
     - tensor count / shapes don't match template's tensor_layout
     - compilation fails (clang error on generated source)
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

plugins:
  templates_dir: "~/.ane_provider/plugins/templates"
  adapters_dir: "~/.ane_provider/plugins/adapters"
  formats_dir: "~/.ane_provider/plugins/formats"
  hot_reload: true                   # Watch dirs for new plugins
  allow_upload: true                 # Allow remote plugin upload via API

models:
  store_path: "~/.ane_provider/models"
  max_disk_gb: 100
  max_upload_gb: 50                  # Single file upload limit
  allowed_patterns: "all"            # Or list: ["decoder_rmsnorm_swiglu", "encoder_bidirectional"]
  resource_limits:                   # Reject models that exceed these
    max_params: 70_000_000_000       # 70B parameters
    max_memory_gb: 32                # Estimated runtime memory
    max_ane_kernels: 500             # Total ANE programs per model

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

The input schema is determined by the model's template `output_type`. The gateway validates input against the template manifest before dispatching.

```
POST /v1/inference HTTP/1.1
Host: provider.example.com:8402
Content-Type: application/json
X-PAYMENT: <x402 payment payload>

{
  "model_id": "sha256:a1b2c3...",
  "input": { ... },             ← Schema defined by template's output_type
  "params": { ... },            ← Optional: temperature, seed, top_p, max_tokens, etc.
  "stream": false,
  "include_logits": false
}
```

Example inputs by output type:

```json
// output_type: "token_sequence" (decoder models)
{ "input": { "tokens": [1, 4532, 817] }, "params": { "max_tokens": 256, "temperature": 0.0, "seed": 42 } }

// output_type: "classification" (encoder models)
{ "input": { "tokens": [101, 2023, 2003, 1037, 3231, 102] } }

// output_type: "embedding" (embedding models)
{ "input": { "tokens": [1, 4532, 817] } }

// output_type: "image" (diffusion models)
{ "input": { "prompt_embedding": [...], "timesteps": 50, "guidance_scale": 7.5 } }

// output_type: "seq2seq" (encoder-decoder models)
{ "input": { "encoder_tokens": [1, 4532], "decoder_tokens": [1] }, "params": { "max_tokens": 128 } }
```

### 10.2 Inference Response

```json
{
  "output": { ... },            // Schema depends on output_type
  "finish_reason": "eos",       // or "max_tokens", "stop_sequence"
  "usage": {
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

Gateway → subprocess communication via stdin/stdout. The protocol is the same regardless of model or template:

```
Gateway writes to stdin:
{
  "command": "inference",
  "weights_path": "/path/to/weights.anew",
  "tokenizer_path": "/path/to/tokenizer.bin",
  "input": { ... },            ← Passed through from client
  "params": { ... }            ← Passed through from client
}
<EOF>

Subprocess writes to stdout:
{
  "output": { ... },            ← Template-defined output
  "logits_hash": "abc...",
  "layer_checkpoints": {"6": "def..."},
  "ane_time_ms": 12.4
}
<EOF>
```

The subprocess reads JSON from stdin, runs inference, writes JSON to stdout, exits. No persistent connection. No shared state. The gateway never interprets the `input` or `output` fields — they are opaque payloads passed between client and compiled template binary.

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
├── core/                            # Core system — model-agnostic, never references specific models
│   ├── gateway/
│   │   ├── server.py                # Flask x402 server
│   │   ├── x402.py                  # x402 payment verification
│   │   ├── scheduler.py             # Job queue + worker pool
│   │   ├── proof.py                 # Proof bundle assembly
│   │   └── posp.py                  # PoSP orchestrator client
│   │
│   ├── sandbox/
│   │   ├── runner.py                # Subprocess exec with sandbox-exec
│   │   ├── ane_sandbox.sb           # macOS sandbox profile
│   │   └── limits.py                # Resource limit enforcement
│   │
│   ├── plugins/
│   │   ├── loader.py                # Plugin auto-discovery (scan dirs, import modules)
│   │   ├── base_template.py         # TemplatePlugin base + manifest parser
│   │   ├── base_adapter.py          # AdapterPlugin base (can_handle, extract_config, export)
│   │   ├── base_format.py           # FormatPlugin base (can_read, read_metadata, iter_tensors)
│   │   └── compiler.py              # Template substitution + clang compilation
│   │
│   ├── store/
│   │   ├── manager.py               # Model store (upload pipeline, eviction, lookup)
│   │   └── anew.py                  # ANEW normalized weight format (read/write)
│   │
│   ├── attestation/
│   │   ├── sep.py                   # Secure Enclave key management
│   │   └── verify.py                # Attestation chain verification
│   │
│   ├── shared_headers/              # C headers included by all templates
│   │   ├── ane_runtime.h            # ANE private API wrapper
│   │   ├── ane_mil_gen.h            # MIL text generation helpers
│   │   ├── proof_output.h           # Logits hashing, proof serialization
│   │   ├── sep_sign.h               # Secure Enclave signing
│   │   └── weight_io.h              # ANEW weight blob loading (mmap)
│   │
│   └── config.py                    # Config loading (YAML)
│
├── plugins/                         # All model-specific knowledge lives here
│   ├── templates/                   # Template plugins (self-describing, auto-discovered)
│   │   ├── decoder_rmsnorm_swiglu/  # Example: handles Llama-pattern models
│   │   │   ├── manifest.yaml
│   │   │   ├── inference.m.tmpl
│   │   │   └── deps/
│   │   │       └── rope.h
│   │   ├── decoder_layernorm_gelu/  # Example: handles GPT-pattern models
│   │   │   ├── manifest.yaml
│   │   │   └── inference.m.tmpl
│   │   ├── .../                     # Any number of templates — drop in to add support
│   │   └── README.md                # How to write a template plugin
│   │
│   ├── adapters/                    # Adapter plugins (auto-discovered .py files)
│   │   ├── hf_causal_lm.py         # Generic HuggingFace causal LM adapter
│   │   ├── hf_seq2seq.py           # Generic HuggingFace seq2seq adapter
│   │   ├── gguf_generic.py         # Generic GGUF model adapter
│   │   ├── .../                     # Any number of adapters — drop in to add support
│   │   └── README.md                # How to write an adapter plugin
│   │
│   └── formats/                     # Format reader plugins (auto-discovered .py files)
│       ├── safetensors_reader.py
│       ├── gguf_reader.py
│       ├── onnx_reader.py
│       ├── .../                     # Any number of format readers — drop in to add support
│       └── README.md                # How to write a format plugin
│
├── requirements.txt
└── README.md                        # Quick start
```

**Key structural property:** The `core/` directory has zero imports from `plugins/`. It only imports plugin base classes and the loader. All model-specific knowledge is in `plugins/` and discovered at runtime. You can delete every plugin and the core still starts — it just can't handle any models until you add plugins back.

---

## 12. Threat Model (Implementation-Specific)

| Threat | Mitigation |
|--------|-----------|
| Malicious model weights (crafted to exploit buffer overflow) | Sandbox: deny network, deny process-exec, memory limit. Adapter validates tensor shapes against template's `tensor_layout` before generating C code. Compilation rejects invalid dims. |
| No plugin can handle uploaded model | Upload rejected with 422 (Unprocessable). Format readers, adapters, and templates all fail gracefully. No partial state left behind. |
| Malicious plugin uploaded via API | Plugin upload requires provider auth. Plugins are Python (adapters/formats) and C (templates) — both reviewed before hot-reload if `allow_upload: true`. Disable via config. Templates are compiled in sandbox test before activation. |
| Adapter returns crafted config to exploit template substitution | Config values are validated against template's `required_config` schema (type, range). Substitution is simple string replacement into `#define` constants — no eval, no format strings. Integer overflow checked pre-substitution. |
| Model too large for device memory | Template manifest declares `resource_hints.memory_per_param_bytes`. Provider estimates total memory (params × bytes + KV-cache + activations) and rejects if exceeding `resource_limits.max_memory_gb`. |
| Denial of service (flood requests) | Bounded queue (503 on overflow). x402 payment required (costs money to spam). Rate limit per wallet. |
| Provider returns cached stale result | Freshness nonce in every request. SEP attestation includes timestamp. PoSP validators use same nonce. |
| Provider runs weaker model (fewer layers) | Binary hash in attestation. Validators compile same template from same config, compare hash. |
| Provider swaps template pattern | Binary hash is tied to specific template source + config. Validators use the same `(pattern, template_version, binary_hash)` triple. Mismatch = slash. |
| Sandbox escape | macOS sandbox-exec is kernel-enforced. ANE access is read-only (no code execution on ANE). Process exits after each job. Same sandbox profile regardless of which template was compiled. |
| Side-channel on model weights | Weights are read-only in sandbox. Network denied. No exfiltration path. |
| Compromised gateway (not subprocess) | Gateway never touches model weights or ANE. It only routes JSON. Attestation is signed by subprocess via SEP. |
| Cross-pattern PoSP mismatch | Validators must use same `(pattern, template_version, binary_hash)`. Different patterns naturally produce different binary hashes, preventing cross-pattern validation. |
| Rogue hot-reloaded plugin | Hot-reload only activates after: (1) plugin passes schema validation, (2) test compilation succeeds (templates), (3) test import succeeds (adapters/formats). Existing active jobs are not affected — they use the already-compiled binary. |

---

## 13. Implementation Sequence

```
Week 1: Plugin system + core pipeline
  ├── Plugin base classes: TemplatePlugin, AdapterPlugin, FormatPlugin
  ├── PluginLoader: directory scan, import, validation
  ├── Shared C headers: ane_runtime.h, ane_mil_gen.h, weight_io.h, proof_output.h
  ├── ANEW format: anew.py (read/write normalized weight blobs)
  ├── Compiler: template substitution + clang invocation
  ├── Store: upload pipeline (format → adapter → compile → cache)
  ├── Runner: subprocess exec with IPC (stdin JSON → stdout JSON)
  ├── First template plugin (decoder_rmsnorm_swiglu) as proof of concept
  ├── First adapter plugin (hf_causal_lm) + first format plugin (safetensors)
  └── Verify: upload any HF causal LM → auto-detect → compile → inference → output

Week 2: Proof + attestation + second template
  ├── proof_output.h: logits_hash + layer_checkpoints in compiled binary output
  ├── sep.py: SEP key gen + signing via Security.framework
  ├── proof.py: assemble proof bundle from subprocess output
  ├── Second template plugin (decoder_layernorm_gelu) to prove the pattern is generic
  ├── Second format plugin (gguf) to prove format plugins are independent
  ├── Plugin hot-reload: watch dirs, validate, activate without restart
  └── Verify: two different model architectures, both produce valid SEP-signed proofs

Week 3: x402 + server + plugin API
  ├── server.py: Flask endpoints (inference, training, models, plugins, health)
  ├── x402.py: payment verification via Coinbase facilitator
  ├── scheduler.py: job queue + worker pool
  ├── Plugin upload endpoints: POST /v1/plugins/templates, /v1/plugins/adapters
  ├── Third template plugin (encoder_bidirectional) — contributed via API upload
  └── Verify: end-to-end paid inference, hot-add a new template via API, serve a new model

Week 4: PoSP + on-chain
  ├── ANEMarketplace.sol (deploy to Base testnet)
  ├── posp.py (commit results, handle challenges)
  ├── Validator mode (re-run jobs on challenge, pattern-aware matching)
  ├── Plugin README docs (how to write a template, adapter, format reader)
  └── Verify: full loop — pay, compute, prove, challenge, settle
         with models using different templates to prove pattern-agnostic verification
```
