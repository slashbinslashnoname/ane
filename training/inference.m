// inference.m — Fast ANE-accelerated inference for Stories110M
// Compiles all linear layers as baked-weight ANE kernels (compile once),
// then runs autoregressive token generation with KV-cache.
// ANE handles: QKV projections, output projection, FFN (W1/W3/W2), classifier
// CPU handles: RMSNorm, RoPE, causal attention, SiLU, residuals, sampling
#include "stories_io.h"
#include "stories_mil.h"
#include "stories_cpu_ops.h"
#include <time.h>

#define MODEL_PATH "../../assets/models/stories110M.bin"
#define CKPT_PATH "ane_stories110M_ckpt.bin"

// Inference uses S=1 for per-token ANE kernels (projections)
// Attention is done on CPU with KV-cache
#define INF_S 1

// ===== ANE inference kernels per layer =====
typedef struct {
    // Fused QKV: one ANE dispatch → Q,K,V (S=1)
    Kern *qkv;
    // Output projection (S=1)
    Kern *wo;
    // FFN: fused W1+W3 up, W2 down (S=1)
    Kern *ffn_up;
    Kern *ffn_down;
} InfLayerKernels;

// ===== MIL generators for inference kernels =====

// Fused QKV conv: x[1,DIM,1,1] → Q,K,V via 3 parallel convs
// Output: single IOSurface with concat [1, 3*DIM, 1, 1]
static NSString *gen_inf_qkv(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, 1]> x) {\n", DIM];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> q = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=x)[name=string(\"cq\")];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> k = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=x)[name=string(\"ck\")];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> v = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=x)[name=string(\"cv\")];\n", DIM];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> out = concat(axis=cax,interleave=cid,values=(q,k,v))[name=string(\"cat\")];\n", 3*DIM];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Fused FFN up: x[1,DIM,1,1] → h1,h3 via parallel W1,W3 convs + SiLU + gate
// Does: h1=W1@x, h3=W3@x, out=silu(h1)*h3
static NSString *gen_inf_ffn_up(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, 1]> x) {\n", DIM];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=x)[name=string(\"c1\")];\n", HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=x)[name=string(\"c3\")];\n", HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> out = mul(x=silu,y=h3)[name=string(\"gt\")];\n", HIDDEN];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN down: gate[1,HIDDEN,1,1] → W2@gate → [1,DIM,1,1]
static NSString *gen_inf_ffn_down(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, 1]> x) {\n", HIDDEN];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", DIM,HIDDEN,DIM,HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=x)[name=string(\"c2\")];\n", DIM];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Output projection: x[1,DIM,1,1] → Wo@x → [1,DIM,1,1]
static NSString *gen_inf_wo(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, 1]> x) {\n", DIM];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=x)[name=string(\"co\")];\n", DIM];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Classifier: x[1,DIM,1,1] → wcls@x → [1,VOCAB,1,1]
static NSString *gen_inf_cls(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, 1]> x) {\n", DIM];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wc = const()[name=string(\"Wc\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wcls.bin\"), offset=uint64(64)))];\n", VOCAB,DIM,VOCAB,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wc,x=x)[name=string(\"cc\")];\n", VOCAB];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ===== Weight loading =====
static LayerWeights g_lw[NLAYERS];
static float g_rms_final[DIM];
static float g_embed[VOCAB * DIM];

static bool load_weights(const char *ckpt, const char *model) {
    // Try checkpoint first
    FILE *f = fopen(ckpt, "rb");
    if (f) {
        CkptHdr hdr;
        fread(&hdr, sizeof(hdr), 1, f);
        if (hdr.magic == 0x424C5A54 && hdr.version == 2) {
            printf("Loading checkpoint: step=%d loss=%.4f\n", hdr.step, hdr.loss);
            for (int L = 0; L < NLAYERS; L++) {
                fread(g_lw[L].Wq, 4, WQ_SZ, f);
                fread(g_lw[L].Wk, 4, WQ_SZ, f);
                fread(g_lw[L].Wv, 4, WQ_SZ, f);
                fread(g_lw[L].Wo, 4, WO_SZ, f);
                fread(g_lw[L].W1, 4, W1_SZ, f);
                fread(g_lw[L].W2, 4, W2_SZ, f);
                fread(g_lw[L].W3, 4, W3_SZ, f);
                fread(g_lw[L].rms_att, 4, DIM, f);
                fread(g_lw[L].rms_ffn, 4, DIM, f);
                // Skip Adam state (m,v for each weight)
                fseek(f, (long)(2*(WQ_SZ+WQ_SZ+WQ_SZ+WO_SZ+W1_SZ+W2_SZ+W3_SZ+DIM+DIM))*4, SEEK_CUR);
            }
            fread(g_rms_final, 4, DIM, f);
            fseek(f, 2*DIM*4, SEEK_CUR); // skip Adam for rms_final
            fread(g_embed, 4, VOCAB*DIM, f);
            fclose(f);
            return true;
        }
        fclose(f);
    }
    // Fall back to pretrained
    return load_pretrained(g_lw, g_rms_final, g_embed, model);
}

// ===== Compile all ANE inference kernels =====
static InfLayerKernels g_lk[NLAYERS];
static Kern *g_cls_kern = NULL;

static bool compile_inf_kernels(void) {
    printf("Compiling ANE inference kernels (4 per layer + 1 classifier)...\n");
    uint64_t t0 = mach_absolute_time();

    for (int L = 0; L < NLAYERS; L++) {
        // QKV fused
        g_lk[L].qkv = compile_kern_mil_w(gen_inf_qkv(), (@{
            @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":build_blob(g_lw[L].Wq,DIM,DIM)},
            @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":build_blob(g_lw[L].Wk,DIM,DIM)},
            @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":build_blob(g_lw[L].Wv,DIM,DIM)},
        }), DIM*1*2, 3*DIM*1*2);
        if (!g_lk[L].qkv) { printf("L%d QKV compile failed\n", L); return false; }

        // Wo
        g_lk[L].wo = compile_kern_mil_w(gen_inf_wo(), (@{
            @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":build_blob(g_lw[L].Wo,DIM,DIM)},
        }), DIM*1*2, DIM*1*2);
        if (!g_lk[L].wo) { printf("L%d Wo compile failed\n", L); return false; }

        // FFN up (W1+W3+SiLU+gate fused)
        g_lk[L].ffn_up = compile_kern_mil_w(gen_inf_ffn_up(), (@{
            @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":build_blob(g_lw[L].W1,HIDDEN,DIM)},
            @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":build_blob(g_lw[L].W3,HIDDEN,DIM)},
        }), DIM*1*2, HIDDEN*1*2);
        if (!g_lk[L].ffn_up) { printf("L%d FFN up compile failed\n", L); return false; }

        // FFN down (W2)
        g_lk[L].ffn_down = compile_kern_mil_w(gen_inf_ffn_down(), (@{
            @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":build_blob(g_lw[L].W2,DIM,HIDDEN)},
        }), HIDDEN*1*2, DIM*1*2);
        if (!g_lk[L].ffn_down) { printf("L%d FFN down compile failed\n", L); return false; }

        printf("  Layer %d OK\n", L);
    }

    // Classifier
    g_cls_kern = compile_kern_mil_w(gen_inf_cls(), (@{
        @"@model_path/weights/wcls.bin": @{@"offset":@0, @"data":build_blob(g_embed,VOCAB,DIM)},
    }), DIM*1*2, VOCAB*1*2);
    if (!g_cls_kern) printf("  Classifier: compile failed, using CPU fallback\n");
    else printf("  Classifier OK\n");

    double ms = tb_ms(mach_absolute_time() - t0);
    int n_kernels = NLAYERS * 4 + (g_cls_kern ? 1 : 0);
    printf("Compiled %d kernels in %.1f ms (%.1f ms/kernel)\n", n_kernels, ms, ms/n_kernels);
    return true;
}

// ===== ANE eval helpers for S=1 =====
// Write fp32 data [DIM] to IOSurface as fp16 [1,DIM,1,1]
static void inf_write(Kern *k, const float *data, int ch) {
    IOSurfaceLock(k->ioIn, 0, NULL);
    cvt_f32_f16((_Float16*)IOSurfaceGetBaseAddress(k->ioIn), data, ch);
    IOSurfaceUnlock(k->ioIn, 0, NULL);
}
// Read fp16 IOSurface → fp32 data
static void inf_read(Kern *k, float *data, int ch_off, int ch) {
    IOSurfaceLock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
    cvt_f16_f32(data, (_Float16*)IOSurfaceGetBaseAddress(k->ioOut) + ch_off, ch);
    IOSurfaceUnlock(k->ioOut, kIOSurfaceLockReadOnly, NULL);
}

// ===== CPU ops for single-token inference =====
// RMSNorm for a single vector x[DIM] with weights w[DIM]
static void rmsnorm_1(float *out, const float *x, const float *w) {
    float ss = 0;
    for (int i = 0; i < DIM; i++) ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / DIM + 1e-5f);
    for (int i = 0; i < DIM; i++) out[i] = x[i] * ss * w[i];
}

// RoPE for single position
static void rope_1(float *q, float *k, int pos) {
    for (int h = 0; h < HEADS; h++) {
        for (int i = 0; i < HD; i += 2) {
            float freq = 1.0f / powf(10000.0f, (float)i / HD);
            float val = pos * freq;
            float cos_v = cosf(val), sin_v = sinf(val);
            int off = h * HD + i;
            float q0 = q[off], q1 = q[off+1];
            q[off]   = q0 * cos_v - q1 * sin_v;
            q[off+1] = q0 * sin_v + q1 * cos_v;
            float k0 = k[off], k1 = k[off+1];
            k[off]   = k0 * cos_v - k1 * sin_v;
            k[off+1] = k0 * sin_v + k1 * cos_v;
        }
    }
}

// Causal attention with KV-cache for single query token
// q[DIM]: current query
// k_cache[max_seq*DIM], v_cache[max_seq*DIM]: cached keys/values
// pos: current position (number of cached KV pairs including current)
// out[DIM]: attention output
static void attention_1(float *out, const float *q,
                        const float *k_cache, const float *v_cache,
                        int pos) {
    float scale = 1.0f / sqrtf((float)HD);
    for (int h = 0; h < HEADS; h++) {
        // Compute scores for this head
        float max_score = -1e9f;
        float scores[pos + 1];
        for (int t = 0; t <= pos; t++) {
            float dot = 0;
            for (int i = 0; i < HD; i++)
                dot += q[h*HD + i] * k_cache[t*DIM + h*HD + i];
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }
        // Softmax
        float sum = 0;
        for (int t = 0; t <= pos; t++) {
            scores[t] = expf(scores[t] - max_score);
            sum += scores[t];
        }
        float inv_sum = 1.0f / sum;
        // Weighted sum of values
        for (int i = 0; i < HD; i++) {
            float val = 0;
            for (int t = 0; t <= pos; t++)
                val += scores[t] * inv_sum * v_cache[t*DIM + h*HD + i];
            out[h*HD + i] = val;
        }
    }
}

// ===== Tokenizer (minimal BPE decode from tokenizer.bin) =====
typedef struct { char *str; float score; } TokenInfo;
static TokenInfo *g_vocab = NULL;
static int g_vocab_size = 0;

static bool load_tokenizer(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    int max_tok_len;
    fread(&g_vocab_size, 4, 1, f);
    fread(&max_tok_len, 4, 1, f);
    g_vocab = (TokenInfo*)calloc(g_vocab_size, sizeof(TokenInfo));
    for (int i = 0; i < g_vocab_size; i++) {
        fread(&g_vocab[i].score, 4, 1, f);
        int len;
        fread(&len, 4, 1, f);
        g_vocab[i].str = (char*)malloc(len + 1);
        fread(g_vocab[i].str, 1, len, f);
        g_vocab[i].str[len] = '\0';
    }
    fclose(f);
    return true;
}

static void decode_token(int token) {
    if (token < 0 || token >= g_vocab_size) return;
    const char *s = g_vocab[token].str;
    // Handle raw byte tokens like <0xXX>
    if (s[0] == '<' && s[1] == '0' && s[2] == 'x') {
        int byte_val = (int)strtol(s + 3, NULL, 16);
        putchar(byte_val);
    } else {
        // Replace SentencePiece's ▁ (U+2581) with space
        while (*s) {
            if ((unsigned char)s[0] == 0xE2 && (unsigned char)s[1] == 0x96 && (unsigned char)s[2] == 0x81) {
                putchar(' ');
                s += 3;
            } else {
                putchar(*s);
                s++;
            }
        }
    }
    fflush(stdout);
}

// ===== Sampling =====
static int sample_token(const float *logits, float temperature) {
    if (temperature < 0.01f) {
        // Greedy
        int best = 0;
        for (int i = 1; i < VOCAB; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }
    // Temperature-scaled softmax sampling
    float max_v = logits[0];
    for (int i = 1; i < VOCAB; i++)
        if (logits[i] > max_v) max_v = logits[i];

    float sum = 0;
    float *probs = (float*)malloc(VOCAB * sizeof(float));
    for (int i = 0; i < VOCAB; i++) {
        probs[i] = expf((logits[i] - max_v) / temperature);
        sum += probs[i];
    }

    float r = (float)rand() / RAND_MAX * sum;
    float cum = 0;
    int tok = 0;
    for (int i = 0; i < VOCAB; i++) {
        cum += probs[i];
        if (cum >= r) { tok = i; break; }
    }
    free(probs);
    return tok;
}

// ===== Top-p (nucleus) sampling =====
typedef struct { float prob; int idx; } ProbIdx;
static int cmp_prob_desc(const void *a, const void *b) {
    float pa = ((ProbIdx*)a)->prob, pb = ((ProbIdx*)b)->prob;
    return (pb > pa) - (pb < pa);
}

static int sample_top_p(const float *logits, float temperature, float top_p) {
    float max_v = logits[0];
    for (int i = 1; i < VOCAB; i++)
        if (logits[i] > max_v) max_v = logits[i];

    ProbIdx *pi = (ProbIdx*)malloc(VOCAB * sizeof(ProbIdx));
    float sum = 0;
    for (int i = 0; i < VOCAB; i++) {
        pi[i].prob = expf((logits[i] - max_v) / temperature);
        pi[i].idx = i;
        sum += pi[i].prob;
    }
    for (int i = 0; i < VOCAB; i++) pi[i].prob /= sum;

    qsort(pi, VOCAB, sizeof(ProbIdx), cmp_prob_desc);

    // Find cutoff
    float cumulative = 0;
    int cutoff = VOCAB;
    for (int i = 0; i < VOCAB; i++) {
        cumulative += pi[i].prob;
        if (cumulative > top_p) { cutoff = i + 1; break; }
    }

    // Resample within top-p
    float r = (float)rand() / RAND_MAX * cumulative;
    float cum = 0;
    int tok = pi[0].idx;
    for (int i = 0; i < cutoff; i++) {
        cum += pi[i].prob;
        if (cum >= r) { tok = pi[i].idx; break; }
    }
    free(pi);
    return tok;
}

// ===== Main inference loop =====
static void generate(int max_tokens, float temperature, float top_p, bool use_ane) {
    // KV cache: [max_tokens][DIM] per layer
    float **k_cache = (float**)malloc(NLAYERS * sizeof(float*));
    float **v_cache = (float**)malloc(NLAYERS * sizeof(float*));
    for (int L = 0; L < NLAYERS; L++) {
        k_cache[L] = (float*)calloc(max_tokens * DIM, sizeof(float));
        v_cache[L] = (float*)calloc(max_tokens * DIM, sizeof(float));
    }

    float *x = (float*)malloc(DIM * sizeof(float));
    float *xnorm = (float*)malloc(DIM * sizeof(float));
    float *q = (float*)malloc(DIM * sizeof(float));
    float *k = (float*)malloc(DIM * sizeof(float));
    float *v = (float*)malloc(DIM * sizeof(float));
    float *attn_out = (float*)malloc(DIM * sizeof(float));
    float *ffn_gate = (float*)malloc(HIDDEN * sizeof(float));
    float *ffn_out = (float*)malloc(DIM * sizeof(float));
    float *wo_out = (float*)malloc(DIM * sizeof(float));
    float *logits = (float*)malloc(VOCAB * sizeof(float));

    int token = 1; // BOS
    uint64_t total_start = mach_absolute_time();
    double total_ane_ms = 0;

    for (int pos = 0; pos < max_tokens; pos++) {
        // Embedding lookup
        memcpy(x, g_embed + token * DIM, DIM * sizeof(float));

        for (int L = 0; L < NLAYERS; L++) {
            // RMSNorm (CPU)
            rmsnorm_1(xnorm, x, g_lw[L].rms_att);

            if (use_ane) {
                // QKV projection (ANE — single fused dispatch)
                uint64_t ane_t0 = mach_absolute_time();
                inf_write(g_lk[L].qkv, xnorm, DIM);
                ane_eval(g_lk[L].qkv);
                inf_read(g_lk[L].qkv, q, 0, DIM);
                inf_read(g_lk[L].qkv, k, DIM, DIM);
                inf_read(g_lk[L].qkv, v, 2*DIM, DIM);
                total_ane_ms += tb_ms(mach_absolute_time() - ane_t0);
            } else {
                // CPU fallback
                cblas_sgemv(CblasRowMajor, CblasNoTrans, DIM, DIM, 1.0f, g_lw[L].Wq, DIM, xnorm, 1, 0.0f, q, 1);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, DIM, DIM, 1.0f, g_lw[L].Wk, DIM, xnorm, 1, 0.0f, k, 1);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, DIM, DIM, 1.0f, g_lw[L].Wv, DIM, xnorm, 1, 0.0f, v, 1);
            }

            // RoPE (CPU)
            rope_1(q, k, pos);

            // Store K,V in cache
            memcpy(k_cache[L] + pos * DIM, k, DIM * sizeof(float));
            memcpy(v_cache[L] + pos * DIM, v, DIM * sizeof(float));

            // Attention with KV-cache (CPU)
            attention_1(attn_out, q, k_cache[L], v_cache[L], pos);

            if (use_ane) {
                // Output projection (ANE)
                uint64_t ane_t0 = mach_absolute_time();
                inf_write(g_lk[L].wo, attn_out, DIM);
                ane_eval(g_lk[L].wo);
                inf_read(g_lk[L].wo, wo_out, 0, DIM);
                total_ane_ms += tb_ms(mach_absolute_time() - ane_t0);
            } else {
                cblas_sgemv(CblasRowMajor, CblasNoTrans, DIM, DIM, 1.0f, g_lw[L].Wo, DIM, attn_out, 1, 0.0f, wo_out, 1);
            }

            // Residual
            for (int i = 0; i < DIM; i++) x[i] += wo_out[i];

            // FFN RMSNorm (CPU)
            rmsnorm_1(xnorm, x, g_lw[L].rms_ffn);

            if (use_ane) {
                // FFN up: W1,W3,SiLU,gate all fused (ANE)
                uint64_t ane_t0 = mach_absolute_time();
                inf_write(g_lk[L].ffn_up, xnorm, DIM);
                ane_eval(g_lk[L].ffn_up);
                inf_read(g_lk[L].ffn_up, ffn_gate, 0, HIDDEN);

                // FFN down: W2 (ANE)
                inf_write(g_lk[L].ffn_down, ffn_gate, HIDDEN);
                ane_eval(g_lk[L].ffn_down);
                inf_read(g_lk[L].ffn_down, ffn_out, 0, DIM);
                total_ane_ms += tb_ms(mach_absolute_time() - ane_t0);
            } else {
                // CPU FFN
                float *h1 = (float*)malloc(HIDDEN * sizeof(float));
                float *h3 = (float*)malloc(HIDDEN * sizeof(float));
                cblas_sgemv(CblasRowMajor, CblasNoTrans, HIDDEN, DIM, 1.0f, g_lw[L].W1, DIM, xnorm, 1, 0.0f, h1, 1);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, HIDDEN, DIM, 1.0f, g_lw[L].W3, DIM, xnorm, 1, 0.0f, h3, 1);
                for (int i = 0; i < HIDDEN; i++)
                    ffn_gate[i] = h1[i] / (1.0f + expf(-h1[i])) * h3[i];
                free(h1); free(h3);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, DIM, HIDDEN, 1.0f, g_lw[L].W2, HIDDEN, ffn_gate, 1, 0.0f, ffn_out, 1);
            }

            // Residual
            for (int i = 0; i < DIM; i++) x[i] += ffn_out[i];
        }

        // Final RMSNorm
        rmsnorm_1(xnorm, x, g_rms_final);

        // Classifier
        if (use_ane && g_cls_kern) {
            uint64_t ane_t0 = mach_absolute_time();
            inf_write(g_cls_kern, xnorm, DIM);
            ane_eval(g_cls_kern);
            inf_read(g_cls_kern, logits, 0, VOCAB);
            total_ane_ms += tb_ms(mach_absolute_time() - ane_t0);
        } else {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, VOCAB, DIM, 1.0f, g_embed, DIM, xnorm, 1, 0.0f, logits, 1);
        }

        // Sample next token
        int next;
        if (top_p < 1.0f)
            next = sample_top_p(logits, temperature, top_p);
        else
            next = sample_token(logits, temperature);

        // Print token (skip BOS)
        if (pos > 0 || next != 1)
            decode_token(next);

        // EOS
        if (next == 2) break;

        token = next;
    }

    double total_ms = tb_ms(mach_absolute_time() - total_start);
    int tokens_generated = max_tokens; // approximate
    printf("\n\n--- %.1f ms total, %.1f ms/token, ANE: %.1f ms (%.0f%%)\n",
           total_ms, total_ms / tokens_generated,
           total_ane_ms, total_ane_ms / total_ms * 100);

    // Cleanup
    for (int L = 0; L < NLAYERS; L++) { free(k_cache[L]); free(v_cache[L]); }
    free(k_cache); free(v_cache);
    free(x); free(xnorm); free(q); free(k); free(v);
    free(attn_out); free(ffn_gate); free(ffn_out); free(wo_out); free(logits);
}

// ===== Usage =====
static void usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH    Model weights (default: %s)\n", MODEL_PATH);
    printf("  --ckpt PATH     Checkpoint file (default: %s)\n", CKPT_PATH);
    printf("  --tokens N      Max tokens to generate (default: 256)\n");
    printf("  --temp FLOAT    Temperature (default: 0.8, 0=greedy)\n");
    printf("  --top-p FLOAT   Top-p nucleus sampling (default: 0.9)\n");
    printf("  --cpu            Use CPU-only (BLAS) instead of ANE\n");
    printf("  --tokenizer PATH Tokenizer file (default: tokenizer.bin)\n");
    printf("  --seed N         Random seed\n");
}

int main(int argc, char **argv) {
    @autoreleasepool {
        mach_timebase_info(&g_tb);

        const char *model_path = MODEL_PATH;
        const char *ckpt_path = CKPT_PATH;
        const char *tok_path = "tokenizer.bin";
        int max_tokens = 256;
        float temperature = 0.8f;
        float top_p = 0.9f;
        bool use_ane = true;
        unsigned int seed = (unsigned int)time(NULL);

        for (int i = 1; i < argc; i++) {
            if (!strcmp(argv[i], "--model") && i+1 < argc) model_path = argv[++i];
            else if (!strcmp(argv[i], "--ckpt") && i+1 < argc) ckpt_path = argv[++i];
            else if (!strcmp(argv[i], "--tokens") && i+1 < argc) max_tokens = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--temp") && i+1 < argc) temperature = atof(argv[++i]);
            else if (!strcmp(argv[i], "--top-p") && i+1 < argc) top_p = atof(argv[++i]);
            else if (!strcmp(argv[i], "--cpu")) use_ane = false;
            else if (!strcmp(argv[i], "--tokenizer") && i+1 < argc) tok_path = argv[++i];
            else if (!strcmp(argv[i], "--seed") && i+1 < argc) seed = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) { usage(argv[0]); return 0; }
        }
        srand(seed);

        printf("ANE Inference — Stories110M\n");
        printf("  Mode: %s\n", use_ane ? "ANE" : "CPU (BLAS)");
        printf("  Max tokens: %d, temp: %.2f, top_p: %.2f, seed: %u\n",
               max_tokens, temperature, top_p, seed);

        // Allocate layer weights
        for (int L = 0; L < NLAYERS; L++) g_lw[L] = layer_weights_alloc();

        // Load weights
        if (!load_weights(ckpt_path, model_path)) {
            fprintf(stderr, "Failed to load weights\n");
            return 1;
        }

        // Load tokenizer
        if (!load_tokenizer(tok_path)) {
            printf("Warning: tokenizer.bin not found, using raw token IDs\n");
        }

        if (use_ane) {
            // Init ANE
            ane_init();

            // Compile kernels (one-time cost)
            if (!compile_inf_kernels()) {
                fprintf(stderr, "ANE compilation failed, falling back to CPU\n");
                use_ane = false;
            }
        }

        printf("\n--- Generated text ---\n");
        generate(max_tokens, temperature, top_p, use_ane);

        // Cleanup
        for (int L = 0; L < NLAYERS; L++) layer_weights_free(&g_lw[L]);
        if (g_vocab) {
            for (int i = 0; i < g_vocab_size; i++) free(g_vocab[i].str);
            free(g_vocab);
        }
    }
    return 0;
}
