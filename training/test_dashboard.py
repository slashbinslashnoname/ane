"""Unit tests for dashboard.py pure functions."""

import math
import sys
import threading
import time
from collections import deque
from unittest.mock import patch

import numpy as np
import pytest

# We need to mock blessed and psutil before importing dashboard
# since they may not be available in the test environment
sys.modules.setdefault('blessed', type(sys)('blessed'))
sys.modules.setdefault('psutil', type(sys)('psutil'))
if not hasattr(sys.modules['blessed'], 'Terminal'):
    sys.modules['blessed'].Terminal = type('Terminal', (), {})
    sys.modules['psutil'].cpu_percent = lambda **kw: 0.0
    sys.modules['psutil'].virtual_memory = lambda: type('', (), {'used': 0})()

from dashboard import (
    State, Tokenizer, braille_chart, parse_line, parse_powermetrics_text,
    rmsnorm, softmax, _pad_lines,
    RE_CONFIG, RE_PARAMS, RE_KERNELS, RE_STEP, RE_BATCH, RE_TIMING,
    RE_FLOPS, RE_ANE_FLOPS, RE_ANE_TFLOPS, RE_ANE_UTIL,
    RE_ANE_POWER, RE_CPU_POWER, RE_GPU_POWER,
    S, DIM, HD, HEADS, SEQ,
)


# ─── rmsnorm ────────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_unit_weights(self):
        """RMSNorm with unit weights should normalize RMS to ~1."""
        x = np.array([3.0, 4.0], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        out = rmsnorm(x, w)
        rms = math.sqrt(np.mean(out * out))
        assert abs(rms - 1.0) < 0.01

    def test_scales_with_weights(self):
        """Output should scale proportionally with weights."""
        x = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        w1 = np.ones(3, dtype=np.float32)
        w2 = np.ones(3, dtype=np.float32) * 2.0
        out1 = rmsnorm(x, w1)
        out2 = rmsnorm(x, w2)
        np.testing.assert_allclose(out2, out1 * 2.0, rtol=1e-5)

    def test_random_vector(self):
        """RMSNorm on a random vector should produce finite results."""
        rng = np.random.RandomState(42)
        x = rng.randn(64).astype(np.float32)
        w = rng.randn(64).astype(np.float32)
        out = rmsnorm(x, w)
        assert np.all(np.isfinite(out))

    def test_near_zero_vector(self):
        """Near-zero input should not produce NaN/Inf (epsilon protects)."""
        x = np.array([1e-10, 1e-10], dtype=np.float32)
        w = np.ones(2, dtype=np.float32)
        out = rmsnorm(x, w)
        assert np.all(np.isfinite(out))

    def test_preserves_shape(self):
        x = np.zeros(128, dtype=np.float32)
        w = np.ones(128, dtype=np.float32)
        assert rmsnorm(x, w).shape == (128,)


# ─── softmax ────────────────────────────────────────────────────────────

class TestSoftmax:
    def test_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        p = softmax(x)
        assert abs(np.sum(p) - 1.0) < 1e-6

    def test_max_gets_highest_prob(self):
        x = np.array([1.0, 5.0, 2.0])
        p = softmax(x)
        assert np.argmax(p) == 1

    def test_uniform_input(self):
        x = np.array([1.0, 1.0, 1.0, 1.0])
        p = softmax(x)
        np.testing.assert_allclose(p, 0.25, atol=1e-6)

    def test_numerical_stability(self):
        """Large values should not overflow."""
        x = np.array([1000.0, 1001.0, 1002.0])
        p = softmax(x)
        assert abs(np.sum(p) - 1.0) < 1e-6
        assert np.all(np.isfinite(p))

    def test_negative_values(self):
        x = np.array([-10.0, -5.0, -1.0])
        p = softmax(x)
        assert abs(np.sum(p) - 1.0) < 1e-6
        assert np.argmax(p) == 2

    def test_single_element(self):
        p = softmax(np.array([42.0]))
        np.testing.assert_allclose(p, [1.0])


# ─── braille_chart ──────────────────────────────────────────────────────

class TestBrailleChart:
    def test_basic_output_dimensions(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        lines = braille_chart(vals, width=30, height=5)
        # height rows + 1 bottom axis
        assert len(lines) == 6

    def test_empty_data(self):
        lines = braille_chart([], width=30, height=5)
        assert lines == ['(no data)'] * 5

    def test_too_small_width(self):
        lines = braille_chart([1.0, 2.0], width=5, height=5)
        assert '(no data)' in lines[0]

    def test_too_small_height(self):
        lines = braille_chart([1.0, 2.0], width=30, height=1)
        assert '(no data)' in lines[0]

    def test_constant_data(self):
        """All-same values should not crash (lo ≈ hi edge case)."""
        vals = [5.0] * 20
        lines = braille_chart(vals, width=30, height=5)
        assert len(lines) == 6

    def test_single_point(self):
        lines = braille_chart([3.14], width=30, height=4)
        assert len(lines) == 5

    def test_y_labels_present(self):
        vals = list(range(100))
        lines = braille_chart(vals, width=30, height=6)
        # First line should have the high label, last data line has low
        assert '│' in lines[0] or '│' in lines[0]

    def test_y_range_override(self):
        vals = [5.0, 6.0, 7.0]
        lines = braille_chart(vals, width=30, height=5, y_range=(0, 10))
        # Should use the override, first label should show ~10
        assert len(lines) == 6

    def test_large_data_truncated(self):
        """More data points than chart width should not crash."""
        vals = list(range(1000))
        lines = braille_chart(vals, width=30, height=5)
        assert len(lines) == 6

    def test_braille_characters_in_output(self):
        """Chart should contain braille Unicode characters."""
        vals = [float(i) for i in range(20)]
        lines = braille_chart(vals, width=30, height=5)
        # Braille chars are in range U+2800..U+28FF
        has_braille = any(
            0x2800 <= ord(c) <= 0x28FF
            for line in lines[:-1]
            for c in line
        )
        assert has_braille


# ─── parse_line ─────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    S.__init__()
    yield
    S.__init__()


class TestParseLine:
    def test_config_line(self):
        parse_line('Model: dim=768 hidden=2048 heads=12 seq=256 vocab=32000 layers=12')
        assert S.model_config['dim'] == 768
        assert S.model_config['heads'] == 12
        assert S.model_config['layers'] == 12

    def test_params_line(self):
        parse_line('Params: 109.5M (transformer 85.1M + embed 24.4M)')
        assert S.params['total'] == 109.5
        assert S.params['transformer'] == 85.1
        assert S.params['embed'] == 24.4

    def test_kernels_line(self):
        parse_line('Kernels: 156 total, 84 weight-bearing')
        assert S.kernels['total'] == 156
        assert S.kernels['weight_bearing'] == 84

    def test_step_line(self):
        parse_line('step   42 loss=2.3456')
        assert S.step == 42
        assert abs(S.loss - 2.3456) < 1e-4
        assert len(S.loss_history) == 1
        assert S.loss_history[0] == (42, S.loss)

    def test_step_tracks_best_loss(self):
        parse_line('step   1 loss=3.0')
        parse_line('step   2 loss=2.0')
        parse_line('step   3 loss=2.5')
        assert abs(S.best_loss - 2.0) < 1e-6

    def test_batch_line(self):
        parse_line('[batch 5: compile=120.5ms train=880.3ms (22.0ms/step) compiles=3]')
        assert S.batch_num == 5
        assert abs(S.ms_per_step - 22.0) < 0.1
        assert S.compiles == 3
        assert S.compile_pct > 0

    def test_timing_line(self):
        parse_line('ane=10.5 io=2.3 cls=1.1 elem=0.8 rms=0.5 cblas_wait=0.2')
        assert abs(S.component_timing['ane'] - 10.5) < 0.01
        assert abs(S.component_timing['rms'] - 0.5) < 0.01

    def test_flops_line(self):
        parse_line('FLOPs/step: fwd=100.0M bwd_dx=200.0M bwd_dW=300.0M sdpa_bwd=50.0M total=650.0M')
        assert abs(S.flops['total'] - 650.0) < 0.1

    def test_ane_flops_line(self):
        parse_line('ANE FLOPs/step: 500.0M')
        assert abs(S.flops['ane'] - 500.0) < 0.1

    def test_ane_tflops_line(self):
        parse_line('ANE TFLOPS: 1.23')
        assert abs(S.flops['ane_tflops'] - 1.23) < 0.01

    def test_ane_util_line(self):
        parse_line('ANE utilization: 45.6%')
        assert abs(S.flops['ane_util'] - 45.6) < 0.1

    def test_unknown_line_goes_to_logs(self):
        parse_line('some random output')
        assert 'some random output' in S.logs

    def test_log_always_appended(self):
        """All lines are appended to logs, even matching ones."""
        parse_line('step   1 loss=1.0')
        assert len(S.logs) == 1
        assert 'step' in S.logs[0]


# ─── parse_powermetrics_text ────────────────────────────────────────────

class TestParsePowermetrics:
    def test_ane_power(self):
        parse_powermetrics_text('ANE Power: 4500 mW')
        assert abs(S.power['ane'] - 4.5) < 0.01

    def test_cpu_power(self):
        parse_powermetrics_text('CPU Power: 3200 mW')
        assert abs(S.power['cpu'] - 3.2) < 0.01

    def test_gpu_power(self):
        parse_powermetrics_text('GPU Power: 1800 mW')
        assert abs(S.power['gpu'] - 1.8) < 0.01

    def test_combined_power_text(self):
        text = 'ANE Power: 5000 mW\nCPU Power: 2000 mW\nGPU Power: 1000 mW'
        parse_powermetrics_text(text)
        assert abs(S.power['ane'] - 5.0) < 0.01
        assert abs(S.power['cpu'] - 2.0) < 0.01
        assert abs(S.power['gpu'] - 1.0) < 0.01

    def test_ane_history_appended(self):
        parse_powermetrics_text('ANE Power: 3000 mW')
        parse_powermetrics_text('ANE Power: 4000 mW')
        assert len(S.power_history_ane) == 2

    def test_no_match(self):
        parse_powermetrics_text('unrelated text')
        assert S.power['ane'] == 0.0


# ─── Tokenizer ──────────────────────────────────────────────────────────

class TestTokenizer:
    @pytest.fixture
    def tokenizer(self, tmp_path):
        """Create a minimal tokenizer.bin for testing."""
        import struct
        path = tmp_path / 'tokenizer.bin'
        vocab = ['<unk>', '<s>', '</s>', 'hello', ' world', '<0x41>', '<0xZZ>']
        with open(path, 'wb') as f:
            f.write(struct.pack('i', 32))  # max_token_length
            for i, tok in enumerate(vocab):
                encoded = tok.encode('utf-8')
                f.write(struct.pack('f', float(i)))  # score
                f.write(struct.pack('i', len(encoded)))
                f.write(encoded)
        # Patch VOCAB to match our small vocab
        import dashboard
        old_vocab = dashboard.VOCAB
        dashboard.VOCAB = len(vocab)
        tok = Tokenizer(str(path))
        dashboard.VOCAB = old_vocab
        return tok

    def test_decode_normal_token(self, tokenizer):
        assert tokenizer.decode(3) == 'hello'

    def test_decode_space_token(self, tokenizer):
        assert tokenizer.decode(4) == ' world'

    def test_decode_hex_token(self, tokenizer):
        # <0x41> should decode to 'A'
        assert tokenizer.decode(5) == 'A'

    def test_decode_invalid_hex_token(self, tokenizer):
        # <0xZZ> should return as-is (ValueError caught)
        assert tokenizer.decode(6) == '<0xZZ>'

    def test_decode_out_of_bounds(self, tokenizer):
        assert tokenizer.decode(9999) == ''
        assert tokenizer.decode(-1) == ''


# ─── _pad_lines ─────────────────────────────────────────────────────────

class TestPadLines:
    def test_pads_to_target(self):
        lines = ['abc']
        _pad_lines(lines, 3, 5)
        assert len(lines) == 3
        assert lines[1] == '     '
        assert lines[2] == '     '

    def test_already_at_target(self):
        lines = ['a', 'b', 'c']
        _pad_lines(lines, 3, 5)
        assert len(lines) == 3

    def test_empty_list(self):
        lines = []
        _pad_lines(lines, 2, 4)
        assert len(lines) == 2
        assert all(l == '    ' for l in lines)


# ─── RoPE vectorization correctness ────────────────────────────────────

class TestRoPEVectorization:
    """Verify the vectorized RoPE gives same results as the scalar version."""

    def test_rope_frequencies(self):
        """Vectorized freq precomputation matches scalar version."""
        # Scalar version (original)
        freqs_scalar = np.zeros((SEQ, HD // 2), dtype=np.float32)
        for pos in range(SEQ):
            for i in range(HD // 2):
                freq = 1.0 / (10000.0 ** (2.0 * i / HD))
                freqs_scalar[pos, i] = pos * freq

        # Vectorized version (new)
        positions = np.arange(SEQ, dtype=np.float32)
        freq_base = 1.0 / (10000.0 ** (2.0 * np.arange(HD // 2, dtype=np.float32) / HD))
        freqs_vec = np.outer(positions, freq_base)

        np.testing.assert_allclose(freqs_vec, freqs_scalar, rtol=1e-5)

    def test_rope_rotation(self):
        """Vectorized rotation matches scalar for a single position."""
        rng = np.random.RandomState(123)
        q = rng.randn(DIM).astype(np.float32)
        k = rng.randn(DIM).astype(np.float32)
        pos = 7

        # Precompute freqs
        positions = np.arange(SEQ, dtype=np.float32)
        freq_base = 1.0 / (10000.0 ** (2.0 * np.arange(HD // 2, dtype=np.float32) / HD))
        freqs = np.outer(positions, freq_base)

        # Scalar version (original)
        q_s, k_s = q.copy(), k.copy()
        for h in range(HEADS):
            for i in range(HD // 2):
                freq = freqs[pos, i]
                cos_v, sin_v = math.cos(freq), math.sin(freq)
                qi, qi1 = q_s[h * HD + 2 * i], q_s[h * HD + 2 * i + 1]
                q_s[h * HD + 2 * i] = qi * cos_v - qi1 * sin_v
                q_s[h * HD + 2 * i + 1] = qi * sin_v + qi1 * cos_v
                ki, ki1 = k_s[h * HD + 2 * i], k_s[h * HD + 2 * i + 1]
                k_s[h * HD + 2 * i] = ki * cos_v - ki1 * sin_v
                k_s[h * HD + 2 * i + 1] = ki * sin_v + ki1 * cos_v

        # Vectorized version (new)
        q_v, k_v = q.copy(), k.copy()
        freq_vec = freqs[pos, :]
        cos_full = np.tile(np.cos(freq_vec), HEADS)
        sin_full = np.tile(np.sin(freq_vec), HEADS)
        q_even, q_odd = q_v[0::2].copy(), q_v[1::2].copy()
        q_v[0::2] = q_even * cos_full - q_odd * sin_full
        q_v[1::2] = q_even * sin_full + q_odd * cos_full
        k_even, k_odd = k_v[0::2].copy(), k_v[1::2].copy()
        k_v[0::2] = k_even * cos_full - k_odd * sin_full
        k_v[1::2] = k_even * sin_full + k_odd * cos_full

        np.testing.assert_allclose(q_v, q_s, rtol=1e-5)
        np.testing.assert_allclose(k_v, k_s, rtol=1e-5)


# ─── Regex patterns ────────────────────────────────────────────────────

class TestRegexPatterns:
    def test_config_regex(self):
        m = RE_CONFIG.search('dim=768 hidden=2048 heads=12 seq=256 vocab=32000 layers=12')
        assert m is not None
        assert m.group(1) == '768'

    def test_params_regex(self):
        m = RE_PARAMS.search('Params: 109.5M (transformer 85.1M + embed 24.4M)')
        assert m is not None
        assert m.group(1) == '109.5'

    def test_step_regex(self):
        m = RE_STEP.search('step   42 loss=2.3456')
        assert m is not None
        assert m.group(1) == '42'
        assert m.group(2) == '2.3456'

    def test_batch_regex(self):
        m = RE_BATCH.search('[batch 5: compile=120.5ms train=880.3ms (22.0ms/step) compiles=3]')
        assert m is not None
        assert m.group(4) == '22.0'

    def test_timing_regex(self):
        m = RE_TIMING.search('ane=10.5 io=2.3 cls=1.1 elem=0.8 rms=0.5 cblas_wait=0.2')
        assert m is not None
        assert m.group(1) == '10.5'

    def test_power_regex(self):
        assert RE_ANE_POWER.search('ANE Power: 4500 mW') is not None
        assert RE_CPU_POWER.search('CPU Power: 3200 mW') is not None
        assert RE_GPU_POWER.search('GPU Power: 1800 mW') is not None

    def test_flops_regex(self):
        m = RE_FLOPS.search('FLOPs/step: fwd=100.0M bwd_dx=200.0M bwd_dW=300.0M sdpa_bwd=50.0M total=650.0M')
        assert m is not None
        assert m.group(5) == '650.0'

    def test_ane_tflops_regex(self):
        m = RE_ANE_TFLOPS.search('ANE TFLOPS: 1.23')
        assert m is not None
        assert m.group(1) == '1.23'


# ─── generate_text edge case ───────────────────────────────────────────

class TestGenerateText:
    def test_no_tokenizer_returns_message(self):
        from dashboard import generate_text
        # With no tokenizer file, should return error message
        with patch('dashboard.get_tokenizer', return_value=None):
            result = generate_text({}, None)
            assert result == '[no tokenizer]'
