// Checks that encode() follows the model's block-diagonal windowed attention.
//
// The HF reference splits the encoder sequence into windows of window_aftercnn tokens
// via cu_seqlens (tests/generate_reference.py), so a window's output depends only on
// its own mel frames. encode() now runs the transformer one window at a time; this
// test checks the properties that follow, which streaming relies on to cache windows:
//
//   1. prefix   : encoding only the first k windows gives the same features as the
//                 first k windows of the full encode
//   2. locality : changing the mel inside one window changes that window's features
//                 and no other window's (full-sequence attention would break this,
//                 since every token would attend across window boundaries)
//   3. streaming: encode_window_from_mel() per window, concatenated, equals encode()
//
// The mel is synthetic and deterministic: these are properties of the architecture,
// not of the audio, so no reference files are needed -- only the model.

#include "audio_encoder.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static float max_abs_diff(const float * a, const float * b, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::abs(a[i] - b[i]));
    }
    return m;
}

// Log-mel-like values in the range the normaliser produces, (x + 4) / 4.
static std::vector<float> make_mel(int n_mel, int n_frames, uint32_t seed) {
    std::vector<float> mel((size_t)n_mel * n_frames);
    uint32_t s = seed;
    for (int m = 0; m < n_mel; ++m) {
        for (int f = 0; f < n_frames; ++f) {
            s = s * 1664525u + 1013904223u;
            const float noise = (float)(s >> 8) / (float)(1u << 24);
            mel[(size_t)m * n_frames + f] = 0.4f + 0.5f * std::sin(0.013f * f + 0.21f * m) + 0.3f * (noise - 0.5f);
        }
    }
    return mel;
}

// First n_frames frames of a mel-major buffer with stride n_frames_full.
static std::vector<float> mel_prefix(const std::vector<float> & mel, int n_mel, int n_frames_full, int n_frames) {
    std::vector<float> out((size_t)n_mel * n_frames);
    for (int m = 0; m < n_mel; ++m) {
        std::memcpy(out.data() + (size_t)m * n_frames, mel.data() + (size_t)m * n_frames_full, n_frames * sizeof(float));
    }
    return out;
}

static int n_failed = 0;

static void check(const char * name, float diff, float tolerance) {
    const bool ok = diff <= tolerance;
    printf("  %-44s max diff %.3e  %s\n", name, diff, ok ? "ok" : "FAIL");
    if (!ok) n_failed++;
}

int main(int argc, char ** argv) {
    std::string model_path = "models/qwen3-asr-0.6b-f16.gguf";
    float tolerance = 1e-5f;
    int n_threads = 4;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--tolerance") == 0 && i + 1 < argc) {
            tolerance = (float)std::atof(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [--model <path>] [--tolerance <val>] [-t <n>]\n", argv[0]);
            return 0;
        }
    }

    qwen3_asr::AudioEncoder encoder;
    if (!encoder.load_model(model_path)) {
        fprintf(stderr, "Failed to load model: %s\n", encoder.get_error().c_str());
        return 1;
    }
    encoder.set_n_threads(n_threads);

    const int n_mel      = encoder.get_hparams().n_mel_bins;
    const int win_frames = encoder.window_mel_frames();   // 800 mel frames = 8 s
    const int win_tok    = encoder.window_tokens();       // 104 encoder tokens

    // Two full windows plus a half-window tail.
    const int n_frames = 2 * win_frames + win_frames / 2;
    const std::vector<float> mel = make_mel(n_mel, n_frames, 12345u);

    printf("window: %d mel frames -> %d tokens; input: %d frames\n", win_frames, win_tok, n_frames);

    std::vector<float> full;
    if (!encoder.encode(mel.data(), n_mel, n_frames, full)) {
        fprintf(stderr, "encode failed: %s\n", encoder.get_error().c_str());
        return 1;
    }

    // Feature width from the output itself, so the test does not assume a projection size.
    std::vector<float> one_window;
    {
        const std::vector<float> w0 = mel_prefix(mel, n_mel, n_frames, win_frames);
        if (!encoder.encode(w0.data(), n_mel, win_frames, one_window)) {
            fprintf(stderr, "encode failed: %s\n", encoder.get_error().c_str());
            return 1;
        }
    }
    const size_t dim     = one_window.size() / win_tok;
    const size_t per_win = (size_t)win_tok * dim;
    const int    n_tok   = (int)(full.size() / dim);
    printf("output: %d tokens x %zu\n\n", n_tok, dim);

    if (one_window.size() % win_tok != 0 || n_tok <= 2 * win_tok) {
        fprintf(stderr, "unexpected output size: %zu values for %d frames\n", full.size(), n_frames);
        return 1;
    }

    // 1. prefix
    printf("1. prefix: later audio does not change completed windows\n");
    check("encode(1 window) vs full[window 0]", max_abs_diff(one_window.data(), full.data(), per_win), tolerance);
    {
        const std::vector<float> w01 = mel_prefix(mel, n_mel, n_frames, 2 * win_frames);
        std::vector<float> two_windows;
        if (!encoder.encode(w01.data(), n_mel, 2 * win_frames, two_windows) || two_windows.size() != 2 * per_win) {
            fprintf(stderr, "encode failed: %s\n", encoder.get_error().c_str());
            return 1;
        }
        check("encode(2 windows) vs full[windows 0-1]", max_abs_diff(two_windows.data(), full.data(), 2 * per_win), tolerance);
    }

    // 2. locality
    printf("2. locality: a window's features depend only on its own frames\n");
    {
        std::vector<float> mel2 = mel;
        const std::vector<float> other = make_mel(n_mel, n_frames, 999u);
        for (int m = 0; m < n_mel; ++m) {
            for (int f = win_frames; f < 2 * win_frames; ++f) {
                mel2[(size_t)m * n_frames + f] = other[(size_t)m * n_frames + f];
            }
        }
        std::vector<float> out2;
        if (!encoder.encode(mel2.data(), n_mel, n_frames, out2) || out2.size() != full.size()) {
            fprintf(stderr, "encode failed: %s\n", encoder.get_error().c_str());
            return 1;
        }
        check("window 0 unchanged", max_abs_diff(out2.data(), full.data(), per_win), tolerance);
        check("window 2 (tail) unchanged",
              max_abs_diff(out2.data() + 2 * per_win, full.data() + 2 * per_win, full.size() - 2 * per_win), tolerance);

        // Sensitivity: the edited window must actually move, or the checks above prove nothing.
        const float moved = max_abs_diff(out2.data() + per_win, full.data() + per_win, per_win);
        const bool ok = moved > 1e-3f;
        printf("  %-44s max diff %.3e  %s\n", "window 1 (edited) changed", moved, ok ? "ok" : "FAIL");
        if (!ok) n_failed++;
    }

    // 3. streaming entry point
    printf("3. streaming: per-window encode_window_from_mel() == encode()\n");
    {
        std::vector<float> stitched;
        for (int start = 0; start < n_frames; start += win_frames) {
            std::vector<float> w;
            if (!encoder.encode_window_from_mel(mel.data(), n_mel, n_frames, start,
                                                std::min(win_frames, n_frames - start), w)) {
                fprintf(stderr, "encode_window_from_mel failed: %s\n", encoder.get_error().c_str());
                return 1;
            }
            stitched.insert(stitched.end(), w.begin(), w.end());
        }
        if (stitched.size() != full.size()) {
            printf("  size mismatch: %zu vs %zu  FAIL\n", stitched.size(), full.size());
            n_failed++;
        } else {
            check("stitched windows vs encode()", max_abs_diff(stitched.data(), full.data(), full.size()), tolerance);
        }
    }

    if (n_failed == 0) {
        printf("\nTEST PASSED\n");
        return 0;
    }
    printf("\nTEST FAILED: %d check(s)\n", n_failed);
    return 1;
}
