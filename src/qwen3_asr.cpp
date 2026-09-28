#include "qwen3_asr.h"
#include "timing.h"

#include <chrono>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <fstream>
#include <ggml-impl.h>

namespace qwen3_asr {

static int64_t get_time_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Index of the first <|audio_pad|> token in the chat template built by
// build_input_tokens(): <|im_start|>system \n <|im_end|> \n <|im_start|> user \n
// <|audio_start|> is 9 tokens, so the audio embeddings are injected from there.
static constexpr int32_t kAudioPadStartPos = 9;

// The model emits a two-token language marker before any transcript text.
static constexpr size_t kLanguagePrefixTokens = 2;

Qwen3ASR::Qwen3ASR() = default;
Qwen3ASR::~Qwen3ASR() = default;

bool Qwen3ASR::load_model(const std::string & model_path) {
    int64_t t_start = get_time_ms();
    
    if (!encoder_.load_model(model_path)) {
        error_msg_ = "Failed to load audio encoder: " + encoder_.get_error();
        return false;
    }

    if (!decoder_.load_model(model_path)) {
        error_msg_ = "Failed to load text decoder: " + decoder_.get_error();
        return false;
    }

    generate_mel_filters(mel_filters_, QWEN_N_MELS, QWEN_N_FFT, QWEN_SAMPLE_RATE);

    model_loaded_ = true;
    int64_t t_end = get_time_ms();

    GGML_LOG_INFO("%s: Qwen3ASR loaded in %lld ms\n", __func__, (long long)(t_end - t_start));
    return true;
}

transcribe_result Qwen3ASR::transcribe(const std::string & audio_path,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    std::vector<float> samples;
    int sample_rate;
    
    if (!load_wav(audio_path, samples, sample_rate)) {
        result.error_msg = "Failed to load audio file: " + audio_path;
        return result;
    }
    
    if (sample_rate != QWEN_SAMPLE_RATE) {
        result.error_msg = "Audio must be 16kHz, got " + std::to_string(sample_rate) + " Hz";
        return result;
    }
    
    return transcribe_internal(samples.data(), samples.size(), params);
}

transcribe_result Qwen3ASR::transcribe(const float * samples, int n_samples,
                                        const transcribe_params & params) {
    transcribe_result result;
    
    if (!model_loaded_) {
        result.error_msg = "Model not loaded";
        return result;
    }
    
    return transcribe_internal(samples, n_samples, params);
}

transcribe_result Qwen3ASR::transcribe_internal(const float * samples, int n_samples,
                                                 const transcribe_params & params) {
    transcribe_result result;
    int64_t t_total_start = get_time_ms();

    encoder_.set_n_threads(params.n_threads);
    decoder_.set_n_threads(params.n_threads);
    
    int64_t t_mel_start = get_time_ms();
    MelSpectrogram mel;
    {
        QWEN3_TIMER("mel_spectrogram");
        if (!log_mel_spectrogram(samples, n_samples, mel_filters_, mel, params.n_threads)) {
            result.error_msg = "Failed to compute mel spectrogram";
            return result;
        }
    }
    result.t_mel_ms = get_time_ms() - t_mel_start;
    
    if (params.print_progress) {
        GGML_LOG_INFO("%s: Mel spectrogram: [%d, %d]\n", __func__, mel.n_mel, mel.n_len);
    }
    
    int64_t t_encode_start = get_time_ms();
    std::vector<float> audio_features;
    {
        QWEN3_TIMER("audio_encoding");
        if (!encoder_.encode(mel.data.data(), mel.n_mel, mel.n_len, audio_features)) {
            result.error_msg = "Failed to encode audio: " + encoder_.get_error();
            return result;
        }
    }
    result.t_encode_ms = get_time_ms() - t_encode_start;
    
    const auto & text_hparams = encoder_.get_text_hparams();
    int32_t n_audio_frames = audio_features.size() / text_hparams.hidden_size;
    
    if (params.print_progress) {
        GGML_LOG_INFO("%s: Audio features: [%d, %d]\n", __func__, n_audio_frames, text_hparams.hidden_size);
    }
    
    std::vector<int32_t> input_tokens = build_input_tokens(n_audio_frames, params.language);
    
    if (params.print_progress) {
        GGML_LOG_INFO("%s: Input tokens: %zu\n", __func__, input_tokens.size());
    }
    
    int64_t t_decode_start = get_time_ms();
    std::vector<int32_t> output_tokens;
    if (!decode_greedy(input_tokens, audio_features, n_audio_frames, params, output_tokens)) {
        result.error_msg = "Decoding failed: " + error_msg_;
        return result;
    }
    result.t_decode_ms = get_time_ms() - t_decode_start;
    
    result.tokens = output_tokens;
    const size_t text_start = std::min<size_t>(output_tokens.size(), 2);  // remove language prefix tokens
    std::vector text_tokens(output_tokens.begin() + text_start, output_tokens.end());
    result.text = decoder_.decode_tokens(text_tokens);
    if (output_tokens.size() >= 2) {
        result.language = decoder_.decode_token(output_tokens[1]);
    }
    result.success = true;
    
    result.t_total_ms = get_time_ms() - t_total_start;
    
    if (params.print_timing) {
        GGML_LOG_INFO("%s: [Timing]\n", __func__);
        GGML_LOG_INFO("    Mel spectrogram: %lld ms\n", static_cast<long long>(result.t_mel_ms));
        GGML_LOG_INFO("    Audio encoding:  %lld ms\n", static_cast<long long>(result.t_encode_ms));
        GGML_LOG_INFO("    Text decoding:   %lld ms\n", static_cast<long long>(result.t_decode_ms));
        GGML_LOG_INFO("    Total:           %lld ms\n", static_cast<long long>(result.t_total_ms));
        GGML_LOG_INFO("    Tokens generated: %zu\n", output_tokens.size());
    }
    
    return result;
}

std::vector<int32_t> Qwen3ASR::build_input_tokens(int32_t n_audio_frames,
                                                   const std::string & language) {
    const auto & cfg = decoder_.get_config();
    
    std::vector<int32_t> tokens;
    tokens.reserve(n_audio_frames + 20);
    
    // Chat template format:
    // <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|><|audio_pad|>...<|audio_end|><|im_end|>\n<|im_start|>assistant\n
    
    // Token IDs from Qwen3 tokenizer:
    // <|im_start|> = 151644
    // <|im_end|> = 151645
    // system = 8948
    // user = 872
    // assistant = 77091
    // \n = 198
    
    const int32_t im_start = 151644;
    const int32_t im_end = 151645;
    const int32_t system_token = 8948;
    const int32_t user_token = 872;
    const int32_t assistant_token = 77091;
    const int32_t newline = 198;
    
    // <|im_start|>system\n<|im_end|>\n
    tokens.push_back(im_start);
    tokens.push_back(system_token);
    tokens.push_back(newline);
    tokens.push_back(im_end);
    tokens.push_back(newline);
    
    // <|im_start|>user\n
    tokens.push_back(im_start);
    tokens.push_back(user_token);
    tokens.push_back(newline);
    
    // <|audio_start|><|audio_pad|>...<|audio_end|>
    tokens.push_back(cfg.audio_start_token_id);
    for (int32_t i = 0; i < n_audio_frames; ++i) {
        tokens.push_back(cfg.audio_pad_token_id);
    }
    tokens.push_back(cfg.audio_end_token_id);
    
    // <|im_end|>\n<|im_start|>assistant\n
    tokens.push_back(im_end);
    tokens.push_back(newline);
    tokens.push_back(im_start);
    tokens.push_back(assistant_token);
    tokens.push_back(newline);
    
    (void)language;
    
    return tokens;
}

bool Qwen3ASR::decode_greedy(const std::vector<int32_t> & input_tokens,
                              const std::vector<float> & audio_features,
                              int32_t n_audio_frames,
                              const transcribe_params & params,
                              std::vector<int32_t> & output_tokens) {
    const auto & cfg = decoder_.get_config();
    
    int32_t n_ctx_needed = input_tokens.size() + params.max_tokens;
    if (!decoder_.init_kv_cache(n_ctx_needed)) {
        error_msg_ = "Failed to initialize KV cache: " + decoder_.get_error();
        return false;
    }
    
    std::vector<float> logits;
    
    // Audio pad tokens start after: <|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>
    // That's 8 tokens before the first audio_pad
    int32_t audio_start_pos = 9;
    
    {
        QWEN3_TIMER("decode.initial_forward");
        if (!decoder_.forward_with_audio(
                input_tokens.data(), input_tokens.size(),
                audio_features.data(), n_audio_frames,
                audio_start_pos, 0, logits)) {
            error_msg_ = "Initial forward pass failed: " + decoder_.get_error();
            return false;
        }
    }
    
    int32_t vocab_size = cfg.vocab_size;
    int32_t n_input = input_tokens.size();
    
    int32_t next_token = sample_greedy(logits.data(), vocab_size);
    
    output_tokens.clear();
    output_tokens.push_back(next_token);
    
    if (progress_callback_) {
        progress_callback_(1, params.max_tokens);
    }
    
    int32_t n_past = n_input;
    
    while (next_token != cfg.eos_token_id && 
           (int32_t)output_tokens.size() < params.max_tokens) {
        
        std::vector single_token = {next_token};
        
        {
            QWEN3_TIMER("decode.token");
            if (!decoder_.forward(single_token.data(), 1, n_past, logits)) {
                error_msg_ = "Forward pass failed at token " + 
                             std::to_string(output_tokens.size()) + ": " + decoder_.get_error();
                return false;
            }
        }
        
        next_token = sample_greedy(logits.data(), vocab_size);
        output_tokens.push_back(next_token);
        
        n_past += 1;
        
        if (progress_callback_) {
            progress_callback_(output_tokens.size(), params.max_tokens);
        }
        
        if (params.print_progress && output_tokens.size() % 10 == 0) {
            GGML_LOG_INFO("Generated %zu tokens...\n", output_tokens.size());
        }
    }
    
    if (output_tokens.back() == cfg.eos_token_id) {
        output_tokens.pop_back();
    }
    
    return true;
}

int32_t Qwen3ASR::sample_greedy(const float * logits, int32_t vocab_size) {
    int32_t max_idx = 0;
    float max_val = logits[0];
    
    for (int32_t i = 1; i < vocab_size; ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }
    
    return max_idx;
}

void Qwen3ASR::set_progress_callback(progress_callback_t callback) {
    progress_callback_ = std::move(callback);
}

// ===========================================================================
// Streaming
//
// One step = one call to streaming_step(), which does four things in order:
//   1. mel      -- recompute the spectrogram over the retained audio
//   2. encode   -- reuse cached windows, re-encode only the partial tail window
//   3. decode   -- rebuild the prompt and greedily generate
//   4. commit   -- splice the new tokens onto the settled ones, then evict
//
// Each is a helper below; streaming_step() is only the sequencing.
// ===========================================================================

namespace {

// What the decoder gets to see this step, and how much of the previous output
// survives it.
struct decoder_context {
    std::vector<int32_t> prompt_prefix;   // text tokens appended to the prompt
    int32_t              n_settled = 0;   // prev_tokens[0, n_settled) are kept as-is
};

// The last `rollback` tokens of the previous output are treated as unstable and are
// re-decoded this step; everything before them is settled.
//
// Only the last `max_prefix_tokens` settled tokens are shown to the model. The rest
// stay in prev_tokens, so bounding the decoder's context never truncates the
// transcript -- it only limits how far back the model can see.
decoder_context plan_decoder_context(const qwen3_asr::streaming_state & st, int32_t rollback) {
    decoder_context ctx;

    const int32_t n_prev = (int32_t) st.prev_tokens.size();
    if (n_prev == 0) {
        return ctx;
    }

    // Cold start deliberately discards the previous output and re-transcribes the whole
    // buffer, which sharpens the first couple of chunks. That is only sound while every
    // sample is still buffered: once a window has been evicted its audio is gone for
    // good, so from then on the transcript must always be carried forward.
    const bool all_audio_retained = st.n_windows_dropped == 0;
    const bool cold_start         = st.n_chunks_fed < st.params.unfixed_chunk_num;
    if (cold_start && all_audio_retained) {
        return ctx;
    }

    ctx.n_settled = std::max(0, n_prev - rollback);

    const int32_t prefix_begin = std::max(0, ctx.n_settled - st.params.max_prefix_tokens);
    ctx.prompt_prefix.assign(st.prev_tokens.begin() + prefix_begin,
                             st.prev_tokens.begin() + ctx.n_settled);
    return ctx;
}

}  // namespace

void Qwen3ASR::init_streaming(streaming_state & st, const streaming_params & params) {
    st = streaming_state();
    st.params = params;
}

bool Qwen3ASR::feed_audio(streaming_state & st, const float * samples, int n_samples) {
    if (!model_loaded_) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (st.finished) {
        error_msg_ = "Streaming session already finished";
        return false;
    }
    if (samples != nullptr && n_samples > 0) {
        st.audio.insert(st.audio.end(), samples, samples + n_samples);
    }
    return streaming_step(st);
}

bool Qwen3ASR::finish_streaming(streaming_state & st) {
    if (!model_loaded_) {
        error_msg_ = "Model not loaded";
        return false;
    }
    if (st.finished) {
        return true;
    }

    // A normal step, deliberately: it rolls the unstable tail back and re-decodes it
    // with the complete audio, which is the last chance to revise those tokens. Running
    // the final pass with rollback disabled would freeze them and let it only append.
    if (!streaming_step(st)) {
        return false;
    }
    st.finished = true;
    return true;
}

// Encoder features for the whole retained buffer: cached windows verbatim, plus the
// partial tail window re-encoded from scratch.
//
// A completed window is final under the model's block-diagonal windowed attention, so
// it is encoded once and never revisited. Only the tail -- at most one window -- is
// recomputed per step.
bool Qwen3ASR::stream_encode_features(streaming_state & st, const MelSpectrogram & mel,
                                      std::vector<float> & features) {
    const int mel_frames_per_window = encoder_.window_mel_frames();
    const int n_complete_windows    = mel.n_len / mel_frames_per_window;
    const int first_uncached_window = st.n_windows_cached - st.n_windows_dropped;

    {
        QWEN3_TIMER("streaming.encode_new_windows");
        for (int w = first_uncached_window; w < n_complete_windows; ++w) {
            std::vector<float> window_features;
            if (!encoder_.encode_window_from_mel(mel.data.data(), mel.n_mel, mel.n_len,
                                                 w * mel_frames_per_window,
                                                 mel_frames_per_window, window_features)) {
                error_msg_ = "Failed to encode window: " + encoder_.get_error();
                return false;
            }
            st.win_cache.insert(st.win_cache.end(), window_features.begin(), window_features.end());
            st.n_windows_cached++;
        }
    }

    features = st.win_cache;

    const int tail_begin  = n_complete_windows * mel_frames_per_window;
    const int tail_frames = mel.n_len - tail_begin;
    if (tail_frames > 0) {
        QWEN3_TIMER("streaming.encode_tail");
        std::vector<float> tail_features;
        if (!encoder_.encode_window_from_mel(mel.data.data(), mel.n_mel, mel.n_len,
                                             tail_begin, tail_frames, tail_features)) {
            error_msg_ = "Failed to encode tail window: " + encoder_.get_error();
            return false;
        }
        features.insert(features.end(), tail_features.begin(), tail_features.end());
    }

    return true;
}

// Greedy decode of one step. Returns only the NEWLY generated tokens; splicing them
// onto the settled ones is the caller's job.
//
// The audio_pad count grows every step, so every later token shifts position and the KV
// cache cannot carry over. Rebuilding it per step is forced by the model carrying audio
// in the prompt rather than in cross-attention.
bool Qwen3ASR::stream_generate(streaming_state & st,
                               const std::vector<float> & features, int32_t n_audio_frames,
                               const std::vector<int32_t> & prompt_prefix,
                               std::vector<int32_t> & generated) {
    const auto & cfg = decoder_.get_config();

    std::vector<int32_t> prompt = build_input_tokens(n_audio_frames, st.params.language);
    prompt.insert(prompt.end(), prompt_prefix.begin(), prompt_prefix.end());

    if (!decoder_.init_kv_cache((int32_t) prompt.size() + st.params.max_new_tokens)) {
        error_msg_ = "Failed to initialize KV cache: " + decoder_.get_error();
        return false;
    }

    std::vector<float> logits;
    {
        QWEN3_TIMER("streaming.decode_prefill");
        if (!decoder_.forward_with_audio(prompt.data(), (int32_t) prompt.size(),
                                         features.data(), n_audio_frames,
                                         /*audio_start_pos=*/kAudioPadStartPos,
                                         /*n_past=*/0, logits)) {
            error_msg_ = "Streaming prefill failed: " + decoder_.get_error();
            return false;
        }
    }

    generated.clear();
    int32_t n_past = (int32_t) prompt.size();
    int32_t next   = sample_greedy(logits.data(), cfg.vocab_size);

    {
        QWEN3_TIMER("streaming.decode_loop");
        while (next != cfg.eos_token_id && (int32_t) generated.size() < st.params.max_new_tokens) {
            generated.push_back(next);
            if (!decoder_.forward(&next, 1, n_past, logits)) {
                error_msg_ = "Streaming decode failed: " + decoder_.get_error();
                return false;
            }
            n_past += 1;
            next = sample_greedy(logits.data(), cfg.vocab_size);
        }
    }

    return true;
}

// Drop the oldest cached windows, and the audio behind them, until the encoder context
// is back within max_cached_windows. Features and samples are dropped together so
// n_windows_dropped stays a valid index into both.
void Qwen3ASR::stream_evict_windows(streaming_state & st, int32_t hidden_size) {
    const size_t features_per_window = (size_t) encoder_.window_tokens() * (size_t) hidden_size;
    const int    samples_per_window  = encoder_.window_mel_frames() * QWEN_HOP_LENGTH;

    while (st.n_windows_cached - st.n_windows_dropped > st.params.max_cached_windows) {
        if (st.win_cache.size() < features_per_window || (int) st.audio.size() <= samples_per_window) {
            break;
        }
        st.win_cache.erase(st.win_cache.begin(), st.win_cache.begin() + features_per_window);
        st.audio.erase(st.audio.begin(), st.audio.begin() + samples_per_window);
        st.n_windows_dropped++;
    }
}

// Rebuild st.text from the token history. prev_tokens[0] and [1] are the language
// marker the model emits before any transcript.
void Qwen3ASR::stream_publish_text(streaming_state & st) {
    const size_t text_begin = std::min<size_t>(st.prev_tokens.size(), kLanguagePrefixTokens);
    st.text = decoder_.decode_tokens(
        std::vector<int32_t>(st.prev_tokens.begin() + text_begin, st.prev_tokens.end()));

    if (st.prev_tokens.size() >= kLanguagePrefixTokens) {
        st.language = decoder_.decode_token(st.prev_tokens[1]);
    }
}

bool Qwen3ASR::streaming_step(streaming_state & st) {
    if (st.audio.empty()) {
        return true;
    }

    encoder_.set_n_threads(st.params.n_threads);
    decoder_.set_n_threads(st.params.n_threads);

    // --- 1. mel ---------------------------------------------------------------
    // Recomputed in full every step. Before normalisation a frame only depends on its own
    // 400-sample window, but the normalisation clamps against the GLOBAL max over the
    // retained buffer (Whisper's log_mel_spectrogram), so one loud new frame -- or evicting
    // the loudest window -- shifts every frame's value. Cached windows keep the features
    // they were encoded with, so streaming is close to, but not bit-identical with, offline.
    int64_t t0 = get_time_ms();
    MelSpectrogram mel;
    {
        QWEN3_TIMER("streaming.mel");
        if (!log_mel_spectrogram(st.audio.data(), (int) st.audio.size(), mel_filters_, mel,
                                 st.params.n_threads)) {
            error_msg_ = "Failed to compute mel spectrogram";
            return false;
        }
    }
    st.t_mel_ms = get_time_ms() - t0;

    // --- 2. encode ------------------------------------------------------------
    t0 = get_time_ms();
    std::vector<float> features;
    if (!stream_encode_features(st, mel, features)) {
        return false;
    }
    st.t_encode_ms = get_time_ms() - t0;

    const int32_t hidden_size    = encoder_.get_text_hparams().hidden_size;
    const int32_t n_audio_frames = (int32_t) (features.size() / (size_t) hidden_size);
    if (n_audio_frames <= 0) {
        return true;
    }

    // --- 3. decode ------------------------------------------------------------
    t0 = get_time_ms();
    const decoder_context ctx = plan_decoder_context(st, st.params.unfixed_token_num);

    std::vector<int32_t> generated;
    if (!stream_generate(st, features, n_audio_frames, ctx.prompt_prefix, generated)) {
        return false;
    }
    st.t_decode_ms = get_time_ms() - t0;

    // --- 4. commit ------------------------------------------------------------
    // An empty generation means the decoder hit EOS straight away -- it heard nothing
    // new. Rolling the unstable tail back and appending nothing would silently delete
    // settled text, so in that case leave the transcript exactly as it was.
    if (!generated.empty() || st.prev_tokens.empty()) {
        std::vector<int32_t> updated_tokens(st.prev_tokens.begin(),
                                            st.prev_tokens.begin() + ctx.n_settled);
        updated_tokens.insert(updated_tokens.end(), generated.begin(), generated.end());
        st.prev_tokens = std::move(updated_tokens);
        stream_publish_text(st);
    }

    st.n_chunks_fed++;
    stream_evict_windows(st, hidden_size);

    return true;
}

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate) {
    return load_wav(path, samples, sample_rate);
}

} // namespace qwen3_asr
