#pragma once

#include "qwen3asr_win_export.h"

#include "mel_spectrogram.h"
#include "audio_encoder.h"
#include "text_decoder.h"

#include <string>
#include <vector>
#include <functional>

namespace qwen3_asr {

// Transcription parameters
struct transcribe_params {
    // Maximum number of tokens to generate
    int32_t max_tokens = 1024;
    
    // Language code (optional, for prompting)
    std::string language = "";
    
    // Number of threads for mel computation
    int32_t n_threads = 4;
    
    // Print progress during transcription
    bool print_progress = false;
    
    // Print timing information
    bool print_timing = true;
};

// Transcription result
struct transcribe_result {
    std::string text;
    std::string language;
    std::vector<int32_t> tokens;
    bool success = false;
    std::string error_msg;
    
    // Timing info (in milliseconds)
    int64_t t_load_ms = 0;
    int64_t t_mel_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_decode_ms = 0;
    int64_t t_total_ms = 0;
};

// ---------------------------------------------------------------------------
// Streaming
//
// Defaults are the official recipe from the Qwen3-ASR Technical Report
// (arXiv:2601.21337, section 4.5): 2 s chunks, 5-token fallback, and the last
// four chunks left unfixed. The encoder's block-diagonal windowed attention is
// what makes this sound: a completed window never changes, so it is encoded
// once and cached for the rest of the session.
// ---------------------------------------------------------------------------
struct streaming_params {
    // Audio added per feed_audio() call, in seconds. Informational -- feed_audio()
    // accepts any length; this is what the caller is expected to pace at.
    float chunk_size_sec = 2.0f;

    // Drop this many tokens off the tail of the previous output before reusing it
    // as the decoder prefix. Absorbs the instability at chunk boundaries.
    int32_t unfixed_token_num = 5;

    // Cold start: give no text prefix at all for this many chunks.
    int32_t unfixed_chunk_num = 2;

    // Encoder context bound: keep at most this many completed windows (4 = ~32 s).
    int32_t max_cached_windows = 4;

    // Decoder context bound: feed back at most this many tokens of previous text.
    int32_t max_prefix_tokens = 150;

    // Tokens generated per step. A 2 s chunk of Chinese is ~10 tokens, so this is
    // mostly a latency guard: on pathological input (e.g. looping audio) the decoder
    // can fail to emit EOS, and an unbounded cap turns one step into seconds.
    int32_t max_new_tokens = 48;

    int32_t n_threads = 4;
    std::string language = "";
};

struct streaming_state {
    streaming_params params;

    // Audio retained for windows [n_windows_dropped, ...). Older audio is evicted
    // along with its cached features.
    std::vector<float> audio;

    // Encoder features for completed windows [n_windows_dropped, n_windows_cached).
    std::vector<float> win_cache;

    int32_t n_windows_cached  = 0;   // absolute index one past the last cached window
    int32_t n_windows_dropped = 0;   // absolute index of the oldest retained window

    std::vector<int32_t> prev_tokens;  // previous step's full output (prefix + generated)
    int32_t n_chunks_fed = 0;

    std::string text;       // transcript so far
    std::string language;
    bool finished = false;

    // Per-step timing (ms)
    int64_t t_mel_ms = 0;
    int64_t t_encode_ms = 0;
    int64_t t_decode_ms = 0;
};

// Progress callback type
using progress_callback_t = std::function<void(int tokens_generated, int max_tokens)>;

// Main ASR class that orchestrates the full pipeline
class Qwen3ASR {
public:
    Qwen3ASR();
    ~Qwen3ASR();
    
    // Load model from GGUF file
    // Returns true on success, false on failure (check get_error())
    bool load_model(const std::string & model_path);
    
    // Transcribe audio file (WAV format, 16kHz mono)
    // Returns transcription result
    transcribe_result transcribe(const std::string & audio_path, 
                                  const transcribe_params & params = transcribe_params());
    
    // Transcribe raw audio samples
    // samples: audio samples normalized to [-1, 1]
    // n_samples: number of samples
    transcribe_result transcribe(const float * samples, int n_samples,
                                  const transcribe_params & params = transcribe_params());
    
    // --- streaming -------------------------------------------------------------
    // Reset st and arm it with params. Safe to call on a used state.
    void init_streaming(streaming_state & st, const streaming_params & params = streaming_params());

    // Append audio (16 kHz mono, [-1,1]) and re-decode. Updates st.text.
    // n_samples may be 0 to re-decode without new audio.
    bool feed_audio(streaming_state & st, const float * samples, int n_samples);

    // Final pass: decode the tail with no rollback and commit everything.
    bool finish_streaming(streaming_state & st);

    // Set progress callback
    void set_progress_callback(progress_callback_t callback);
    
    // Get error message
    const std::string & get_error() const { return error_msg_; }
    
    // Check if model is loaded
    bool is_loaded() const { return model_loaded_; }
    
    // Get model config
    const text_decoder_config & get_config() const { return decoder_.get_config(); }
    
private:
    // Internal transcription implementation
    transcribe_result transcribe_internal(const float * samples, int n_samples,
                                           const transcribe_params & params);
    
    // Build input token sequence for audio
    std::vector<int32_t> build_input_tokens(int32_t n_audio_frames, 
                                             const std::string & language);
    
    // One streaming step: mel -> encode -> decode -> commit. See src/qwen3_asr.cpp.
    bool streaming_step(streaming_state & st);

    // Encoder features for the retained buffer: cached windows plus the partial tail.
    bool stream_encode_features(streaming_state & st, const MelSpectrogram & mel,
                                std::vector<float> & features);

    // Greedy decode of one step; returns only the newly generated tokens.
    bool stream_generate(streaming_state & st,
                         const std::vector<float> & features, int32_t n_audio_frames,
                         const std::vector<int32_t> & prompt_prefix,
                         std::vector<int32_t> & generated);

    // Bound the encoder context by dropping the oldest windows and their audio.
    void stream_evict_windows(streaming_state & st, int32_t hidden_size);

    // Rebuild st.text / st.language from st.prev_tokens.
    void stream_publish_text(streaming_state & st);

    // Greedy decoding loop
    bool decode_greedy(const std::vector<int32_t> & input_tokens,
                       const std::vector<float> & audio_features,
                       int32_t n_audio_frames,
                       const transcribe_params & params,
                       std::vector<int32_t> & output_tokens);
    
    // Sample next token (greedy: argmax)
    int32_t sample_greedy(const float * logits, int32_t vocab_size);
    
    // Components
    AudioEncoder encoder_;
    TextDecoder decoder_;
    MelFilters mel_filters_;
    
    // State
    bool model_loaded_ = false;
    std::string error_msg_;
    progress_callback_t progress_callback_;
};

bool load_audio_file(const std::string & path, std::vector<float> & samples, int & sample_rate);

} // namespace qwen3_asr
