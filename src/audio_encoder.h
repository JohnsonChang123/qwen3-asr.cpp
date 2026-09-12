#pragma once

#include "qwen3asr_win_export.h"
#include "gguf_loader.h"

#include <vector>

namespace qwen3_asr {

struct audio_encoder_state {
    ggml_backend_t backend_cpu = nullptr;
    ggml_backend_t backend_gpu = nullptr;
    ggml_backend_sched_t sched = nullptr;
    
    std::vector<uint8_t> compute_meta;
    
    ggml_tensor * embd_conv = nullptr;
    ggml_tensor * embd_enc = nullptr;
};

class AudioEncoder {
public:
    AudioEncoder();
    ~AudioEncoder();
    
    bool load_model(const std::string & model_path);
    void set_n_threads(int n_threads);
    
    bool encode(const float * mel_data, int n_mel, int n_frames, 
                std::vector<float> & output);
    
    bool encode_conv_only(const float * mel_data, int n_mel, int n_frames,
                          std::vector<float> & output);

    // Encode without chunking (for testing/debugging)
    bool encode_no_chunk(const float * mel_data, int n_mel, int n_frames,
                         std::vector<float> & output);
    
    [[nodiscard]] const audio_encoder_hparams & get_hparams() const { return model_.hparams; }
    [[nodiscard]] const text_decoder_hparams & get_text_hparams() const { return model_.text_hparams; }
    
    [[nodiscard]] const std::string & get_error() const { return error_msg_; }
    
    // --- streaming entry points -------------------------------------------------
    // Number of encoder tokens in one attention window (window_aftercnn). Full windows
    // are always exactly this size, so streaming can cache them once and never redo them.
    [[nodiscard]] int window_tokens() const;

    // Number of mel frames that make up one attention window (n_window_infer).
    [[nodiscard]] int window_mel_frames() const { return model_.hparams.n_window_infer; }

    // Encode exactly one attention window straight from a mel frame range.
    // mel_data is the FULL mel, laid out mel-major: mel_data[m * mel_stride + frame].
    bool encode_window_from_mel(const float * mel_data, int n_mel, int mel_stride,
                                int frame_offset, int n_frames_range,
                                std::vector<float> & output);

private:
    // Conv frontend over a mel frame range, in 100-frame (1 second) chunks.
    bool conv_encode_range(const float * mel_data, int n_mel, int mel_stride,
                           int frame_offset, int n_frames_range,
                           std::vector<float> & conv_out);

    // Encode ONE attention window (n_tokens <= window_aftercnn). Windows are independent
    // under the model's block-diagonal windowed attention, so this is both equivalent to
    // masking and the unit that streaming caches.
    bool encode_transformer(const float * conv_out, int n_tokens, std::vector<float> & output);

    ggml_cgraph * build_graph_conv(int n_frames);
    ggml_cgraph * build_graph_encoder(int n_ctx);

    bool compute_graph(ggml_cgraph * graph);

    audio_encoder_model model_;
    audio_encoder_state state_;
    std::string error_msg_;
    
    int n_threads_ = 4;
};

} // namespace qwen3_asr
