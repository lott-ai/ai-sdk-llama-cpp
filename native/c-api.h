#ifndef LLAMA_C_API_H
#define LLAMA_C_API_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _WIN32
#define LLAMA_API __declspec(dllexport)
#else
#define LLAMA_API __attribute__((visibility("default")))
#endif

// ============================================================================
// Types
// ============================================================================

typedef int32_t llama_handle_t;

typedef struct {
    const char* model_path;
    int32_t gpu_layers;
    int32_t context_size;
    int32_t threads;
    bool debug;
    const char* chat_template;
} llama_load_options_t;

typedef struct {
    const char* role;
    const char* content;
} llama_chat_message_t;

typedef struct {
    llama_chat_message_t* messages;
    size_t message_count;
    int32_t max_tokens;
    float temperature;
    float top_p;
    int32_t top_k;
    const char** stop_sequences;
    size_t stop_sequence_count;
} llama_generate_options_t;

typedef struct {
    char* text;
    int32_t prompt_tokens;
    int32_t completion_tokens;
    char* finish_reason;
    char* error;
} llama_generate_result_t;

// Token callback for streaming: return 0 to continue, non-zero to stop
typedef int (*llama_token_callback_t)(const char* token, void* user_data);

// ============================================================================
// Functions
// ============================================================================

/**
 * Load a model from a GGUF file.
 * Returns a handle > 0 on success, -1 on failure.
 * Check llama_get_last_error() for error details.
 */
LLAMA_API llama_handle_t llama_load_model(const llama_load_options_t* options);

/**
 * Unload a model.
 * Returns true on success.
 */
LLAMA_API bool llama_unload_model(llama_handle_t handle);

/**
 * Check if a model is loaded.
 */
LLAMA_API bool llama_is_model_loaded(llama_handle_t handle);

/**
 * Generate text from messages (non-streaming).
 * Caller must call llama_free_result() on the returned result.
 */
LLAMA_API llama_generate_result_t* llama_generate(
    llama_handle_t handle,
    const llama_generate_options_t* options
);

/**
 * Generate text from messages (streaming).
 * Calls the callback for each token.
 * Caller must call llama_free_result() on the returned result.
 */
LLAMA_API llama_generate_result_t* llama_generate_stream(
    llama_handle_t handle,
    const llama_generate_options_t* options,
    llama_token_callback_t callback,
    void* user_data
);

/**
 * Free a result structure.
 */
LLAMA_API void llama_free_result(llama_generate_result_t* result);

/**
 * Get the last error message.
 * Returns NULL if no error.
 */
LLAMA_API const char* llama_get_last_error(void);

/**
 * Clear the last error.
 */
LLAMA_API void llama_clear_error(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_C_API_H

