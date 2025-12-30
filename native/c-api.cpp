#include "c-api.h"
#include "llama-wrapper.h"
#include <memory>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <cstring>

// ============================================================================
// Global State
// ============================================================================

static std::unordered_map<llama_handle_t, std::unique_ptr<llama_wrapper::LlamaModel>> g_models;
static std::mutex g_models_mutex;
static std::atomic<llama_handle_t> g_next_handle{1};
static thread_local std::string g_last_error;

// ============================================================================
// Helper Functions
// ============================================================================

static void set_error(const std::string& error) {
    g_last_error = error;
}

static char* strdup_safe(const std::string& str) {
    if (str.empty()) {
        return nullptr;
    }
    char* result = static_cast<char*>(malloc(str.length() + 1));
    if (result) {
        std::memcpy(result, str.c_str(), str.length() + 1);
    }
    return result;
}

static llama_generate_result_t* create_result() {
    auto* result = static_cast<llama_generate_result_t*>(calloc(1, sizeof(llama_generate_result_t)));
    return result;
}

// ============================================================================
// C API Implementation
// ============================================================================

extern "C" {

LLAMA_API llama_handle_t llama_load_model(const llama_load_options_t* options) {
    if (!options || !options->model_path) {
        set_error("Invalid options: model_path is required");
        return -1;
    }

    auto model = std::make_unique<llama_wrapper::LlamaModel>();

    llama_wrapper::ModelParams model_params;
    model_params.model_path = options->model_path;
    model_params.n_gpu_layers = options->gpu_layers > 0 ? options->gpu_layers : 99;
    model_params.debug = options->debug;
    model_params.chat_template = options->chat_template ? options->chat_template : "auto";

    if (!model->load(model_params)) {
        set_error("Failed to load model from: " + std::string(options->model_path));
        return -1;
    }

    llama_wrapper::ContextParams ctx_params;
    ctx_params.n_ctx = options->context_size > 0 ? options->context_size : 2048;
    ctx_params.n_threads = options->threads > 0 ? options->threads : 4;

    if (!model->create_context(ctx_params)) {
        set_error("Failed to create context");
        return -1;
    }

    llama_handle_t handle = g_next_handle++;

    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        g_models[handle] = std::move(model);
    }

    return handle;
}

LLAMA_API bool llama_unload_model(llama_handle_t handle) {
    std::lock_guard<std::mutex> lock(g_models_mutex);
    auto it = g_models.find(handle);
    if (it != g_models.end()) {
        g_models.erase(it);
        return true;
    }
    return false;
}

LLAMA_API bool llama_is_model_loaded(llama_handle_t handle) {
    std::lock_guard<std::mutex> lock(g_models_mutex);
    auto it = g_models.find(handle);
    return it != g_models.end() && it->second->is_loaded();
}

LLAMA_API llama_generate_result_t* llama_generate(
    llama_handle_t handle,
    const llama_generate_options_t* options
) {
    auto* result = create_result();
    if (!result) {
        return nullptr;
    }

    llama_wrapper::LlamaModel* model = nullptr;

    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(handle);
        if (it == g_models.end()) {
            result->error = strdup_safe("Invalid model handle");
            result->finish_reason = strdup_safe("error");
            return result;
        }
        model = it->second.get();
    }

    // Convert messages
    std::vector<llama_wrapper::ChatMessage> messages;
    for (size_t i = 0; i < options->message_count; i++) {
        llama_wrapper::ChatMessage msg;
        msg.role = options->messages[i].role ? options->messages[i].role : "";
        msg.content = options->messages[i].content ? options->messages[i].content : "";
        messages.push_back(msg);
    }

    // Set up generation params
    llama_wrapper::GenerationParams params;
    params.max_tokens = options->max_tokens > 0 ? options->max_tokens : 256;
    params.temperature = options->temperature > 0 ? options->temperature : 0.7f;
    params.top_p = options->top_p > 0 ? options->top_p : 0.9f;
    params.top_k = options->top_k > 0 ? options->top_k : 40;

    for (size_t i = 0; i < options->stop_sequence_count; i++) {
        if (options->stop_sequences[i]) {
            params.stop_sequences.push_back(options->stop_sequences[i]);
        }
    }

    // Generate
    llama_wrapper::GenerationResult gen_result = model->generate(messages, params);

    result->text = strdup_safe(gen_result.text);
    result->prompt_tokens = gen_result.prompt_tokens;
    result->completion_tokens = gen_result.completion_tokens;
    result->finish_reason = strdup_safe(gen_result.finish_reason);

    return result;
}

LLAMA_API llama_generate_result_t* llama_generate_stream(
    llama_handle_t handle,
    const llama_generate_options_t* options,
    llama_token_callback_t callback,
    void* user_data
) {
    auto* result = create_result();
    if (!result) {
        return nullptr;
    }

    llama_wrapper::LlamaModel* model = nullptr;

    {
        std::lock_guard<std::mutex> lock(g_models_mutex);
        auto it = g_models.find(handle);
        if (it == g_models.end()) {
            result->error = strdup_safe("Invalid model handle");
            result->finish_reason = strdup_safe("error");
            return result;
        }
        model = it->second.get();
    }

    // Convert messages
    std::vector<llama_wrapper::ChatMessage> messages;
    for (size_t i = 0; i < options->message_count; i++) {
        llama_wrapper::ChatMessage msg;
        msg.role = options->messages[i].role ? options->messages[i].role : "";
        msg.content = options->messages[i].content ? options->messages[i].content : "";
        messages.push_back(msg);
    }

    // Set up generation params
    llama_wrapper::GenerationParams params;
    params.max_tokens = options->max_tokens > 0 ? options->max_tokens : 256;
    params.temperature = options->temperature > 0 ? options->temperature : 0.7f;
    params.top_p = options->top_p > 0 ? options->top_p : 0.9f;
    params.top_k = options->top_k > 0 ? options->top_k : 40;

    for (size_t i = 0; i < options->stop_sequence_count; i++) {
        if (options->stop_sequences[i]) {
            params.stop_sequences.push_back(options->stop_sequences[i]);
        }
    }

    // Generate with streaming callback
    llama_wrapper::GenerationResult gen_result = model->generate_streaming(
        messages,
        params,
        [callback, user_data](const std::string& token) -> bool {
            if (callback) {
                return callback(token.c_str(), user_data) == 0;
            }
            return true;
        }
    );

    result->text = strdup_safe(gen_result.text);
    result->prompt_tokens = gen_result.prompt_tokens;
    result->completion_tokens = gen_result.completion_tokens;
    result->finish_reason = strdup_safe(gen_result.finish_reason);

    return result;
}

LLAMA_API void llama_free_result(llama_generate_result_t* result) {
    if (result) {
        free(result->text);
        free(result->finish_reason);
        free(result->error);
        free(result);
    }
}

LLAMA_API const char* llama_get_last_error(void) {
    if (g_last_error.empty()) {
        return nullptr;
    }
    return g_last_error.c_str();
}

LLAMA_API void llama_clear_error(void) {
    g_last_error.clear();
}

} // extern "C"

