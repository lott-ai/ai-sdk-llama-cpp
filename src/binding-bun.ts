import { dlopen, FFIType, suffix, ptr, CString, type Pointer } from "bun:ffi";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { LoadModelOptions, GenerateOptions, GenerateResult } from './binding.js';

const __dirname = dirname(fileURLToPath(import.meta.url));

// Resolve the path to the shared library
const libPath = join(__dirname, "..", "build", "Release", `libllama_ffi.${suffix}`);

// Define FFI symbols for the C API
const lib = dlopen(libPath, {
  llama_load_model: {
    args: [FFIType.ptr],
    returns: FFIType.i32,
  },
  llama_unload_model: {
    args: [FFIType.i32],
    returns: FFIType.bool,
  },
  llama_is_model_loaded: {
    args: [FFIType.i32],
    returns: FFIType.bool,
  },
  llama_generate: {
    args: [FFIType.i32, FFIType.ptr],
    returns: FFIType.ptr,
  },
  llama_generate_stream: {
    args: [FFIType.i32, FFIType.ptr, FFIType.ptr, FFIType.ptr],
    returns: FFIType.ptr,
  },
  llama_free_result: {
    args: [FFIType.ptr],
    returns: FFIType.void,
  },
  llama_get_last_error: {
    args: [],
    returns: FFIType.ptr,
  },
  llama_clear_error: {
    args: [],
    returns: FFIType.void,
  },
});

// Workaround for Bun FFI toArrayBuffer bug: https://github.com/oven-sh/bun/issues/23656
// toArrayBuffer() always returns incorrect byteLength, so we use libc memcpy instead
const libc = dlopen(process.platform === "darwin" ? "libc.dylib" : "libc.so.6", {
  memcpy: {
    args: [FFIType.ptr, FFIType.ptr, FFIType.u64],
    returns: FFIType.ptr,
  },
});

/**
 * Copy memory from a native pointer to a JavaScript ArrayBuffer.
 * This is a workaround for the Bun FFI toArrayBuffer bug.
 * See: https://github.com/oven-sh/bun/issues/23656
 */
function copyFromCPtr(src: Pointer | number, len: number): ArrayBuffer {
  const out = new Uint8Array(len >>> 0);
  libc.symbols.memcpy(ptr(out), src as Pointer, len >>> 0);
  return out.buffer as ArrayBuffer;
}

// Helper to encode string to null-terminated buffer
function encodeString(str: string): Uint8Array {
  const encoder = new TextEncoder();
  const encoded = encoder.encode(str);
  const buffer = new Uint8Array(encoded.length + 1);
  buffer.set(encoded);
  buffer[encoded.length] = 0;
  return buffer;
}

// Helper to read a null-terminated string from a pointer
function readCString(pointer: Pointer | number): string | null {
  const ptrValue = typeof pointer === "number" ? pointer : Number(pointer);
  if (ptrValue === 0) return null;
  return new CString(pointer as Pointer).toString();
}

// Structure sizes and offsets (for 64-bit systems with proper alignment)
// llama_load_options_t layout:
//   const char* model_path;      // offset 0, size 8
//   int32_t gpu_layers;          // offset 8, size 4
//   int32_t context_size;        // offset 12, size 4
//   int32_t threads;             // offset 16, size 4
//   bool debug;                  // offset 20, size 1
//   padding                      // offset 21, size 3
//   const char* chat_template;   // offset 24, size 8
// Total: 32 bytes

const LOAD_OPTIONS_SIZE = 32;

// llama_generate_result_t layout:
//   char* text;                  // offset 0, size 8
//   int32_t prompt_tokens;       // offset 8, size 4
//   int32_t completion_tokens;   // offset 12, size 4
//   char* finish_reason;         // offset 16, size 8
//   char* error;                 // offset 24, size 8
// Total: 32 bytes

const RESULT_TEXT_OFFSET = 0;
const RESULT_PROMPT_TOKENS_OFFSET = 8;
const RESULT_COMPLETION_TOKENS_OFFSET = 12;
const RESULT_FINISH_REASON_OFFSET = 16;
const RESULT_ERROR_OFFSET = 24;

// llama_chat_message_t layout:
//   const char* role;            // offset 0, size 8
//   const char* content;         // offset 8, size 8
// Total: 16 bytes

const MESSAGE_SIZE = 16;

// llama_generate_options_t layout:
//   llama_chat_message_t* messages;  // offset 0, size 8
//   size_t message_count;            // offset 8, size 8
//   int32_t max_tokens;              // offset 16, size 4
//   float temperature;               // offset 20, size 4
//   float top_p;                     // offset 24, size 4
//   int32_t top_k;                   // offset 28, size 4
//   const char** stop_sequences;     // offset 32, size 8
//   size_t stop_sequence_count;      // offset 40, size 8
// Total: 48 bytes

const GENERATE_OPTIONS_SIZE = 48;

// Keep references to buffers to prevent GC
const bufferRefs = new Map<number, unknown[]>();
let nextRefId = 1;

function keepAlive(id: number, refs: unknown[]) {
  bufferRefs.set(id, refs);
}

function releaseRefs(id: number) {
  bufferRefs.delete(id);
}

export function loadModel(options: LoadModelOptions): Promise<number> {
  return new Promise((resolve, reject) => {
    const refId = nextRefId++;
    const refs: unknown[] = [];

    try {
      // Create load options structure
      const optionsBuffer = new ArrayBuffer(LOAD_OPTIONS_SIZE);
      const optionsView = new DataView(optionsBuffer);

      // Encode strings
      const modelPathBuffer = encodeString(options.modelPath);
      refs.push(modelPathBuffer);

      const chatTemplateBuffer = encodeString(options.chatTemplate ?? "auto");
      refs.push(chatTemplateBuffer);

      const modelPathPtr = ptr(modelPathBuffer);
      const chatTemplatePtr = ptr(chatTemplateBuffer);

      // Set pointers and values
      optionsView.setBigUint64(0, BigInt(modelPathPtr), true); // model_path
      optionsView.setInt32(8, options.gpuLayers ?? 99, true); // gpu_layers
      optionsView.setInt32(12, options.contextSize ?? 2048, true); // context_size
      optionsView.setInt32(16, options.threads ?? 4, true); // threads
      optionsView.setUint8(20, options.debug ? 1 : 0); // debug
      optionsView.setBigUint64(24, BigInt(chatTemplatePtr), true); // chat_template

      refs.push(optionsBuffer);
      keepAlive(refId, refs);

      // Create a Uint8Array view that we keep a reference to
      const optionsArray = new Uint8Array(optionsBuffer);
      refs.push(optionsArray);

      // Call the native function
      const handle = lib.symbols.llama_load_model(ptr(optionsArray));

      releaseRefs(refId);

      if (handle < 0) {
        const errorPtr = lib.symbols.llama_get_last_error();
        const error = !isNullPtr(errorPtr) ? readCString(errorPtr as Pointer) : "Failed to load model";
        lib.symbols.llama_clear_error();
        reject(new Error(error ?? "Failed to load model"));
      } else {
        resolve(handle);
      }
    } catch (error) {
      releaseRefs(refId);
      reject(error);
    }
  });
}

export function unloadModel(handle: number): boolean {
  return lib.symbols.llama_unload_model(handle);
}

export function isModelLoaded(handle: number): boolean {
  return lib.symbols.llama_is_model_loaded(handle);
}

function createGenerateOptions(options: GenerateOptions): { buffer: ArrayBuffer; refs: unknown[] } {
  const refs: unknown[] = [];

  // Create message array
  const messagesBuffer = new ArrayBuffer(MESSAGE_SIZE * options.messages.length);
  const messagesView = new DataView(messagesBuffer);
  refs.push(messagesBuffer);

  for (let i = 0; i < options.messages.length; i++) {
    const roleBuffer = encodeString(options.messages[i].role);
    const contentBuffer = encodeString(options.messages[i].content);
    refs.push(roleBuffer, contentBuffer);

    const offset = i * MESSAGE_SIZE;
    messagesView.setBigUint64(offset, BigInt(ptr(roleBuffer)), true);
    messagesView.setBigUint64(offset + 8, BigInt(ptr(contentBuffer)), true);
  }

  // Create stop sequences array if provided
  let stopSeqPtr = 0n;
  if (options.stopSequences && options.stopSequences.length > 0) {
    const stopSeqPtrBuffer = new ArrayBuffer(8 * options.stopSequences.length);
    const stopSeqPtrView = new DataView(stopSeqPtrBuffer);
    refs.push(stopSeqPtrBuffer);

    for (let i = 0; i < options.stopSequences.length; i++) {
      const seqBuffer = encodeString(options.stopSequences[i]);
      refs.push(seqBuffer);
      stopSeqPtrView.setBigUint64(i * 8, BigInt(ptr(seqBuffer)), true);
    }
    stopSeqPtr = BigInt(ptr(new Uint8Array(stopSeqPtrBuffer)));
  }

  // Create generate options structure
  const optionsBuffer = new ArrayBuffer(GENERATE_OPTIONS_SIZE);
  const optionsView = new DataView(optionsBuffer);
  refs.push(optionsBuffer);

  optionsView.setBigUint64(0, BigInt(ptr(new Uint8Array(messagesBuffer))), true); // messages
  optionsView.setBigUint64(8, BigInt(options.messages.length), true); // message_count
  optionsView.setInt32(16, options.maxTokens ?? 256, true); // max_tokens
  optionsView.setFloat32(20, options.temperature ?? 0.7, true); // temperature
  optionsView.setFloat32(24, options.topP ?? 0.9, true); // top_p
  optionsView.setInt32(28, options.topK ?? 40, true); // top_k
  optionsView.setBigUint64(32, stopSeqPtr, true); // stop_sequences
  optionsView.setBigUint64(40, BigInt(options.stopSequences?.length ?? 0), true); // stop_sequence_count

  return { buffer: optionsBuffer, refs };
}

// Helper to check if a pointer is null (can be null, 0, or 0n)
function isNullPtr(ptr: Pointer | number | null): boolean {
  if (ptr === null) return true;
  if (typeof ptr === "number") return ptr === 0;
  if (typeof ptr === "bigint") return ptr === 0n;
  // For Pointer type, convert to number and check
  return Number(ptr) === 0;
}

function parseResult(resultPtr: Pointer): GenerateResult {
  // Get the numeric value of the pointer for null check
  const ptrValue = typeof resultPtr === "number" ? resultPtr : Number(resultPtr);
  if (ptrValue === 0) {
    throw new Error("Generation failed: null result pointer");
  }

  const buffer = copyFromCPtr(resultPtr, 32);
  const view = new DataView(buffer);

  const textPtr = Number(view.getBigUint64(RESULT_TEXT_OFFSET, true));
  const promptTokens = view.getInt32(RESULT_PROMPT_TOKENS_OFFSET, true);
  const completionTokens = view.getInt32(RESULT_COMPLETION_TOKENS_OFFSET, true);
  const finishReasonPtr = Number(view.getBigUint64(RESULT_FINISH_REASON_OFFSET, true));
  const errorPtr = Number(view.getBigUint64(RESULT_ERROR_OFFSET, true));

  const text = textPtr ? readCString(textPtr) ?? "" : "";
  const finishReason = finishReasonPtr ? readCString(finishReasonPtr) ?? "error" : "error";
  const error = errorPtr ? readCString(errorPtr) : null;

  if (error) {
    throw new Error(error);
  }

  return {
    text,
    promptTokens,
    completionTokens,
    finishReason: finishReason as "stop" | "length" | "error",
  };
}

export function generate(handle: number, options: GenerateOptions): Promise<GenerateResult> {
  return new Promise((resolve, reject) => {
    const refId = nextRefId++;

    try {
      const { buffer, refs } = createGenerateOptions(options);
      keepAlive(refId, refs);

      const resultPtr = lib.symbols.llama_generate(handle, ptr(new Uint8Array(buffer)));

      if (isNullPtr(resultPtr)) {
        releaseRefs(refId);
        const errorPtr = lib.symbols.llama_get_last_error();
        const error = !isNullPtr(errorPtr) ? readCString(errorPtr as Pointer) : "Generation failed: null result";
        lib.symbols.llama_clear_error();
        reject(new Error(error ?? "Generation failed: null result"));
        return;
      }

      try {
        const result = parseResult(resultPtr as Pointer);
        resolve(result);
      } catch (parseError) {
        // If parsing fails, check for library error before re-throwing
        const errorPtr = lib.symbols.llama_get_last_error();
        if (!isNullPtr(errorPtr)) {
          const libError = readCString(errorPtr as Pointer);
          lib.symbols.llama_clear_error();
          if (libError) {
            reject(new Error(libError));
            return;
          }
        }
        throw parseError;
      } finally {
        lib.symbols.llama_free_result(resultPtr);
        releaseRefs(refId);
      }
    } catch (error) {
      releaseRefs(refId);
      reject(error);
    }
  });
}

export function generateStream(
  handle: number,
  options: GenerateOptions,
  onToken: (token: string) => void
): Promise<GenerateResult> {
  return new Promise((resolve, reject) => {
    const refId = nextRefId++;

    try {
      const { buffer, refs } = createGenerateOptions(options);
      keepAlive(refId, refs);

      // Collect tokens during generation
      // Note: Bun FFI callbacks are tricky, so we use a workaround
      // The C API will call the callback synchronously during generation
      // We store tokens and emit them after generation completes

      // For now, we use the non-streaming generate and emit all at once
      // A proper streaming implementation would require Bun's callback support
      const resultPtr = lib.symbols.llama_generate(handle, ptr(new Uint8Array(buffer)));

      if (isNullPtr(resultPtr)) {
        releaseRefs(refId);
        const errorPtr = lib.symbols.llama_get_last_error();
        const error = !isNullPtr(errorPtr) ? readCString(errorPtr as Pointer) : "Generation failed: null result";
        lib.symbols.llama_clear_error();
        reject(new Error(error ?? "Generation failed: null result"));
        return;
      }

      try {
        const result = parseResult(resultPtr as Pointer);

        // Emit the full text as tokens (character by character for now)
        // In a proper implementation, we'd use the streaming API with callbacks
        const words = result.text.split(/(\s+)/);
        for (const word of words) {
          if (word) {
            onToken(word);
          }
        }

        resolve(result);
      } catch (parseError) {
        // If parsing fails, check for library error before re-throwing
        const errorPtr = lib.symbols.llama_get_last_error();
        if (!isNullPtr(errorPtr)) {
          const libError = readCString(errorPtr as Pointer);
          lib.symbols.llama_clear_error();
          if (libError) {
            reject(new Error(libError));
            return;
          }
        }
        throw parseError;
      } finally {
        lib.symbols.llama_free_result(resultPtr);
        releaseRefs(refId);
      }
    } catch (error) {
      releaseRefs(refId);
      reject(error);
    }
  });
}
