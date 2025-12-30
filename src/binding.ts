/**
 * Environment detection and binding router.
 * Automatically selects the correct binding based on the runtime environment.
 */
export interface LoadModelOptions {
  modelPath: string;
  gpuLayers?: number;
  contextSize?: number;
  threads?: number;
  debug?: boolean;
  /**
   * Chat template to use for formatting messages.
   * - "auto" (default): Use the template embedded in the GGUF model file
   * - Template name: Use a specific built-in template (e.g., "llama3", "chatml", "gemma")
   */
  chatTemplate?: string;
}

export interface ChatMessage {
  role: string;
  content: string;
}

export interface GenerateOptions {
  messages: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  stopSequences?: string[];
}

export interface GenerateResult {
  text: string;
  promptTokens: number;
  completionTokens: number;
  finishReason: 'stop' | 'length' | 'error';
}

export interface Binding {
  loadModel(options: LoadModelOptions): Promise<number>;
  unloadModel(handle: number): boolean;
  generate(handle: number, options: GenerateOptions): Promise<GenerateResult>;
  generateStream(
    handle: number,
    options: GenerateOptions,
    onToken: (token: string) => void
  ): Promise<GenerateResult>;
  isModelLoaded(handle: number): boolean;
}

/**
 * Detect if running in Bun runtime
 */
function isBun(): boolean {
  return typeof globalThis.Bun !== "undefined";
}

/**
 * Detect if running in Node.js runtime
 */
function isNode(): boolean {
  return (
    typeof process !== "undefined" &&
    process.versions != null &&
    process.versions.node != null &&
    !isBun()
  );
}

// Lazy-loaded binding
let cachedBinding: Binding | null = null;

/**
 * Get the appropriate binding for the current runtime environment.
 * Throws an error if the runtime is not supported.
 */
async function getBinding(): Promise<Binding> {
  if (cachedBinding) {
    return cachedBinding;
  }

  if (isBun()) {
    // Dynamic import for Bun binding
    const bunBinding = await import("./binding-bun.js");
    cachedBinding = {
      loadModel: bunBinding.loadModel,
      unloadModel: bunBinding.unloadModel,
      generate: bunBinding.generate,
      generateStream: bunBinding.generateStream,
      isModelLoaded: bunBinding.isModelLoaded,
    };
  } else if (isNode()) {
    // Dynamic import for Node binding
    const nodeBinding = await import("./binding-node.js");
    cachedBinding = {
      loadModel: nodeBinding.loadModel,
      unloadModel: nodeBinding.unloadModel,
      generate: nodeBinding.generate,
      generateStream: nodeBinding.generateStream,
      isModelLoaded: nodeBinding.isModelLoaded,
    };
  } else {
    throw new Error(
      "Unsupported runtime environment. Only Node.js and Bun are supported."
    );
  }

  return cachedBinding;
}

// Synchronous binding access for functions that need it
let syncBinding: Binding | null = null;

function getSyncBinding(): Binding {
  if (!syncBinding) {
    throw new Error(
      "Binding not initialized. Call loadModel() first to initialize the binding."
    );
  }
  return syncBinding;
}

/**
 * Load a model from a GGUF file.
 */
export async function loadModel(options: LoadModelOptions): Promise<number> {
  const binding = await getBinding();
  syncBinding = binding;
  return binding.loadModel(options);
}

/**
 * Unload a model.
 */
export function unloadModel(handle: number): boolean {
  return getSyncBinding().unloadModel(handle);
}

/**
 * Generate text from messages (non-streaming).
 */
export async function generate(
  handle: number,
  options: GenerateOptions
): Promise<GenerateResult> {
  const binding = await getBinding();
  return binding.generate(handle, options);
}

/**
 * Generate text from messages (streaming).
 */
export async function generateStream(
  handle: number,
  options: GenerateOptions,
  onToken: (token: string) => void
): Promise<GenerateResult> {
  const binding = await getBinding();
  return binding.generateStream(handle, options, onToken);
}

/**
 * Check if a model is loaded.
 */
export function isModelLoaded(handle: number): boolean {
  return getSyncBinding().isModelLoaded(handle);
}

/**
 * Get the current runtime environment name.
 */
export function getRuntimeName(): "bun" | "node" | "unknown" {
  if (isBun()) return "bun";
  if (isNode()) return "node";
  return "unknown";
}

