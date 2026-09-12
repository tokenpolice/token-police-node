/**
 * Active Enforcement Engine.
 * Applies extremely lightweight pre-flight monkey-patches to target SDKs.
 * These patches only run tp.check() and block the call if the budget is 0.
 * They DO NOT parse the response or handle telemetry.
 *
 * Behavior is kept in parity with the TokenPolice Python SDK.
 */

import { performance } from "node:perf_hooks";
import { AsyncLocalStorage } from "node:async_hooks";
import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { join } from "node:path";
import { failSafeSync, failSafeAsync } from "./safe";
import { getClient } from "./state";
import * as state from "./state";
import {
  evaluate as localEvaluate,
  effectiveProvider,
} from "./localEvaluator";
import { isNoopReroute, preferRequestedModel } from "./rerouteNoop";
import {
  stashLocalDecision as _storeStashLocalDecision,
  claimLocalDecision,
  type LocalDecision,
} from "./localDecisionStore";
import {
  stashRoutingMarker,
  copySessionMetadata,
  stampRoutingMarker,
  rowMetadataFromSession,
} from "./routingMarkerStore";
import { buildCallOutcome } from "./_classify";
import { TokenPoliceBlockedError } from "./exceptions";
import {
  getCurrentSession,
  consumePendingSpanName,
  manualSpanIds,
  runWithReservedSpanOrder,
  runWithAnthropicStreamOtelSuppress,
  runWithLlamaIndexOtelSuppress,
} from "./context";
import {
  buildPromptComposition,
  buildResponseComposition,
  extractPendingToolCalls,
  codePointLength,
} from "./composition";
import {
  maybeRegisterOpenAIAgentsTracing,
  wrapLlamaIndexTools,
} from "./frameworkTools";

let _isInstrumented = false;

/**
 * Logger scoped to the enforcement engine.
 *
 * `debug` is silent unless the active client was created with `logErrors: true`
 * (mirrors the same gate in telemetry.ts / client.ts — the three loggers are
 * intentionally per-module but share this mechanism). Pass a `() => string`
 * callback to defer building an expensive message until debug is actually on.
 */
const logger = {
  debug: (msg: string | (() => string)) => {
    try {
      if ((getClient() as any)?.logErrors) {
        console.log(`[TokenPolice Debug] ${typeof msg === "function" ? msg() : msg}`);
      }
    } catch { /* fail-open: silent */ }
  },
};

// ── App-first provider-SDK resolution ────────────────────────────
//
// The enforcer must attach to the SAME provider-SDK module copy the app
// actually calls, and telemetry (telemetry.ts) already resolves app-first via
// createRequire(process.cwd()/node_modules). The old bare `require(...)` here
// was the wrong anchor twice over: (1) with a `file:`/pnpm/monorepo install,
// Node realpaths the SDK directory and a bare require resolves the SDK's OWN
// nested devDependency copies — never the app's copy; (2) in the ESM dist,
// esbuild rewrites bare `require` into a `__require` shim that THROWS
// ("Dynamic require of ... is not supported"), so every resolution silently
// failed under `import`. Mirroring telemetry's anchors exactly (app
// node_modules first, SDK-relative second) keeps both layers on the same
// module record — that identity is the invariant enforcement depends on.
//
// TIMING CONTRACT: the app anchor is evaluated PER RESOLUTION ATTEMPT (see
// _getAppRequire below), never frozen at SDK-module-load time. Telemetry
// constructs its appRequire inside each _apply* call, so an app that starts
// with a different cwd (systemd/pm2 launch from `/`), loads token-police via
// a static import, and only then `process.chdir(APP_ROOT)` before init() must
// still resolve from the FINAL cwd on both layers. Resolution runs only at
// init()/uninstrument() time — never per LLM call — so re-constructing the
// anchor each time is free. Only the SUCCESS cache (_resolvedModules) pins a
// copy, deliberately, so uninstrument() restores onto the exact object that
// was wrapped.

/**
 * Require anchored at the APP's node_modules — constructed from the CURRENT
 * process.cwd() at each call, mirroring the per-call appRequire construction
 * in telemetry.ts (never frozen at module-load time; see the timing contract
 * above). `undefined` when no rung is available. Never throws.
 */
function _getAppRequire(): NodeRequire | undefined {
  try {
    return createRequire(join(process.cwd(), "node_modules"));
  } catch {
    // createRequire/process.cwd can throw on a hostile/deleted cwd — fall back
    // to the ambient require (real in the CJS build; in the ESM build it is
    // esbuild's throwing shim, which is why every CALL of the returned anchor
    // sits in a try/catch).
    try {
      return typeof require !== "undefined" ? require : undefined;
    } catch {
      return undefined;
    }
  }
}

/** SDK-relative require. Mirrors the sdkRequire chain in telemetry.ts —
 * esbuild/tsup substitutes import.meta.url in the CJS output, so this is safe
 * across both dist files. `undefined` when no rung is available. */
const _sdkRequire: NodeRequire | undefined = (() => {
  try {
    return createRequire(import.meta.url);
  } catch {
    /* fall through */
  }
  try {
    if (typeof require !== "undefined") return require;
  } catch {
    /* fall through */
  }
  try {
    return createRequire(join(process.cwd(), "node_modules"));
  } catch {
    return undefined;
  }
})();

/**
 * Modules the enforcer has resolved (and possibly wrapped), keyed by module
 * name. uninstrument() must restore onto the EXACT module object that was
 * wrapped, so successful resolutions are cached and reused for the lifetime
 * of the process.
 */
const _resolvedModules = new Map<string, any>();

/**
 * Resolve a provider SDK module app-first (the app's node_modules), then
 * SDK-relative — the same order telemetry uses for instrumentors, so both
 * layers attach to the same module copy. Never throws; returns undefined when
 * the module can't be loaded, preserving the silent-skip semantics the old
 * bare `require` call sites had (optional peer deps).
 */
function resolveProviderModule(moduleName: string): any | undefined {
  // Anchor list is built at CALL time so the app anchor tracks the current
  // cwd (timing contract above); _sdkRequire is static and stays frozen.
  return _resolveWithAnchors(moduleName, [_getAppRequire(), _sdkRequire], _resolvedModules);
}

/** Anchor-order resolution core behind resolveProviderModule — split out so
 * the test suite can inject its own anchors/cache. Never throws. */
function _resolveWithAnchors(
  moduleName: string,
  anchors: Array<NodeRequire | undefined>,
  cache: Map<string, any>,
): any | undefined {
  try {
    if (cache.has(moduleName)) return cache.get(moduleName);
    for (const req of anchors) {
      if (!req) continue;
      try {
        const mod = req(moduleName);
        cache.set(moduleName, mod);
        return mod;
      } catch {
        /* not resolvable from this anchor — try the next */
      }
    }
  } catch {
    /* fail-open: resolution must never throw into the wrap/init path */
  }
  return undefined;
}

/**
 * Whether a module resolves from the APP's node_modules. Presence in the app
 * is what makes a zero-wrap silence dangerous (the app is calling an SDK we
 * never attached to) — SDK-only devDependency copies don't count. Never
 * throws.
 */
function _appCanResolve(moduleName: string): boolean {
  try {
    // Fresh anchor per call — the zero-wrap audit must judge presence against
    // the app's CURRENT cwd, not the cwd at SDK load time (timing contract
    // above), or a chdir-after-import app would have its warning suppressed
    // in exactly the state where enforcement missed the app copy.
    const appReq = _getAppRequire();
    if (!appReq) return false;
    appReq.resolve(moduleName);
    return true;
  } catch {
    return false;
  }
}

/** Node's global CJS module cache, reached through whichever require anchor is
 * available (createRequire instances share the one global cache). Never
 * throws; returns an empty object when no cache is reachable. */
function _requireCache(): Record<string, any> {
  try {
    return (_getAppRequire() as any)?.cache ?? (_sdkRequire as any)?.cache ?? {};
  } catch {
    return {};
  }
}

/**
 * Wrapped-target counts per `_TARGET_METHODS` moduleName. Feeds the zero-wrap
 * warning in client.ts init() (a provider present in the app but with zero
 * wraps means enforcement silently degraded). Reset by uninstrument().
 */
const _wrappedTargetCounts = new Map<string, number>();

/**
 * Zero-wrap audit for init(): distinct `_TARGET_METHODS` module names that ARE
 * resolvable from the app's node_modules but received ZERO enforcement wraps.
 * These are exactly the packages whose calls would silently skip the
 * pre-flight /check, get logged under the wrong provider, and (for
 * manual-telemetry targets) lose all telemetry/cost. Never throws — returns []
 * on any internal failure.
 */
export function getUnwrappedResolvableProviders(
  canResolve: (moduleName: string) => boolean = _appCanResolve,
): string[] {
  try {
    const names: string[] = [];
    const seen = new Set<string>();
    for (const t of _TARGET_METHODS) {
      if (seen.has(t.moduleName)) continue;
      seen.add(t.moduleName);
      if ((_wrappedTargetCounts.get(t.moduleName) ?? 0) > 0) continue;
      if (canResolve(t.moduleName)) names.push(t.moduleName);
    }
    return names;
  } catch {
    return [];
  }
}

/**
 * Describes a method to protect with pre-flight enforcement.
 */
interface TargetMethod {
  /** npm package name to import. */
  moduleName: string;
  /** Class/object to patch (e.g., "OpenAI" — or null for module-level). */
  objectPath: string[];
  /** Method name to wrap. */
  method: string;
  /** Whether the method is async. */
  isAsync: boolean;
  /**
   * When true, the wrapper also extracts token usage from the response and
   * logs it directly via tp.log() — used for SDKs that have no OpenLLMetry
   * instrumentor (e.g. @google/genai), so the telemetry SpanProcessor never
   * sees them.
   */
  manualTelemetry?: boolean;
  /** When true (with manualTelemetry), the method returns an async iterable. */
  streaming?: boolean;
  /**
   * Explicit provider name override. When unset, the provider is derived from
   * moduleName via _detectProvider. Needed when two entries share the same
   * moduleName but represent different API shapes — e.g. OpenAI Chat
   * Completions (provider="openai") vs the OpenAI Responses API
   * (provider="openai_responses").
   */
  provider?: string;
  /**
   * Modality hint for non-text generation: one of `image_gen`, `audio_tts`,
   * `audio_stt`, `video_gen`, `ocr`. Activates the modality code path —
   * pre-flight `/check` carries an intent, telemetry uses an explicit
   * usage shape, and items/duration are extracted from the call via
   * the registered handler in `MODALITY_HANDLERS`.
   */
  modality?: "image_gen" | "audio_tts" | "audio_stt" | "video_gen" | "ocr";
  /**
   * Explicit usage shape (e.g. `openai_images`,
   * `google_imagen`, `together_image`). Required when `modality` is set.
   * For embedding entries (operation="embedding"), set to the per-provider
   * embedding shape (`openai_embeddings`, `voyage_embed`, etc.).
   */
  shape?: string;
  /**
   * Operation type — when "embedding", the wrapper extracts usage via the
   * embedding-aware extractor, builds an input-only composition, and tags
   * the /log payload with `operation: "embedding"` so the dashboard can
   * segment spend and the loop-detector can skip bulk ingest traces.
   * Defaults to "chat" (omitted).
   */
  operation?: string;
}

/**
 * Registry of SDK methods to protect with pre-flight checks.
 * We only protect the outermost execution methods.
 */
const _TARGET_METHODS: TargetMethod[] = [
  // OpenAI — Chat Completions
  {
    moduleName: "openai",
    objectPath: ["OpenAI", "Chat", "Completions", "prototype"],
    method: "create",
    isAsync: true,
  },
  // OpenAI — Legacy Completions
  {
    moduleName: "openai",
    objectPath: ["OpenAI", "Completions", "prototype"],
    method: "create",
    isAsync: true,
  },
  // OpenAI — Responses API. Used by the @openai/agents SDK
  // (Runner / run() invoke client.responses.create internally) and by direct
  // Responses callers. @traceloop/instrumentation-openai@0.27 wraps
  // Responses.create (non-stream only), but we keep the manual telemetry
  // path for full coverage: extract usage from response.usage
  // (non-streaming) or the final response.completed stream event. The
  // wrapper's runtime check (result[Symbol.asyncIterator]) auto-routes
  // streaming calls (responses.create({ stream: true, ... })) through the
  // stream wrapper, so we don't set `streaming: true` here.
  {
    moduleName: "openai",
    objectPath: ["OpenAI", "Responses", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
    provider: "openai_responses",
  },
  // Anthropic — Messages
  {
    moduleName: "@anthropic-ai/sdk",
    objectPath: ["Anthropic", "Messages", "prototype"],
    method: "create",
    isAsync: true,
  },
  // Anthropic — Beta.Messages. A SIBLING class (extends APIResource, NOT
  // Messages — verified in both 0.30.1 and 0.90.0), so patching
  // Messages.prototype above does NOT reach it: without this row
  // `client.beta.messages.*` gets ZERO pre-flight /check — no block, no
  // reroute, no budget gate. Telemetry was already covered (traceloop
  // instruments beta), so this row is enforcement-only. Deliberately NO rows
  // for countTokens / parse / beta.messages.batches: countTokens is a free
  // metadata endpoint — a /check there could block a token-counting call for
  // zero revenue protection.
  {
    moduleName: "@anthropic-ai/sdk",
    objectPath: ["Anthropic", "Beta", "Messages", "prototype"],
    method: "create",
    isAsync: true,
  },
  // Cohere v2 (cohere-ai — CohereClientV2) is instrumented separately by
  // _instrumentCohere. Its `chat` / `chatStream` are instance-bound arrow
  // fields (set in the constructor), not prototype methods, so they cannot be
  // patched via a static object path. @traceloop/instrumentation-cohere only
  // supports the legacy v1 CohereClient, so v2 token usage is extracted via
  // the manual-telemetry path (reading response.usage directly).
  // Google GenAI (@google/genai).
  // The public `generateContent` / `generateContentStream` are instance
  // arrow-fields (not on the prototype, so un-patchable). They delegate to
  // these real prototype methods, which make the actual API call and return
  // the response carrying `usageMetadata`. There is no OpenLLMetry instrumentor
  // for @google/genai, so we extract tokens manually here.
  {
    moduleName: "@google/genai",
    objectPath: ["Models", "prototype"],
    method: "generateContentInternal",
    isAsync: true,
    manualTelemetry: true,
  },
  {
    moduleName: "@google/genai",
    objectPath: ["Models", "prototype"],
    method: "generateContentStreamInternal",
    isAsync: true,
    manualTelemetry: true,
    streaming: true,
  },
  // AWS Bedrock — AWS SDK v3 command pattern: client.send(new ConverseCommand()).
  // `send` is generic across all commands, so the wrapper filters to Converse
  // calls only. The Node SDK ships no Bedrock OpenLLMetry instrumentor (its
  // package.json bundles only the traceloop openai + anthropic instrumentors),
  // so there is nothing in the path to populate gen_ai.usage.* for Converse.
  // Token usage is therefore extracted manually here (manualTelemetry) by
  // reading response.usage directly (see _extractUsage, provider "bedrock").
  {
    moduleName: "@aws-sdk/client-bedrock-runtime",
    objectPath: ["BedrockRuntimeClient", "prototype"],
    method: "send",
    isAsync: true,
    manualTelemetry: true,
  },
  // Cerebras (native @cerebras/cerebras_cloud_sdk). No OpenLLMetry instrumentor
  // exists, so this wrapper extracts token usage manually (manualTelemetry).
  // The SDK is OpenAI-compatible — `Cerebras.Chat.Completions.prototype.create`
  // makes the API call and returns an OpenAI-shaped response carrying `.usage`.
  {
    moduleName: "@cerebras/cerebras_cloud_sdk",
    objectPath: ["Cerebras", "Chat", "Completions", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
  },
  // Together.ai (native together-ai npm package). No OpenLLMetry-JS instrumentor
  // exists for Together, so this wrapper handles telemetry manually. The SDK is
  // OpenAI-compatible — `Together.Chat.Completions.prototype.create` returns
  // either an OpenAI-shaped response (non-stream) or an AsyncIterable of
  // OpenAI-shaped chunks (stream=true). The streaming path is detected at
  // runtime via Symbol.asyncIterator on the resolved value, so no `streaming`
  // hint is needed here.
  {
    moduleName: "together-ai",
    objectPath: ["Together", "Chat", "Completions", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
  },
  // Groq (native `groq-sdk` npm package). Groq ships its own Stainless-generated
  // package (`import Groq from "groq-sdk"`) with a private client — it is NOT
  // the `openai` package, so no OpenLLMetry-JS instrumentor patches it and a
  // native-groq call would be un-metered AND un-enforced. Handled manually
  // (manualTelemetry), mirroring cerebras/together. Groq is OpenAI-compatible:
  // `Groq.Chat.Completions.prototype.create` returns an OpenAI-shaped response
  // (non-stream `.usage` is top-level OpenAI-shaped) or, for stream=true, an
  // AsyncIterable of chunks whose usage rides the FINAL chunk under
  // `chunk.x_groq.usage` (top-level `chunk.usage` is absent) — unwrapped in
  // _chunkHasUsage / _extractUsage.
  {
    moduleName: "groq-sdk",
    objectPath: ["Groq", "Chat", "Completions", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
  },
  // LangChain (@langchain/core BaseChatModel) is patched separately by
  // _instrumentLangChain — see autoInstrument below. The CJS module returned
  // by require("@langchain/core/language_models/chat_models") is a DIFFERENT
  // object from the ESM module that @langchain/openai / @langchain/anthropic
  // / @langchain/google-genai actually consume, so a generic _TARGET_METHODS
  // entry would patch the wrong prototype.

  // ── Non-text modalities (image / audio / video) ────────────────────────
  // Each entry runs on the manual-telemetry path and emits an explicit
  // usage shape with items/duration produced by the registered handler.
  // OpenAI — images.generate (DALL-E 2/3, gpt-image-1/2)
  {
    moduleName: "openai",
    objectPath: ["OpenAI", "Images", "prototype"],
    method: "generate",
    isAsync: true,
    manualTelemetry: true,
    modality: "image_gen",
    shape: "openai_images",
  },
  // OpenAI — audio.speech.create (tts-1, gpt-4o-mini-tts)
  {
    moduleName: "openai",
    objectPath: ["OpenAI", "Audio", "Speech", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
    modality: "audio_tts",
    shape: "openai_audio_tts",
  },
  // OpenAI — audio.transcriptions.create (whisper-1, gpt-4o-transcribe)
  {
    moduleName: "openai",
    objectPath: ["OpenAI", "Audio", "Transcriptions", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
    modality: "audio_stt",
    shape: "openai_audio_stt",
  },
  // OpenAI — audio.translations.create
  {
    moduleName: "openai",
    objectPath: ["OpenAI", "Audio", "Translations", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
    modality: "audio_stt",
    shape: "openai_audio_stt",
  },
  // Google Imagen + Veo. Method names follow the @google/genai "Internal"
  // delegate pattern; if a release exposes only the public alias the lookup
  // silently no-ops.
  {
    moduleName: "@google/genai",
    objectPath: ["Models", "prototype"],
    method: "generateImagesInternal",
    isAsync: true,
    manualTelemetry: true,
    modality: "image_gen",
    shape: "google_imagen",
    provider: "google",
  },
  {
    moduleName: "@google/genai",
    objectPath: ["Models", "prototype"],
    method: "generateImages",
    isAsync: true,
    manualTelemetry: true,
    modality: "image_gen",
    shape: "google_imagen",
    provider: "google",
  },
  {
    moduleName: "@google/genai",
    objectPath: ["Models", "prototype"],
    method: "generateVideosInternal",
    isAsync: true,
    manualTelemetry: true,
    modality: "video_gen",
    shape: "google_veo",
    provider: "google",
  },
  {
    moduleName: "@google/genai",
    objectPath: ["Models", "prototype"],
    method: "generateVideos",
    isAsync: true,
    manualTelemetry: true,
    modality: "video_gen",
    shape: "google_veo",
    provider: "google",
  },
  // ── Embeddings ──────────────────────────────────────────────────────
  // Manual-telemetry path: no OpenLLMetry-JS instrumentor covers any
  // embedding endpoint, and embedding usage is input-only (no streaming,
  // vector output). _extractEmbeddingUsage handles per-provider response
  // shapes; the composition layer's `operation="embedding"` mode emits
  // input-only entries. Embedding traffic is exempt from the server's
  // loop/anomaly detection, so bulk RAG ingest is never throttled.
  // OpenAI — embeddings.create
  {
    moduleName: "openai",
    objectPath: ["OpenAI", "Embeddings", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
    operation: "embedding",
    shape: "openai_embeddings",
  },
  // Google GenAI — Models.prototype.embedContentInternal. The public
  // `embedContent` arrow-field delegates here (mirrors generateContentInternal).
  {
    moduleName: "@google/genai",
    objectPath: ["Models", "prototype"],
    method: "embedContentInternal",
    isAsync: true,
    manualTelemetry: true,
    operation: "embedding",
    shape: "google_genai_embeddings",
    provider: "google",
  },
  {
    moduleName: "@google/genai",
    objectPath: ["Models", "prototype"],
    method: "embedContent",
    isAsync: true,
    manualTelemetry: true,
    operation: "embedding",
    shape: "google_genai_embeddings",
    provider: "google",
  },
  // Together.ai — Together.Embeddings.prototype.create.
  {
    moduleName: "together-ai",
    objectPath: ["Together", "Embeddings", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
    operation: "embedding",
    shape: "together_embed",
  },
  // Cohere v2, Mistral, HuggingFace embeddings: their target methods are
  // instance-bound arrow-fields (Cohere, Mistral) or require-cache
  // submodule exports (HuggingFace), so they cannot be patched via a
  // static objectPath walk. They're patched inside their dedicated
  // _instrument* helpers — see _instrumentCohere / _instrumentMistral /
  // _instrumentHuggingFace below.

  // Together-AI — images (FLUX, SD3). together-ai <0.30 exposes
  // `Images.prototype.create`; ≥0.30 renamed it to `Images.prototype.generate`
  // (matching the Python SDK). Both entries are registered: exactly one
  // resolves on any install and _wrapMethod skips the other (method not
  // found → fail-open return, diagnostic only under logErrors).
  {
    moduleName: "together-ai",
    objectPath: ["Together", "Images", "prototype"],
    method: "create",
    isAsync: true,
    manualTelemetry: true,
    modality: "image_gen",
    shape: "together_image",
  },
  {
    moduleName: "together-ai",
    objectPath: ["Together", "Images", "prototype"],
    method: "generate",
    isAsync: true,
    manualTelemetry: true,
    modality: "image_gen",
    shape: "together_image",
  },
];

/**
 * Stores original methods so we can restore them on uninstrument(), and doubles
 * as the "already instrumented" dedup guard (`_originals.has(key)`).
 */
const _originals = new Map<string, Function>();

/**
 * Restore thunks captured at patch time — the authoritative way uninstrument()
 * puts every wrapped method back. Each thunk closes over the exact target
 * object (prototype/class) + method + original function that was patched, so
 * restoration is a direct `target[method] = original` with no re-resolution.
 *
 * This exists because many classes are reached via the "probe-instance trick"
 * (patch a prototype obtained from a throw-away instance, because the class is
 * NOT exported at the package root — Cohere v2's V2Client, OpenRouter's Chat,
 * Voyage, the LangChain/LlamaIndex framework classes, Anthropic Batches). The
 * old string-key restore path re-resolved `require(moduleName)` + an object
 * path off the package root, which silently no-ops for every such class (and,
 * for Voyage, actively wrote a bogus static method onto the constructor). A
 * thunk captured at the patch site restores correctly regardless of how the
 * target was reached. Thunks also clear any per-prototype idempotency marker so
 * a later re-instrument() re-wraps. Reset on uninstrument().
 */
const _restoreThunks: Array<() => void> = [];

/**
 * HuggingFace patches target require-cache submodule exports, not a path on the
 * package root — uninstrument() cannot resolve them via _resolvePath, so we
 * remember the exact exports object + original function here for restoration.
 */
const _hfRestore: Array<{
  exportsObj: any;
  method: string;
  original: Function;
}> = [];

/**
 * The Mistral `Chat` class is not root-exported from `@mistralai/mistralai`,
 * so we reach `Chat.prototype` via the probe-instance trick. uninstrument()
 * cannot get back to that prototype via _resolvePath, so we remember each
 * patched prototype + method here for restoration.
 */
const _mistralRestore: Array<{
  proto: any;
  method: string;
  original: Function;
}> = [];

/**
 * Vercel AI SDK LanguageModel classes (`@ai-sdk/*`, gateway, community
 * providers) aren't root-exported — the instrumenter walks
 * `Object.getPrototypeOf(probe)` to reach each prototype. uninstrument()
 * cannot get back to those prototypes via _resolvePath on a package root, so
 * we remember each patched prototype + method here for restoration.
 */
const _aiSdkRestore: Array<{
  proto: any;
  method: string;
  original: Function;
}> = [];

/**
 * The Anthropic `Messages` / `AsyncMessages` `.stream()` context-manager
 * helper is patched in place on the class prototype. uninstrument() restores
 * from here and clears the `__tpStreamPatched` idempotency marker so a later
 * re-instrument works.
 */
const _anthropicStreamRestore: Array<{
  proto: any;
  method: string;
  original: Function;
}> = [];

/**
 * Prototypes already patched by the AI SDK instrumenter. The same
 * LanguageModel class can be reached from several entry points (factory call,
 * `.languageModel()`, `.chat()`, singleton) and several module copies, so
 * dedupe by prototype identity rather than by string key. Reassigned (not
 * cleared — WeakSet has no clear) on uninstrument so re-instrumenting works.
 */
let _patchedAiSdkProtos = new WeakSet<object>();

/**
 * Push a firewall observation, deduped within an Anthropic `.stream()` window.
 *
 * A `.stream()` call normally runs ONE pre-flight (single-flight): the
 * `.stream()` wrapper kicks off THE check and publishes it on the per-call
 * latch (`checkPromise`/`checkedBody`); the `create({stream:true})` the SDK
 * internally delegates to awaits that shared promise before dispatch and syncs
 * the reroute model from `checkedBody` onto its own body copy (the only
 * wire-visible effect — the vendor copied params before the check resolved).
 * Degraded paths (no AsyncLocalStorage, kickoff failure, latch not inherited)
 * fall back to layer-local checks and may run TWO — never zero — and those
 * two evaluate the same rules, pushing identical observations. Hence this
 * dedupe stays: at most one observation per (rule_id, outcome) per stream
 * window, on the same per-call latch that dedupes the log row. Outside a
 * stream window `getStore()` is undefined → a plain push, unchanged.
 * Fail-open: any bookkeeping error degrades to pushing anyway.
 */
/**
 * Wrap a wrapper function so each invocation runs inside a per-call
 * observations scope (state.runWithCallObsScope): the pre-flight's pushed
 * observations get tagged with this call's minted key and the call's own
 * drain (success/failure/block) claims exactly them. Reuse-if-exists is
 * handled inside runWithCallObsScope — nested wrapper invocations
 * (vercel→provider, `.stream()`→internal create, langchain→provider) are
 * SDK-internal delegation of the SAME logical customer call and share one
 * key. Preserves `this`, arguments, return value, and error propagation
 * exactly (TokenPoliceBlockedError still surfaces unchanged); on any scope
 * failure the body runs bare (untagged pushes ≈ pre-keying behavior).
 */
/**
 * Guarded accessors over the state module's obs-scope surface. The typeof
 * gates make the enforcer degrade to a bare invocation (untagged pushes,
 * null-claim drains ≈ pre-keying behavior) when the export is absent — e.g.
 * a partial test mock of ../src/state — instead of throwing into the
 * customer's call.
 */
function _runWithCallObsScope<T>(fn: () => T): T {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const run = (state as any).runWithCallObsScope;
  if (typeof run === "function") return run(fn);
  return fn();
}

function _reenterObsScope<T>(key: string | null, fn: () => T): T {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const run = (state as any).runWithObsKey;
  if (typeof run === "function") return run(key, fn);
  return fn();
}

/** The current call's obs key, or null (also when the export is absent). */
function _currentObsKey(): string | null {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const get = (state as any).getCurrentObsKey;
    return (typeof get === "function" ? get() : null) ?? null;
  } catch {
    return null;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _withCallObsScope<T extends (...args: any[]) => any>(fn: T): T {
  return function (this: any, ...args: any[]) {
    return _runWithCallObsScope(() => fn.apply(this, args));
  } as T;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _pushObservationOnce(obs: any): void {
  try {
    const latch = _anthropicStreamLatchStorage.getStore();
    if (latch) {
      const key = `${(obs && obs.rule_id) || ""}|${(obs && obs.outcome) || ""}`;
      if (!latch.observed) latch.observed = new Set<string>();
      if (latch.observed.has(key)) return;
      latch.observed.add(key);
    }
  } catch {
    // fail-open: fall through to a plain push
  }
  state.pushObservation(obs);
}

/**
 * Apply a REROUTE directive returned from /check by mutating the call's
 * model field in place. Only acts when reroute.mode === "enforce" — shadow
 * directives are ignored on the SDK side (the server already recorded
 * the would-be reroute). Fail-safe: never throws.
 *
 * `body` is the request payload object (for SDKs that take `{ model, ... }`
 * as a single argument) or `null` when the SDK uses positional args.
 */
/**
 * Apply a REROUTE directive. Returns status for audit wiring:
 * - "applied" — model field swapped
 * - "rejected" — directive refused (unappliable call shape / cross-provider /
 * serving-unverified); a `reroute_rejected` observation was pushed
 * - "noop" — nothing to do (no directive, dry_run, alias-noop target)
 * Never throws (fail-open).
 */
function _applyReroute(
  result: any,
  body: Record<string, any> | null | undefined,
  provider: string | undefined,
  // Unrecognized custom base_url — refuse the swap even when the
  // module/serving slug would otherwise equal the target (server may still
  // return a REROUTE based on the module provider in the check payload).
  servingUnverified: boolean = false,
  // The pre-flight's matching-only model hint (body-less call shapes). Audit
  // context ONLY — used for the `from.model` on a rejection observation; never
  // a swap target.
  modelHint?: string | null,
): "applied" | "rejected" | "noop" {
  try {
    const reroute = result?.reroute;
    if (!reroute || reroute.mode !== "enforce") return "noop";
    const target = reroute.model;
    if (!target || typeof target !== "string") return "noop";

    const pushRejected = (reason: string) => {
      try {
        _pushObservationOnce({
          rule_id: reroute.rule_id || null,
          outcome: "reroute_rejected",
          mode: "enforce",
          rejection_reason: reason,
          reroute: {
            from: {
              provider: reroute.original?.provider ?? provider ?? "",
              model:
                (body && typeof body === "object" && body.model) ||
                modelHint ||
                "",
            },
            to: { provider: reroute.provider || "", model: target },
          },
          // Only include rule_name when a non-empty string is present so
          // Node/Python observation shapes match (omit key; never null).
          ...(typeof reroute.rule_name === "string" && reroute.rule_name
            ? { rule_name: reroute.rule_name }
            : {}),
        });
      } catch {
        // fail-open
      }
    };

    // A live ENFORCE directive on a call shape whose body has no top-level
    // `model` (framework hint-only paths, Bedrock Converse, native OpenRouter
    // envelope) can never be applied here — record the refusal instead of
    // resolving silently. Ordered FIRST so an unappliable shape is reported as
    // exactly that, not misattributed to serving_unverified or (when the
    // framework provider couldn't be derived) cross_provider_unsupported.
    const appliable = !!(body && typeof body === "object" && "model" in body);
    if (!appliable) {
      pushRejected("unappliable_call_shape");
      return "rejected";
    }
    if (servingUnverified) {
      pushRejected("serving_unverified");
      return "rejected";
    }
    // Skip cross-provider reroutes: the local evaluator rejects them
    // (`cross_provider_unsupported`), so this apply path must match — run the
    // directive's `reroute.provider` and the call provider through the same
    // `effectiveProvider()` (trim + lowercase + alias canonicalization) the
    // evaluator uses, so alias slugs (e.g. together_ai vs together) compare
    // equal and both sides agree. Fires only when a target provider is present
    // and differs; absent/falsy target provider preserves the existing fallback.
    if (
      reroute.provider &&
      effectiveProvider(String(reroute.provider)) !== effectiveProvider(String(provider || ""))
    ) {
      pushRejected("cross_provider_unsupported");
      return "rejected";
    }
    if (body && typeof body === "object" && "model" in body) {
      // The target resolves to the model already requested (alias vs its
      // dated snapshot) — swapping would only unpin the snapshot and emit a
      // phantom REQUEST_REROUTED. Nothing applied, nothing observed.
      if (isNoopReroute(body.model, target)) return "noop";
      const originalModel = body.model;
      body.model = target;
      try {
        const session = getCurrentSession();
        session.metadata = { ...(session.metadata || {}) } as Record<
          string,
          unknown
        >;
        // Only include rule_name when a non-empty string is present so
        // Node/Python `_tp_routing` shapes match (omit key; never null).
        const routing: Record<string, unknown> = {
          rule_id: reroute.rule_id,
          mode: "enforce",
          original_model: originalModel,
          actual_model: target,
          original_provider: reroute.original?.provider ?? provider,
          actual_provider: reroute.provider ?? provider,
        };
        if (typeof reroute.rule_name === "string" && reroute.rule_name) {
          routing.rule_name = reroute.rule_name;
        }
        (session.metadata as Record<string, unknown>)._tp_routing = routing;
        // The session write above stays exactly as it was — it is the
        // /check-payload input rules may match on, and its shape is pinned by
        // tests. But session metadata is copied into EVERY row the SDK emits
        // afterwards, so on its own it painted this call's marker onto tool
        // spans, agent/chain anchors and unrelated sibling calls. Stash it
        // per-call too: the row side strips `_tp_routing` from every copy and
        // re-adds it only for a model-call row whose own obs key matches this
        // one. Keyed exactly like the `local_decision` stash a few lines of
        // execution later — this runs DURING check execution, i.e. inside the
        // call's obs scope, so the ALS read is this call's own key.
        stashRoutingMarker(session, routing, _currentObsKey());
      } catch {
        // session may be missing — drop the routing metadata silently
      }
      return "applied";
    }
    return "noop";
  } catch {
    // never throw from reroute handling
    return "noop";
  }
}

/**
 * Pre-flight check runner (async). Wrapped with fail-safe.
 *
 * Accepts the request body so a REROUTE directive can mutate `body.model`
 * before the original LLM call is dispatched. Pass `null` when the SDK
 * doesn't take an object-shaped body — a REROUTE directive is then never
 * applied, and the refusal is recorded as REROUTE_REJECTED with reason
 * `unappliable_call_shape` (rules still match for audit purposes).
 */
// Helper: run the local evaluator if daemon mode + healthy cache.
// Returns null when caller should drop to inline /check.
function _localEvaluate(
  tp: ReturnType<typeof getClient>,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  session: any,
  body: Record<string, any> | null | undefined,
  provider: string | undefined,
  intent: Record<string, unknown> | null = null,
  forceShadow: boolean = false,
  // Unrecognized custom base_url — REROUTE refuse only (provider field
  // stays the module/serving slug for matchConditions / groupBy).
  servingUnverified: boolean = false,
  // Matching-only model, for call shapes whose body carries no top-level
  // `model` (Bedrock `modelId`, OpenRouter `chatRequest.model`, framework LLM
  // instances). NEVER written into `body` — a REROUTE never applies there;
  // the enforce-mode refusal is recorded as REROUTE_REJECTED
  // (`unappliable_call_shape`).
  modelHint?: string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
): { decision: any; observations: any[]; armableMiss?: boolean } | null {
  if (!tp || tp.deployment !== "daemon") return null;
  const pack = state.getPack();
  if (pack === null || !state.isCacheHealthy()) return null;
  const targetModel =
    body && typeof body === "object" && typeof body.model === "string"
      ? (body.model as string)
      : typeof modelHint === "string" && modelHint
        ? modelHint
        : "";
  try {
    const ctx: Record<string, unknown> = {
      model: targetModel,
      provider: provider || "",
      trace_id: session?.traceId || "",
    };
    if (servingUnverified) ctx.serving_unverified = true;
    if (intent && typeof intent === "object") {
      ctx.intent = intent;
      const kind = (intent as { kind?: unknown }).kind;
      if (typeof kind === "string" && kind) ctx.modality = kind;
    }
    return localEvaluate(pack, session, ctx, forceShadow);
  } catch {
    return null;
  }
}

/**
 * Should a locally-ALLOWED call still be verified with an inline /check?
 *
 * Entity arming (`entity_blocked` / `entity_rerouted`) reaches the pack ONLY
 * over SSE, so while the stream is down an entity the collector has already
 * blocked keeps evaluating to "allowed" locally — the pack itself stays
 * perfectly healthy, so no existing signal catches it. Verify only when BOTH
 * hold: the allow hinged on an entity-list miss (`armableMiss`), and the stream
 * has been down longer than the configured grace. Everything else — healthy
 * stream, within-grace windows, unguarded traffic, local block/reroute — keeps
 * its exact prior path. Pure local reads, fully guarded: on any doubt it
 * returns false (today's zero-round-trip behavior).
 */
function _streamStaleNeedsVerify(
  tp: ReturnType<typeof getClient>,
  ev: { armableMiss?: boolean } | null,
): boolean {
  try {
    if (!ev || ev.armableMiss !== true) return false;
    const graceSeconds =
      typeof tp?.streamStaleGraceSeconds === "number" ? tp.streamStaleGraceSeconds : 60;
    return !state.isStreamFresh(graceSeconds * 1000);
  } catch {
    return false;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function _stashLocalDecision(
  session: any,
  outcome: string,
  ruleId: string | null,
  verifiedByCheck: boolean,
  reroute?: any,
  ruleName?: string | null,
): void {
  if (!session) return;
  const decision: LocalDecision = {
    outcome,
    rule_id: ruleId,
    mode: "enforce",
    verified_by_check: verifiedByCheck,
  };
  // Surface rule_name on /log applied rows (audit/routing UIs).
  const rn =
    (typeof ruleName === "string" && ruleName) ||
    (reroute && typeof reroute.rule_name === "string" && reroute.rule_name) ||
    "";
  if (rn) decision.rule_name = rn;
  if (reroute) decision.reroute = reroute;
  // Keyed per-call stash (replaces the old flat `session._local_decision`
  // slot, which N concurrent calls overwrote). Every stash site runs DURING
  // check execution, i.e. inside the call's obs scope, so the ALS read here is
  // this call's own key — the same key its /log drain will claim with. Null
  // (no scope / partial state mock) stashes untagged, which any drain claims:
  // exactly the old single-slot behavior on degraded paths.
  _storeStashLocalDecision(session, decision, _currentObsKey());
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _emitLocalBlockLog(tp: ReturnType<typeof getClient>, session: any): void {
  try {
    if (!tp) return;
    // Keyed drains: this runs DURING check execution (inside the call's obs
    // scope), so the ALS read here is the blocking call's own key. Null
    // (no scope) claims only untagged + stale entries.
    const obsKey = _currentObsKey();
    const localDecision = claimLocalDecision(session, obsKey);
    const observations = state.drainObservations(obsKey);
    if (!localDecision && observations.length === 0) return;
    tp.log(
      session.userId,
      session.paidPlan,
      session.workflowName,
      session.sessionId,
      "blocked",        // model-name placeholder; block-decision entries are not billed as calls
      "",
      0, 0, 0,
      // Row metadata, not the live session object: a sibling call's applied
      // reroute must not ride this blocked row (and this row must never
      // mutate the session's own metadata). Keyed to the blocking call, so a
      // call that was rerouted AND then blocked still shows its own marker.
      rowMetadataFromSession(session, obsKey),
      // Locally-blocked rows must carry real span ids like every other /log
      // emission (mirrors the failure path below). Without a span_id the row
      // lands with an empty span_id (collides in the trace tree) and — more
      // importantly — is invisible to the collector's per-span idempotency
      // guard, so an infra replay of the same body double-counts budgets and
      // audit rows. manualSpanIds() also supplies trace_id (the active OTel
      // trace when there is one, else session.traceId — the value this call
      // site used to send) plus the enclosing parent_span_id.
      { ...manualSpanIds(session), span_name: session.workflowName },
      undefined,
      undefined,
      {
        local_decision: localDecision || undefined,
        observations: observations.length > 0 ? observations : undefined,
        planSource: session.planSource,
      },
    );
    // No slot to clear — the claim above already removed this call's entry.
  } catch {
    // fail-safe
  }
}

/**
 * Resolve the `operation` stamped on failure-path /log rows.
 *
 * Prefer registry `operation` (e.g. embedding). Fall back to registry
 * `modality` (image_gen/audio_tts/...) so failed modality rows classify like
 * the success path (`_logModality` uses `operation: modality`). Gemini TTS
 * reuses generateContent with no registry modality pin — when the pre-call
 * intent is audio_tts, force that operation so check/log stay mirrored.
 */
function _resolveFailOperation(
  operation?: string,
  modality?: string,
  intent?: { kind?: unknown } | null,
): string | undefined {
  if (intent && (intent as { kind?: unknown }).kind === "audio_tts") {
    return "audio_tts";
  }
  return operation || modality;
}

/**
 * Stash the attempted model + provider + operation on the session so
 * _emitCallFailureLog can populate them when the original LLM call raises
 * (e.g. invalid api key, wrong model id). Without this, the recorded entry
 * lands with model='unknown'/provider=''/operation='chat' and
 * mis-attributes embedding failures as chat failures.
 *
 * `shape` is the usage shape the wrapper would have logged had the call
 * succeeded (registry / modality / framework override) and `wireKey` is the
 * module wire key (`_wireParseKey`) for host-remapped clients — both let the
 * failure row carry the SAME usage_shape as its successful siblings.
 *
 * `modelHint` is a caller-pre-derived (fail-open) fallback for bodies that
 * carry no top-level `model` (bedrock `modelId`, native-OpenRouter envelope) —
 * used only when `body.model` is absent, never re-derived here.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _stashAttemptContext(
  session: any,
  effProvider: string,
  args: any[],
  operation?: string,
  shape?: string | null,
  wireKey?: string | null,
  modelHint?: string,
): void {
  try {
    if (!session) return;
    session._attempted_provider = effProvider || "";
    session._attempted_operation = operation;
    const body = args && args[0];
    session._attempted_model =
      body && typeof body === "object" && typeof body.model === "string" && body.model
        ? body.model
        : typeof modelHint === "string" && modelHint
          ? modelHint
          : undefined;
    // Written last: these are additive diagnostics, so a hostile setter here
    // must never cost the model/provider/operation fields above.
    session._attempted_shape = shape || undefined;
    session._attempted_wire_key = wireKey || undefined;
  } catch {
    // fail-safe
  }
}

/**
 * Emit a /log entry when the original LLM call raised. Mirrors Python's
 * `_emit_call_failure_log`. Populates `model`/`provider` from the stashed
 * attempt context so the recorded failure has real diagnostic value, and
 * drains any shadow `observations` + `local_decision` so the audit trail
 * around the failed call stays complete.
 *
 * Wrapped in try/catch — the failure-logging path itself must never throw
 * into customer code.
 *
 * Returns true ONLY when a /log call was actually dispatched; false on every
 * early-return / swallowed error. Callers that dedupe layered failure emits
 * (the anthropic `.stream()` latch) must key off this — marking a latch for a
 * no-op emit would suppress the only real row.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _emitCallFailureLog(tp: ReturnType<typeof getClient>, session: any): boolean {
  let dispatched = false;
  try {
    if (!tp || !session) return false;
    const callOutcome = session._call_outcome;
    // Keyed drains. Failure emits run either inside the wrapper body (still in
    // the call's obs scope) or inside a stream-drain callback that re-entered
    // the captured scope via runWithObsKey — either way the ALS read here is
    // the failed call's own key; null claims only untagged + stale entries.
    // The decision claim is destructive, but the only early return below
    // requires it to be absent, so nothing is dropped on that path.
    const obsKey = _currentObsKey();
    const observations = state.drainObservations(obsKey);
    const localDecision = claimLocalDecision(session, obsKey);
    if (!callOutcome && observations.length === 0 && !localDecision) return false;

    const attemptedModel: string =
      (typeof session._attempted_model === "string" && session._attempted_model) || "unknown";
    const attemptedProvider: string =
      (typeof session._attempted_provider === "string" && session._attempted_provider) || "";
    // `_attempted_operation` is stashed by manual / embedding wrappers so
    // failure rows from embedding calls don't mis-default to operation="chat".
    const attemptedOperation: string =
      (typeof session._attempted_operation === "string" && session._attempted_operation) || "chat";

    // The request prompt was captured pre-flight into _pendingCompositions —
    // attach it so failed rows show WHAT was attempted. Guarded: only when
    // exactly one pending prompt exists for this trace (with concurrent
    // in-flight calls the attribution would be ambiguous, and consuming
    // another call's entry would silently strip ITS composition — prefer the
    // old empty-composition behavior there). The success log for a failed
    // call never runs, so consuming the entry here also prevents a stale
    // leak.
    let failedPromptComp: unknown[] | undefined;
    try {
      const prefix = `${session.traceId}:`;
      const withPrompt = Object.keys(session._pendingCompositions || {}).filter(
        (k) =>
          k.startsWith(prefix) &&
          Array.isArray(session._pendingCompositions[k]?.prompt) &&
          session._pendingCompositions[k].prompt.length > 0,
      );
      if (withPrompt.length === 1) {
        failedPromptComp = session._pendingCompositions[withPrompt[0]].prompt;
        delete session._pendingCompositions[withPrompt[0]];
      }
    } catch {
      // fail-safe: composition on failure rows is best-effort
    }

    // N4 (embeddings) generalized by to EVERY operation — without an
    // explicit usage block the client synthesizes `openai_compatible_chat`,
    // stamping an OpenAI CHAT shape onto every failed span whatever the
    // provider (failed cohere rows read `openai_compatible_chat` while their
    // successful siblings read `cohere_chat`). Resolve the shape the success
    // path would have logged, from the same tables it uses. Counts stay zero
    // (a failed call has no usage), so the row is still unmeasured.
    //
    // Priority: the wrapper's own stashed shape (registry / modality /
    // framework override — authoritative, it is the literal value the success
    // path passes as its shape override) → the embedding table → the chat table.
    let failureUsage: { shape: string; raw: Record<string, number> } | undefined;
    try {
      const stashedShape =
        typeof session._attempted_shape === "string" ? session._attempted_shape : "";
      let shape: string;
      if (stashedShape) {
        shape = stashedShape;
      } else if (attemptedOperation === "embedding") {
        shape = _resolveEmbeddingShape(attemptedProvider);
      } else {
        shape = _resolveUsageShape(attemptedProvider, []);
        // The wire shape belongs to the MODULE client, not the serving
        // vendor — an Anthropic SDK pointed at api.minimax.io stashes serving
        // provider "minimax" but speaks `anthropic_messages`, which is exactly
        // what its successful siblings log. Narrow + additive: engages ONLY
        // when the serving slug fell through to the resolver's default AND the
        // wire key resolves to a real entry, so no in-table provider changes
        // behavior. Result-based because the resolver is a switch, not a table
        // we can membership-test; provider "litellm" maps explicitly to that
        // same default, but litellm paths never stash a wire key, so it can
        // never be overridden here.
        const wireKey =
          typeof session._attempted_wire_key === "string"
            ? session._attempted_wire_key
            : "";
        if (shape === "openai_compatible_chat" && wireKey) {
          const wireShape = _resolveUsageShape(wireKey, []);
          if (wireShape !== "openai_compatible_chat") shape = wireShape;
        }
      }
      failureUsage = { shape, raw: { prompt_tokens: 0, total_tokens: 0 } };
    } catch {
      // fail-safe: fall back to the client's synth
      failureUsage = undefined;
    }

    tp.log(
      session.userId,
      session.paidPlan,
      session.workflowName,
      session.sessionId,
      attemptedModel,
      attemptedProvider,
      0, 0, 0,
      // Keyed row metadata (see `_emitLocalBlockLog`): a rerouted call's own
      // failure row keeps `_tp_routing` — that is one of the multi-row cases
      // the store's peek-many semantics exist for — while a sibling's failure
      // row never inherits it.
      rowMetadataFromSession(session, obsKey),
      // Failed manual/REST calls must still carry real span ids — without them
      // the row lands with empty span_id (collides in the trace tree) and empty
      // parent_span_id (floats to the trace root instead of nesting under the
      // enclosing agent/chain). manualSpanIds() resolves the parent off the
      // active OTel span (or the anchored session.rootSpanId) like the success path.
      { ...manualSpanIds(session), span_name: session.workflowName },
      failedPromptComp,
      undefined,
      {
        local_decision: localDecision || undefined,
        observations: observations.length > 0 ? observations : undefined,
        call_outcome: callOutcome || undefined,
        operation: attemptedOperation,
        usage: failureUsage,
        planSource: session.planSource,
      },
    );
    dispatched = true;
    session._call_outcome = null;
    // No slot to clear — the claim above already removed this call's entry.
    session._attempted_model = undefined;
    session._attempted_provider = undefined;
    session._attempted_operation = undefined;
    session._attempted_shape = undefined;
    session._attempted_wire_key = undefined;
  } catch {
    // fail-safe — SDK must never throw out of the failure path
  }
  return dispatched;
}

const _runAsyncCheck = failSafeAsync(
  async (
    body?: Record<string, any> | null,
    provider?: string,
    intent?: Record<string, unknown> | null,
    // Callers whose real wire call re-reads its own arguments (the Bedrock
    // embedding InvokeModel path) pass canReroute=false: block/allow decisions
    // still enforce, but reroute directives are treated as allowed instead of
    // being recorded as applied (which would misreport savings). Every other
    // body-less shape passes canReroute=true with a `modelHint` below, so a
    // live directive resolves as REROUTE_REJECTED rather than silently.
    canReroute: boolean = true,
    // Custom base_url host not in the map — keep `provider` for match/
    // groupBy /check payload, but refuse every REROUTE apply (State A + B).
    servingUnverified: boolean = false,
    // Matching + audit context ONLY, for call shapes whose body has no
    // top-level `model` (Bedrock `modelId`, OpenRouter `chatRequest.model`,
    // LangChain/LlamaIndex/Vercel AI SDK/embedding instances). Used for the
    // local evaluator ctx and the /check payload; never merged into `body`, so
    // `_applyReroute` never applies on these paths (no phantom `_tp_routing` /
    // applied row for a swap the wire call never saw) — the enforce-mode
    // refusal is recorded as REROUTE_REJECTED (`unappliable_call_shape`).
    modelHint?: string,
  ): Promise<void> => {
    const tp = getClient();
    if (!tp) return;
    // Active for any non-'off' deployment. dry_run now runs the EXACT same path
    // as enforce (local eval AND inline /check) in all deployments — it differs
    // only in that it never actually blocks/reroutes at the end. 'off' never
    // reaches here as an active pre-flight.
    const active = tp.firewall !== "off";
    if (!active) return;
    // dry_run computes the REAL decision (forceShadow=false below) and proceeds
    // through /check exactly like enforce, but suppresses the final action.
    const isDryRun = tp.firewall === "dry_run";

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const session: any = getCurrentSession();

    // ── State A: SSE healthy → local eval first ──────────────────
    const ev = _localEvaluate(
      tp, session, body, provider, intent ?? null, false, servingUnverified, modelHint,
    );
    if (ev) {
      for (const obs of ev.observations) _pushObservationOnce(obs);
      // Hot path — unless the allow rests on an entity list a stale stream may
      // no longer be delivering, in which case fall through to the SAME inline
      // /check State B uses (bounded timeout, fail-open on any error).
      if (ev.decision.status === "allowed" && !_streamStaleNeedsVerify(tp, ev)) return;
    }
    const localDecision = ev?.decision || null;

    const targetModel =
      body && typeof body === "object" && typeof body.model === "string"
        ? (body.model as string)
        : typeof modelHint === "string" && modelHint
          ? modelHint
          : undefined;
    const result = await tp.check(
      session.userId,
      session.paidPlan,
      session.workflowName,
      session.sessionId,
      session.metadata,
      session.traceId,
      targetModel,
      provider,
      intent ?? undefined,
      session.planSource,
    );
    // Accept either the camelCase `failOpen` or the deprecated snake_case
    // `fail_open` — check() sets both, but be robust to either being present.
    const checkFailOpen = !!(result.failOpen || result.fail_open);

    if (localDecision && localDecision.status === "blocked") {
      // dry_run: the pre-flight request itself is audited server-side as a
      // dry-run (would-)block decision; suppress the action — never stash, never
      // emit a synthetic block log, never throw. The call proceeds.
      if (isDryRun) return;
      // Deliberate exception to fail-open (by design): during a server outage (checkFailOpen),
      // an entity already present in the last-known-good streamed blocked set still
      // blocks (localDecision === "blocked"). This is not a throw-on-connectivity
      // failure — it is the server's own explicit prior block decision. A non-blocked
      // entity on outage never reaches here and is allowed (fail-open).
      const verified = !checkFailOpen && result.status === "blocked";
      if (verified || checkFailOpen) {
        _stashLocalDecision(session, "blocked", localDecision.rule_id, !checkFailOpen);
        _emitLocalBlockLog(tp, session);
        throw new TokenPoliceBlockedError(
          `TokenPolice: Budget exceeded — ${result.reason || "Policy Violation"}`,
          {
            reason: result.reason,
            ruleId: result.ruleId || localDecision.rule_id || undefined,
            kind: result.detail || (result.status === "blocked" ? "budget" : undefined),
            traceId: result.traceId,
          },
        );
      }
      // /check disagrees → drop the local block.
      return;
    }

    if (localDecision && localDecision.status === "rerouted" && canReroute) {
      // dry_run dial: never apply. Ship a would_reroute observation so /log
      // can emit WOULD_REROUTE — ENFORCE-rule + dry_run dial is the try-before-
      // enforce funnel (customerE); without this the would-KPI goes dark
      // because /check only issues REROUTE_DIRECTIVE_ISSUED for ENFORCE rules.
      if (isDryRun) {
        try {
          const from = localDecision.reroute?.from || {};
          const to = localDecision.reroute?.to || {};
          // mode = per-rule executionMode (ENFORCE here). The dial is only
          // sdk_firewall_mode on the /log header — never overwrite mode with dry_run.
          _pushObservationOnce({
            rule_id: localDecision.rule_id || null,
            ...(typeof localDecision.rule_name === "string" && localDecision.rule_name
              ? { rule_name: localDecision.rule_name }
              : {}),
            outcome: "would_reroute",
            mode: (localDecision.mode as string) || "enforce",
            reroute: {
              from: { provider: from.provider || provider || "", model: from.model || "" },
              to: { provider: to.provider || "", model: to.model || "" },
            },
          });
        } catch {
          // fail-open
        }
        return;
      }
      if (checkFailOpen || result.status === "allowed") {
        const target = localDecision.reroute?.to || {};
        // Pass rule_name from local decision (directive pack `name`).
        const syntheticReroute: Record<string, unknown> = {
          mode: "enforce" as const,
          model: target.model,
          provider: target.provider,
          rule_id: localDecision.rule_id,
          original: localDecision.reroute?.from || {},
        };
        if (typeof localDecision.rule_name === "string" && localDecision.rule_name) {
          syntheticReroute.rule_name = localDecision.rule_name;
        }
        const synthetic = {
          status: "allowed",
          reroute: syntheticReroute,
        };
        // Only claim applied when the swap landed (refuse pushes observation).
        const applyStatus = _applyReroute(
          synthetic, body ?? null, provider, servingUnverified, modelHint,
        );
        if (applyStatus === "applied") {
          _stashLocalDecision(
            session,
            "rerouted",
            localDecision.rule_id,
            !checkFailOpen,
            localDecision.reroute,
            localDecision.rule_name,
          );
        }
        return;
      }
      if (result.status === "blocked") {
        _stashLocalDecision(session, "blocked", localDecision.rule_id, true);
        _emitLocalBlockLog(tp, session);
        throw new TokenPoliceBlockedError(
          `TokenPolice: Budget exceeded — ${result.reason || "Policy Violation"}`,
          {
            reason: result.reason,
            ruleId: result.ruleId || localDecision.rule_id || undefined,
            kind: result.detail || "budget",
            traceId: result.traceId,
          },
        );
      }
      {
        const applyStatus = _applyReroute(
          result, body ?? null, provider, servingUnverified, modelHint,
        );
        // If apply path ran after a local reroute decision, keep the
        // stashed applied decision only when the swap actually landed; on
        // reject the observation was already pushed by _applyReroute.
        if (applyStatus === "applied") {
          _stashLocalDecision(
            session,
            "rerouted",
            result.ruleId || localDecision.rule_id || null,
            true,
            {
              from: result.reroute?.original
                ? { provider: result.reroute.original.provider, model: result.reroute.original.model }
                : localDecision.reroute?.from,
              to: {
                provider: result.reroute?.provider,
                model: result.reroute?.model,
              },
            },
            result.reroute?.rule_name || localDecision.rule_name,
          );
        }
      }
      return;
    }

    // ── State B fallback ─────────────────────────────────────────
    // dry_run dial: suppress enforce action; for an ENFORCE reroute directive
    // still ship would_reroute so the would-KPI is not silent (see State A).
    if (isDryRun) {
      if (canReroute && result?.reroute && result.reroute.mode === "enforce") {
        try {
          const rr = result.reroute;
          // mode = directive/rule executionMode ('enforce'), not the SDK dial.
          _pushObservationOnce({
            rule_id: result.ruleId || rr.rule_id || null,
            ...(typeof rr.rule_name === "string" && rr.rule_name
              ? { rule_name: rr.rule_name }
              : {}),
            outcome: "would_reroute",
            mode: typeof rr.mode === "string" && rr.mode ? rr.mode : "enforce",
            reroute: {
              from: {
                provider: rr.original?.provider || provider || "",
                model: rr.original?.model || (body && typeof body === "object" ? body.model : "") || "",
              },
              to: { provider: rr.provider || "", model: rr.model || "" },
            },
          });
        } catch {
          // fail-open
        }
      }
      return;
    }
    if (result.status === "blocked") {
      _stashLocalDecision(session, "blocked", result.ruleId || null, true);
      _emitLocalBlockLog(tp, session);
      throw new TokenPoliceBlockedError(
        `TokenPolice: Budget exceeded — ${result.reason || "Policy Violation"}`,
        {
          reason: result.reason,
          ruleId: result.ruleId || undefined,
          kind: result.detail || "budget",
          traceId: result.traceId,
        },
      );
    }
    // State B apply must leave an auditable trail — stash applied
    // local_decision so /log emits REQUEST_REROUTED; rejects push observations.
    if (canReroute) {
      const applyStatus = _applyReroute(
        result, body ?? null, provider, servingUnverified, modelHint,
      );
      if (applyStatus === "applied" && result?.reroute) {
        _stashLocalDecision(
          session,
          "rerouted",
          result.ruleId || result.reroute.rule_id || null,
          true,
          {
            from: result.reroute.original
              ? { provider: result.reroute.original.provider, model: result.reroute.original.model }
              : { provider: provider || "", model: body && typeof body === "object" ? body.model : "" },
            to: { provider: result.reroute.provider, model: result.reroute.model },
          },
          result.reroute.rule_name,
        );
      }
    }
  },
);

/**
 * Resolves a nested object path like ["OpenAI", "Chat", "Completions", "prototype"]
 * to the actual object, starting from the module root.
 */
function _resolvePath(root: any, path: string[]): any {
  let obj = root;
  for (const key of path) {
    if (obj == null) return null;
    obj = obj[key];
  }
  return obj;
}

/**
 * instrumentModules accepts either the provider CLASS (the documented form)
 * or a module namespace / CJS root that carries the class as a named or
 * default export. Stainless SDK roots stopped re-exporting resource statics
 * (anthropic 0.50.1, openai 5, groq-sdk 1, together-ai 0.30), and true ESM
 * namespaces never carried them — only the class itself reliably holds
 * `.Messages` / `.Chat` / `.Beta` across versions, so _resolvePath's walk
 * (e.g. ["Anthropic", "Messages", "prototype"]) only succeeds from the class.
 * Feature-detect the class (never version-detect); on any failure return the
 * input verbatim so behaviour degrades to the pre-normalization wrap attempt
 * (never throws — golden rule).
 */
function _pickClassExport(mod: any, className: string): any {
  try {
    if (typeof mod === "function") return mod;          // class passed directly
    const named = mod?.[className];
    if (typeof named === "function") return named;      // namespace named export
    const dflt = mod?.default;
    if (typeof dflt === "function") return dflt;        // namespace default export
    const nested = dflt?.[className];
    if (typeof nested === "function") return nested;    // interop double-wrap
  } catch {
    /* hostile getter — fall through to verbatim */
  }
  return mod;
}

// npm module name -> the `instrumentModules` key that actually reaches it.
// Total over every `moduleName` in _TARGET_METHODS and in every internal
// _wrapMethod call site. Each VALUE must satisfy both halves of the contract:
// the autoInstrument() prep loop consumes it (it lowercases keys, so casing is
// free) AND the public `InstrumentModules` interface declares it. Suggesting a
// key that fails either half is worse than saying nothing — the customer pastes
// it, the prep loop ignores it, and they land in the enforced-but-unmetered
// state this diagnostic exists to prevent (that is exactly how a lowercase
// `openai` reached a fleet app). Add a row here whenever a provider is added.
const _INSTRUMENT_MODULE_KEYS: Record<string, string> = {
  "openai": "openAI",
  "@anthropic-ai/sdk": "anthropic",
  "@google/genai": "googleGenAI",
  "@aws-sdk/client-bedrock-runtime": "bedrock",
  "cohere-ai": "cohere",
  "@mistralai/mistralai": "mistral",
  "@huggingface/inference": "huggingFace",
  "voyageai": "voyageai",
  "together-ai": "together",
  "groq-sdk": "groq",
  "@openrouter/sdk": "openRouter",
  "@cerebras/cerebras_cloud_sdk": "cerebras",
};

/**
 * Wraps a specific method to inject a pre-flight check.
 */
// Diagnostic for silent wrapper-install failures. Two paths can fall
// through without warning: (1) _resolvePath returns null — the module is
// loaded but its shape doesn't match the registry's objectPath, almost
// always an ESM/tsx hazard where the SDK's require() resolves to a
// different module record than the user's import; (2) the method isn't on
// the resolved object — SDK upgrade renamed it. Both produce the same
// user-visible symptom (call runs unwrapped, no /log POST) and both have
// the same remedy in the ESM case: pass instrumentModules to tp.init.
// Gated on tp.logErrors so production stays silent by default.
function _warnInstallFailure(target: TargetMethod, reason: string): void {
  try {
    const tp = getClient();
    if (!tp?.logErrors) return;
    // DRIFT GUARD: never fabricate a key. `target.moduleName` is arbitrary —
    // the public protect() API forwards any string — and a future provider may
    // be wired up before its row lands above. The typeof check also stops an
    // inherited Object.prototype member ("constructor", "toString") from
    // masquerading as a mapping. Unmapped -> drop the key-specific sentence and
    // print the generic remedy: less specific, still correct.
    const mapped = _INSTRUMENT_MODULE_KEYS[target.moduleName];
    const moduleKey = typeof mapped === "string" ? mapped : undefined;
    // Only the keys _pickClassExport normalizes accept the provider-class form;
    // the rest wrap statics off the module namespace, so only that shape works.
    const classKeyed = ["openai", "@anthropic-ai/sdk", "@cerebras/cerebras_cloud_sdk", "together-ai", "groq-sdk"]
      .includes(target.moduleName);
    const valueHint = classKeyed ? "<provider class or imported module>" : "<imported module namespace>";
    const remedy = moduleKey
      ? `In ESM / tsx apps, pass instrumentModules: { ${moduleKey}: ${valueHint} } to tp.init() ` +
        `so the SDK patches the same module the rest of your code uses.`
      : `In ESM / tsx apps, pass the imported module to tp.init() via instrumentModules ` +
        `so the SDK patches the same module the rest of your code uses.`;
    console.warn(
      `[TokenPolice] Could not install wrapper for ${target.moduleName} ` +
      `${target.objectPath.join(".")}.${target.method} — ${reason}. ` + remedy,
    );
  } catch {
    // fail-safe — never let the diagnostic itself break the customer's app
  }
}

// Setup-path failure diagnostic. Used by the init-time install guards (the
// instrumentModules prep loop + the per-target wrap loop) so a malformed /
// partially-installed SDK is logged-and-skipped rather than thrown into the
// customer's init(). Gated on tp.logErrors — production stays silent by
// default — and self-wrapped so the diagnostic can never throw.
function _warnSetupFailure(context: string, err: unknown): void {
  try {
    const tp = getClient();
    if (!tp?.logErrors) return;
    console.warn(
      `[TokenPolice] instrumentation setup error (${context}) — skipped: ` +
      `${err instanceof Error ? err.message : String(err)}`,
    );
  } catch {
    // fail-safe — never let the diagnostic itself break the customer's app
  }
}

/**
 * Anthropic APIPromise-surface preservation (golden-rule fix).
 *
 * The real `Messages.prototype.create` returns the vendor's `APIPromise` — a
 * Promise subclass carrying `.withResponse()` / `.asResponse()` / `.parse()`.
 * Our replacement is an `async function`, which collapses that into a plain
 * Promise. From @anthropic-ai/sdk 0.35.0 onward (verified by binary search;
 * still present through 0.90.0) the vendor's own MessageStream — and
 * identically BetaMessageStream — does
 *   `await messages.create({ ...params, stream: true }, opts).withResponse()`
 * (lib/MessageStream.js:150-153), so with the wrapper installed
 * `client.messages.stream()` throws
 * "messages.create(...).withResponse is not a function" INTO CUSTOMER CODE.
 * On ≤0.34 (e.g. 0.30.1) the vendor never reads `.withResponse()`, so the
 * shim is installed and simply never called — a verified no-op there.
 *
 * This wraps the installed method in a NON-async passthrough that re-attaches
 * the three surfaces to the returned promise. Feature-detected, NEVER
 * version-detected (same idiom as `anthropicStreamBypass`): decorate ONLY a
 * thenable that LACKS `.withResponse` — a real APIPromise (or any future
 * vendor surface) is returned byte-for-byte untouched. `withResponse()`
 * resolves the vendor's exact three-key shape `{ data, response, request_id }`
 * (core/api-promise.js:54). The `response` value MUST be a synthetic empty
 * `Response` — NOT `null`. `null` looks safe because `_connected` is
 * null-safe (`response?.headers.get('request-id')`, MessageStream.js:169-175,
 * and the vendor itself calls `this._connected(null)` on its
 * fromReadableStream path), but `_connected` is not the only consumer:
 * `MessageStream.withResponse()` / `BetaMessageStream.withResponse()`
 * (lib/MessageStream.js:92-95) hard-throw
 * `Error: Could not resolve a 'Response' object` on a falsy response, and
 * `stream.request_id` silently becomes `undefined` — so `response: null`
 * turns a customer's `messages.stream(p).withResponse()` on an ALLOWED call
 * into a throw (a golden-rule violation we once shipped; do not reintroduce
 * it). Hence a synthetic `new Response(null, { status: 200 })`, created
 * LAZILY inside the async callbacks (never at decoration time), memoized so
 * `withResponse()` and `asResponse()` hand back the SAME instance, and
 * guarded so a runtime with no global `Response` (older Node, edge runtimes,
 * workerd) degrades to `null` instead of throwing. `request_id: null` is a
 * deliberate, accepted degradation — telemetry loss beats a throw; recovering
 * the REAL Response/request-id is a separate follow-up.
 *
 * ASSIGNMENT ORDER IS LOAD-BEARING — do not "simplify" it or the gate away.
 * A hostile / frozen / proxied promise can throw on property assignment, so
 * each surface gets its OWN try/catch and `withResponse` — the one method the
 * vendor actually calls — is installed FIRST: a partial failure must still
 * install the load-bearing method. Even the feature-detect reads sit inside a
 * try/catch (a throwing getter means merely READING `p.withResponse` — or
 * `p.then` — can throw). The helper must never originate a throw into
 * customer code. It is deliberately NOT `async`: it must hand back whatever
 * the wrapped impl returned (decorated in place), never a fresh Promise
 * wrapping it.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _preserveApiPromiseSurface(impl: any): any {
  return function (this: any, ...args: any[]): any {
    const p = impl.apply(this, args);
    try {
      if (typeof p?.then === "function" && typeof p.withResponse !== "function") {
        // Lazy, memoized synthetic Response shared by withResponse() AND
        // asResponse() (same instance per promise — a customer comparing the
        // two sees consistent objects). Built only when a callback actually
        // runs, never at decoration time; if the runtime has no usable global
        // `Response`, memoize `null` — the shim itself must never throw.
        let syntheticResponse: any = null;
        let syntheticResponseBuilt = false;
        const getSyntheticResponse = (): any => {
          if (!syntheticResponseBuilt) {
            syntheticResponseBuilt = true;
            try {
              const ResponseCtor: any = (globalThis as any).Response;
              if (typeof ResponseCtor === "function") {
                syntheticResponse = new ResponseCtor(null, { status: 200 });
              }
            } catch {
              // fail-open: no/hostile Response constructor — stay null
            }
          }
          return syntheticResponse;
        };
        try {
          p.withResponse = async () => ({
            data: await p,
            response: getSyntheticResponse(),
            request_id: null,
          });
        } catch {
          // fail-open: frozen/readonly/proxied promise — leave undecorated
        }
        try {
          p.asResponse = async () => {
            await p;
            return getSyntheticResponse();
          };
        } catch {
          // fail-open
        }
        try {
          p.parse = () => p;
        } catch {
          // fail-open
        }
      }
    } catch {
      // fail-open: the feature-detect reads themselves can throw (hostile
      // getters on `then` / `withResponse`) — return p undecorated
    }
    return p;
  };
}

function _wrapMethod(target: TargetMethod, overrideModule?: any): void {
  let mod: any;
  if (overrideModule) {
    mod = overrideModule;
  } else {
    // App-first resolution (see resolveProviderModule) — never a bare
    // require(), which resolves the wrong copy on file:/pnpm topologies and
    // throws outright in the ESM dist.
    mod = resolveProviderModule(target.moduleName);
    if (mod === undefined) {
      return; // SDK not installed, skip — silent on purpose (optional peer dep)
    }
  }

  const obj = _resolvePath(mod, target.objectPath);
  if (!obj) {
    _warnInstallFailure(target, "module loaded but objectPath did not resolve");
    return;
  }

  let original = obj[target.method];
  if (!original || typeof original !== "function") {
    _warnInstallFailure(target, `method "${target.method}" not found on resolved class`);
    return;
  }

  // Modality targets (image / audio / video) have their own _logModality
  // call and don't want a parallel span from a third-party instrumentor
  // (e.g. @traceloop/instrumentation-openai also wraps Images.generate, which
  // would otherwise produce a duplicate `openai_chat`-shaped row alongside
  // our `openai_images` row). Walk the shimmer-style `__original` chain
  // back to the real method so our wrapper invokes it directly, then
  // assign that real method onto the prototype.
  if (target.modality) {
    let unwrapped = 0;
    while (
      typeof (original as any).__original === "function" &&
      (original as any).__original !== original
    ) {
      original = (original as any).__original;
      unwrapped++;
      if (unwrapped > 10) break; // safety net
    }
    if (unwrapped > 0) {
      obj[target.method] = original;
    }
  }

  const key = `${target.moduleName}:${target.objectPath.join(".")}:${target.method}`;
  if (_originals.has(key)) return; // Already instrumented

  _originals.set(key, original);
  // Capture the restore at the patch site: every branch below assigns
  // `obj[target.method]`, and `obj` is the exact (possibly probe-derived,
  // not-root-exported) object we're about to patch. uninstrument() cannot
  // re-resolve it from the string key for prototypes reached via a live
  // instance, so remember how to put it back directly.
  _restoreThunks.push(() => {
    obj[target.method] = original;
  });

  // Count the install for the zero-wrap audit (getUnwrappedResolvableProviders
  // / the init()-time warning). Every branch below assigns the wrapper, so
  // counting here is accurate for all target flavors.
  _wrappedTargetCounts.set(
    target.moduleName,
    (_wrappedTargetCounts.get(target.moduleName) ?? 0) + 1,
  );

  // Detect provider from module name (allow per-target override for SDKs whose
  // moduleName collides with another shape, e.g. OpenAI Responses).
  const provider = target.provider ?? _detectProvider(target.moduleName);

  // Anthropic streaming compat probe. @traceloop/instrumentation-anthropic
  // (≤0.27, the latest) declares support for @anthropic-ai/sdk >=0.9.1 but its
  // streaming branch constructs `new moduleExports.APIPromise(client, …)` — an
  // export (and constructor signature) that only exists in NEWER anthropic
  // SDKs. On older SDKs (e.g. 0.30.x) every streamed call through the patched
  // `create` — including `messages.stream()`, which awaits
  // `create({stream:true})` internally — throws
  // "APIPromise is not a constructor" INTO CUSTOMER CODE. When the root export
  // is missing, the wrapper below bypasses the Traceloop layer for streaming
  // calls and logs manually instead. New SDKs (export present) keep the
  // working Traceloop path byte-for-byte.
  const anthropicStreamBypass =
    provider === "anthropic" &&
    target.method === "create" &&
    typeof (
      mod?.APIPromise ??
      mod?.default?.APIPromise ??
      mod?.Anthropic?.APIPromise
    ) !== "function";

  if (target.manualTelemetry) {
    // SDKs with no usable OpenLLMetry instrumentor (e.g. @google/genai, and
    // AWS Bedrock's Converse API): the wrapper does the pre-flight check AND
    // extracts/logs token usage itself, since the telemetry SpanProcessor
    // never observes (or never gets usage from) these calls.
    obj[target.method] = _withCallObsScope(async function (
      this: any,
      ...args: any[]
    ): Promise<any> {
      // Inside a LangChain-instrumented call → pass through. The LangChain
      // wrapper already ran the single pre-flight check + composition.
      const _s = getCurrentSession();
      if (_s.inLangchain || _s.inLlamaIndex) {
        return await original.apply(this, args);
      }
      // OpenAI Agents JS runs tools in its own (non-OTel) tracing — register
      // our tool-span processor once the SDK is in use (cheap idempotent guard).
      maybeRegisterOpenAIAgentsTracing();
      // AWS Bedrock: `send` is generic across every command. Only Converse and
      // ConverseStream carry an LLM payload — BOTH now fall through the shared
      // manual path below (body-carrying pre-flight check + composition +
      // manual /log). ConverseStream's streamed token usage is harvested by a
      // dedicated stream tap (`_wrapBedrockConverseStream`) at the streaming-
      // detection point, because its top-level output is `{ stream, $metadata }`
      // (not itself async-iterable) with usage on the terminal
      // `{ metadata: { usage } }` event nested inside `.stream`. InvokeModel for
      // embedding-model-prefixed modelId gets a dedicated manual handler
      // (response body needs decoding to extract usage).
      const isBedrock =
        target.moduleName === "@aws-sdk/client-bedrock-runtime";
      if (isBedrock) {
        const cmdName = args[0]?.constructor?.name ?? "";
        // Route embedding InvokeModel calls through the dedicated handler.
        if (_isBedrockEmbeddingInvoke(args[0])) {
          return await _handleBedrockEmbeddingInvoke(this, original, args);
        }
        if (cmdName !== "ConverseCommand" && cmdName !== "ConverseStreamCommand") {
          return await original.apply(this, args);
        }
      }

      // For Bedrock the request params live on the command's `.input`; the
      // composition + usage helpers expect the request object as args[0].
      const kwargsArgs = isBedrock ? [args[0]?.input ?? {}] : args;

      // 1. Pre-flight check — pass body so REROUTE rules can swap model.
      const reqBody =
        kwargsArgs[0] && typeof kwargsArgs[0] === "object"
          ? (kwargsArgs[0] as Record<string, any>)
          : null;
      // Static registry modality (image/tts endpoints) OR Gemini TTS request
      // evidence on generateContent (same method as chat — no registry pin).
      let intent = target.modality
        ? _buildIntent(provider, target.modality, kwargsArgs)
        : null;
      if (
        !intent &&
        (provider === "google" || provider === "gemini") &&
        _wantsGoogleAudioOut(kwargsArgs)
      ) {
        intent = { kind: "audio_tts" };
      }
      // Two request shapes reaching this generic site carry no top-level
      // `model`, so the pre-flight ran with an empty model — model/provider
      // BLOCK rules never matched and budget group-bys bucketed to "unknown".
      // Pass the model as a matching-only HINT (never merged into reqBody, so
      // REROUTE never applies on these paths; the enforce-mode refusal is
      // recorded as REROUTE_REJECTED `unappliable_call_shape`).
      let modelHint: string | undefined;
      try {
        if (!(reqBody && typeof reqBody.model === "string" && reqBody.model)) {
          // Bedrock: ConverseCommand.input carries `modelId`.
          const byId = reqBody?.modelId;
          // Native OpenRouter SDK (Speakeasy envelope): `{ chatRequest: {...} }`.
          const chatReq = reqBody?.chatRequest;
          const byEnvelope =
            chatReq && typeof chatReq === "object" ? chatReq.model : undefined;
          if (typeof byId === "string" && byId) modelHint = byId;
          else if (typeof byEnvelope === "string" && byEnvelope) {
            modelHint = byEnvelope;
          }
        }
      } catch {
        // fail-open: hostile getter → no hint, same as before
      }
      await _runAsyncCheck(reqBody, provider, intent, true, false, modelHint);

      // 2. Reserve this call's span order + name up front so prompt/response
      // composition and the logged span all share one consistent key.
      let order = 0;
      let spanName: string | null = null;
      try {
        order = getCurrentSession().nextSpanOrder();
        spanName = consumePendingSpanName();
      } catch {
        // fail-safe
      }
      const startTime = new Date();

      // 3. Capture prompt composition (fail-safe). Operation hint routes
      // embedding calls through the input-only parser.
      _captureCompositionAt(provider, kwargsArgs, undefined, order,
                            undefined, target.operation);

      // Request-config service tier (Google GenAI takes `service_tier` on the
      // request config and never echoes it in usageMetadata, so the response-side
      // read in _logManual is inert for Gemini). Stash it as a fallback the
      // response tier still overrides at log time. Google-only, fail-open.
      _stashRequestServiceTier(provider, kwargsArgs, getCurrentSession(), order);

      // Stash attempt context so failures land with real model/provider/operation.
      // See _resolveFailOperation.
      // `target.shape` is the exact override the success path forwards to
      // _logManual/_logModality below — pass it so a failed row carries the
      // same usage_shape. `modelHint` rides along so failures on the
      // modelId/envelope shapes carry the same model the pre-flight matched on.
      _stashAttemptContext(
        getCurrentSession(),
        provider,
        kwargsArgs,
        _resolveFailOperation(target.operation, target.modality, intent),
        target.shape,
        undefined,
        modelHint,
      );

      // 4. Call original. Routed through the include_usage injector: for the
      // together chat-stream path (G5-1) the injection makes the final usage
      // chunk deterministic instead of replica-dependent; every other manual
      // provider gets a null handle inside and runs `original` unchanged.
      // Strip-and-retry + body restore live in _callWithInjectedStreamUsage.
      const _callStart = Date.now();
      // Monotonic anchor for latency (TTFT/total) — immune to wall-clock steps.
      const _callStartMono = performance.now();
      let result: any;
      let _usageInjected = false;
      try {
        const _r = await _callWithInjectedStreamUsage(
          original,
          this,
          args,
          provider,
        );
        result = _r.result;
        _usageInjected = _r.injected === true;
      } catch (err) {
        const elapsedMs = Date.now() - _callStart;
        const session = getCurrentSession();
        try {
          (session as any)._call_outcome = buildCallOutcome(err, elapsedMs);
        } catch {
          // fail-safe
        }
        _emitCallFailureLog(getClient(), session);
        throw err;
      }

      // 5a. Streaming: wrap the async iterable, log on completion.
      // `target.streaming` is a static hint (e.g. @google/genai has a
      // dedicated stream method); the runtime check also covers SDKs
      // where one method serves both modes (e.g. OpenRouter's `send`).
      // Modality calls (image/audio/video) never stream — they bypass
      // this branch via the `if (target.modality)` short-circuit below.
      if (target.modality && target.shape) {
        // Best-effort: when the caller opted into stream_format=sse on a
        // TTS-capable model, intercept the SSE stream to harvest usage and
        // hand back a normal binary Response. No-op for every other case.
        result = await _maybeTapOpenAITtsSse(provider, target, kwargsArgs, result);
        // Pass shape as authoritative override so binary bodies (TTS audio,
        // image bytes) never get read as text via `.text` / `text()`.
        _captureCompositionAt(provider, kwargsArgs, result, order, target.shape);
        const elapsedSeconds = Math.max(0, (Date.now() - _callStart) / 1000);
        _logModality(
          provider,
          target.modality,
          target.shape,
          kwargsArgs,
          result,
          order,
          spanName,
          startTime,
          elapsedSeconds,
        );
        return result;
      }
      // Bedrock ConverseStream: the SDK returns `{ stream, $metadata }` — the
      // top-level object is NOT async-iterable, so the generic `isStreaming`
      // check below is false and the terminal usage on the nested
      // `{ metadata: { usage } }` event would otherwise never be logged. Route
      // it to a dedicated tap that reattaches a lazy usage-harvesting generator
      // to `result.stream` (preserving object identity + `$metadata`) and logs
      // once on drain — the SAME `_logManual` the non-streaming Converse path
      // uses. Gated tight on `provider === "bedrock"` + a present iterable
      // `.stream`, so no other provider's streaming path is affected. Placed
      // BEFORE `isStreaming` and the 5b non-streaming `_logManual` so a Bedrock
      // stream logs exactly once, only via the wrapper's `finally`.
      if (
        provider === "bedrock" &&
        result?.stream &&
        typeof result.stream[Symbol.asyncIterator] === "function"
      ) {
        return _wrapBedrockConverseStream(
          result,
          provider,
          kwargsArgs,
          order,
          spanName,
          startTime,
          _callStartMono,
        );
      }
      const isStreaming =
        target.streaming === true ||
        (result != null &&
          typeof result[Symbol.asyncIterator] === "function");
      if (isStreaming) {
        return _wrapManualStream(
          result,
          provider,
          kwargsArgs,
          order,
          spanName,
          startTime,
          _callStartMono,
          _usageInjected,
        );
      }

      // 5b. Non-streaming: capture response composition + log immediately.
      // Embedding responses are vectors — pass the shape so
      // buildResponseComposition returns [] via _EMBEDDING_SHAPES.
      const respShape = target.operation === "embedding" ? target.shape : undefined;
      _captureCompositionAt(provider, kwargsArgs, result, order, respShape, target.operation);
      // Gemini TTS / native-audio: request-side modalities OR response AUDIO
      // details (snake + camel) reclass to audio_tts. Text-only unchanged.
      const logOperation = _resolveGoogleLogOperation(
        provider,
        target.operation || "chat",
        kwargsArgs,
        result,
      );
      _logManual(provider, kwargsArgs, result, order, spanName, startTime,
                 logOperation, target.shape ?? null);
      return result;
    });

    // Anthropic APIPromise-surface hook, manualTelemetry copy: this branch
    // returns HERE, before the shared hook at the bottom of _wrapMethod, so
    // without this line an anthropic manualTelemetry target (reachable via the
    // public escape hatch `protect(..., { manual: true, provider:
    // "anthropic" })`) would collapse the APIPromise and reintroduce
    // "p.withResponse is not a function". See _preserveApiPromiseSurface.
    if (provider === "anthropic") {
      obj[target.method] = _preserveApiPromiseSurface(obj[target.method]);
    }
    return;
  }

  if (target.isAsync) {
    obj[target.method] = _withCallObsScope(async function (
      this: any,
      ...args: any[]
    ): Promise<any> {
      // Inside a LangChain-instrumented call → pass through. The LangChain
      // wrapper already ran the single pre-flight check + composition.
      const _s = getCurrentSession();
      if (_s.inLangchain || _s.inLlamaIndex) {
        const passResult = await original.apply(this, args);
        // LlamaIndex + Anthropic streaming: @llamaindex/anthropic forwards only
        // content deltas (no usage events), and this pass-through skips the
        // anthropic stream bypass — so the LlamaIndex wrapper would log 0/0
        // tokens. Tap the inner stream (usage-stash only, no logging) so
        // _logLlamaIndex can fall back to the real counts. Fail-open: on any
        // tap error the original stream is returned untouched.
        if (
          _s.inLlamaIndex &&
          provider === "anthropic" &&
          (args[0] as any)?.stream === true &&
          passResult != null &&
          typeof passResult[Symbol.asyncIterator] === "function"
        ) {
          return _tapLlamaIndexAnthropicUsage(passResult, _s);
        }
        return passResult;
      }

      // Serving provider from the bound client's base URL (e.g. OpenAI SDK →
      // openrouter.ai reports "openrouter"; Anthropic SDK → api.minimax.io
      // reports "minimax"). Unrecognized custom hosts keep the module provider
      // for match/groupBy and set servingUnverified so REROUTE still refuses.
      // Wire/parse key stays on the module client (OpenAI-shaped bytes
      // stay OpenAI-parsed even when serving remaps to minimax/xai/…).
      const serving = _resolveServingProvider(provider, this);
      const effProvider = serving.provider;
      const wireKey = _wireParseKey(provider);

      // 1. Pre-flight check (@failSafe handles swallowing random errors).
      // Pass the request body so a REROUTE rule can mutate body.model
      // before the original LLM call is dispatched.
      const reqBody =
        args[0] && typeof args[0] === "object"
          ? (args[0] as Record<string, any>)
          : null;
      // Single-flight: when this create({stream:true}) was internally delegated
      // from a patched `.stream()` call, that wrapper already kicked off THE
      // pre-flight for this same logical request — share it instead of issuing
      // a second /check (which double-counted REROUTE_DIRECTIVE_ISSUED on the
      // customer's audit surface and doubled hot-path latency). The gate is
      // deliberately narrow: an inherited latch can only exist inside a
      // `.stream()` construction window (per-call ALS scope), `stream: true` is
      // what the vendor delegation always passes, and a present `checkPromise`
      // proves a real check is in flight. Every degraded path (ALS stub,
      // kickoff failure, no delegation) falls through to the layer-local check
      // below — degradation is "two checks", never "zero checks".
      let _sharedPreflight: StreamLatch | null = null;
      if (provider === "anthropic" && reqBody && (args[0] as any)?.stream === true) {
        try {
          const _inh = _anthropicStreamLatchStorage.getStore();
          if (_inh && _inh.checkPromise) _sharedPreflight = _inh;
        } catch {
          // fail-open → run own check
        }
      }
      if (_sharedPreflight) {
        try {
          await _sharedPreflight.checkPromise;
        } catch (e) {
          // GOLDEN RULE: only a typed, verified enforce block may propagate.
          // Re-thrown pre-dispatch it preserves today's flow exactly — the
          // vendor MessageStream catches it, and the consumer surfaces still
          // deliver the typed error via the wrapper's gate on the same promise.
          if (e instanceof TokenPoliceBlockedError) throw e;
          // any other rejection is check-machinery failure → fail-open (allow)
        }
        // Reroute sync: the shared check applied any REROUTE by mutating the
        // `.stream()` wrapper's body in place (checkedBody). That mutation
        // cannot reach the wire — the vendor shallow-copied params into this
        // layer's body BEFORE the check resolved — so replicate the ONLY
        // wire-visible effect onto this layer's body. All side effects
        // (observations, _tp_routing stash, local decision) already happened
        // exactly once inside the shared check — do NOT re-apply them.
        try {
          const _cb = _sharedPreflight.checkedBody;
          if (
            _cb && typeof _cb.model === "string" && _cb.model && reqBody &&
            typeof reqBody.model === "string" && reqBody.model !== _cb.model
          ) {
            reqBody.model = _cb.model;
          }
        } catch {
          // fail-open: reroute may not reach the wire; never throw
        }
      } else {
        await _runAsyncCheck(
          reqBody, effProvider, null, true, serving.servingUnverified,
        );
      }

      const _session = getCurrentSession();
      // Reserve the span order for THIS call up front — before the provider
      // call is dispatched. The instrumentor's onStart consumes this reservation
      // (via runWithReservedSpanOrder below) so the span it opens carries exactly
      // this order, and every pre/post-call composition stash for this call is
      // keyed by the same value. Because the order is fixed before the provider
      // call, two concurrent same-session calls can no longer cross-attribute
      // their stashes (see tests/concurrentAttribution.test.ts).
      const _spanOrder = _session.nextSpanOrder();

      // 2. Capture prompt composition (fail-safe, never blocks).
      // Wire key = module client (not serving): OpenAI SDK messages stay
      // OpenAI-parsed when baseURL remaps to xai/minimax.
      _capturePromptComposition(wireKey, args, _spanOrder);
      // Stash a provider override only for a recognized host remap.
      if (effProvider !== provider) {
        try {
          _stashProviderOverride(effProvider, _session, _spanOrder);
        } catch {
          // fail-safe
        }
        // Gateway-routed call (e.g. OpenAI SDK -> OpenRouter): stash the verbatim
        // vendor-prefixed model slug + its vendor head so telemetry onEnd restores
        // the full slug (Traceloop-JS strips the prefix) and forwards
        // model_extras.original_provider. Gated ON recognized host remap —
        // mirrors Python `_stash_gateway_request_model`. Internally fail-safe.
        _stashGatewayRequestModel(_session, reqBody, _spanOrder);
      }
      // Forward the serving endpoint so the server can identify the serving
      // provider (e.g. a native SDK pointed at api.minimax.io). Unconditional —
      // the direct-compatible-base_url case keeps `provider` unchanged, so this
      // must NOT be gated on the provider-override branch above.
      _stashApiBase(_extractBaseURL(this), _session, _spanOrder);

      // The streaming usage tap keys on the SAME reserved order so its drain-time
      // stash lands on the compKey telemetry.onEnd reads for this span.
      const _streamOrder = _spanOrder;

      // Stash attempt context so a failed call lands with real model/provider.
      // Also carry the module wire key — on a recognized host remap
      // (Anthropic SDK → api.minimax.io) `effProvider` is the serving vendor
      // but the bytes (and the success row's shape) stay anthropic_messages.
      _stashAttemptContext(_session, effProvider, args, undefined, undefined, wireKey);

      // Anthropic streaming bypass (see the probe comment in _wrapMethod): on
      // SDK versions where the Traceloop streaming wrapper would throw into
      // customer code, skip it — walk the shimmer `__original` chain back to
      // the true `create` and own telemetry manually. No instrumentor span will
      // start, so this path reuses the order already reserved above (the prompt
      // composition + api_base were stashed under it) and consumes the pending
      // span name here.
      let callee: any = original;
      let _bypassOrder = _spanOrder;
      let _bypassSpanName: string | null = null;
      const _isAnthropicStreamBypass =
        anthropicStreamBypass && (args[0] as any)?.stream === true;
      if (_isAnthropicStreamBypass) {
        try {
          const o: any = original;
          if (o && o.__wrapped === true && typeof o.__original === "function") {
            callee = o.__original;
          }
          _bypassSpanName = consumePendingSpanName();
        } catch {
          // fail-safe: fall back to the standard path
          callee = original;
        }
      }

      // 3. Call original (OpenLLMetry will handle telemetry internally here).
      // A6: chat-completions streams opened without include_usage get it
      // injected (with a strip-and-retry-once net for strict compat
      // servers) so streamed spend isn't silently lost.
      const _callStart = Date.now();
      // Monotonic anchor for stream latency (TTFT/total) — immune to clock steps.
      const _callStartMono = performance.now();
      const _bypassStart = new Date();
      let result: any;
      let _usageInjected = false;
      try {
        // Run the provider call inside the span-order reservation so the
        // instrumentor's onStart opens its span with `_spanOrder` instead of
        // peeking the mutable counter. `.run()` scoping auto-unsets the
        // reservation when the call settles, so it can never leak onto a later
        // span. On the anthropic-stream-bypass path no instrumentor span starts,
        // so the reservation simply goes unconsumed.
        ({ result, injected: _usageInjected } = await runWithReservedSpanOrder(
          { order: _spanOrder, consumed: false },
          () => _callWithInjectedStreamUsage(callee, this, args, provider),
        ));
      } catch (err) {
        const elapsedMs = Date.now() - _callStart;
        try {
          (_session as any)._call_outcome = buildCallOutcome(err, elapsedMs);
        } catch {
          // fail-safe
        }
        const _failureDispatched = _emitCallFailureLog(getClient(), _session);
        // F-17-A: when this create was internally delegated from a patched
        // `.stream()` call, the vendor MessageStream catches this rejection and
        // re-delivers it to the consumer's iterator — whose wrapper catch would
        // emit a SECOND, degraded failure row. Mark the shared latch so that
        // layer skips. Marked ONLY after a REAL dispatch (a no-op emit must not
        // suppress the consumer's own emission). This catch is shared by ALL
        // providers: getStore() is undefined for any non-delegated call, making
        // the mark a guaranteed no-op there.
        if (_failureDispatched) {
          try {
            _streamLatchMarkFailure(_anthropicStreamLatchStorage.getStore());
          } catch {
            // fail-safe
          }
        }
        throw err;
      }
      try {
        (_session as any)._call_outcome = buildCallOutcome(null, Date.now() - _callStart);
      } catch {
        // fail-safe
      }

      // Anthropic streaming bypass: the Traceloop layer was skipped, so no OTel
      // span exists for this call — tap the genuine Stream (non-destructively)
      // and log manually when it drains. The composition/service-tier/onEnd-tap
      // steps below are span-path concerns and don't apply here.
      if (_isAnthropicStreamBypass) {
        return _tapAnthropicStreamBypass(
          result, this, args, _bypassOrder, _bypassSpanName, _bypassStart, _session,
          _callStartMono,
        );
      }

      // 4. Capture response composition (fail-safe). Streams are skipped:
      // buildResponseComposition on a raw Stream object would Tier-3 to
      // String(stream) = "[object Object]" — the stream tap below stashes
      // the real accumulated composition instead.
      // Wire key (module), not serving.
      if (result == null || typeof result[Symbol.asyncIterator] !== "function") {
        _captureResponseComposition(wireKey, result, _spanOrder);
        // Non-streaming openai-wire verbatim usage: the Traceloop OpenAI
        // instrumentor emits only bare prompt/completion/total token attrs,
        // silently dropping prompt_tokens_details / completion_tokens_details
        // (cached, reasoning). Stash the response's verbatim usage object so
        // telemetry onEnd can forward it as usage.raw. Streams are covered by
        // the tap below; anthropic-wire is excluded inside (wireKey gate).
        _stashNonStreamVerbatimUsage(wireKey, result, _session, _spanOrder);
      }
      // Provider-reported service tier (OpenAI response top level / Anthropic
      // usage object) → forwarded as usage.tier so batch/flex/priority pricing
      // applies. Non-streaming only here; the stream tap stashes it per-chunk.
      _stashServiceTier(result, _session, _spanOrder);

      // 5. Streaming usage tap (fail-open, non-destructive). A Traceloop span
      // for an OpenAI-compatible streaming call carries no usage (its success
      // hook fires before the stream drains), so onEnd would drop the row.
      // Tap the stream to harvest usage from the include_usage final chunk and
      // stash it for onEnd. When the SDK injected include_usage, the tap also
      // strips the synthetic terminal chunk from the customer's iterator.
      // No-op for non-streaming results; span usage attrs still win when
      // present, so providers Traceloop handles are unchanged.
      // parse key MUST be the module wire key, never serving. Passing
      // remapped minimax/xai here caused silent total loss of streamed rows
      // (wrong chunk-usage branch / Vercel-AI parse on OpenAI chunks).
      if (result != null && typeof result[Symbol.asyncIterator] === "function") {
        result = _tapStreamUsageForOnEnd(
          wireKey, result, _session, args, _streamOrder, _usageInjected,
          _callStartMono,
        );
      }

      return result;
    });
  } else {
    obj[target.method] = function (this: any, ...args: any[]): any {
      // NOTE: The sync wrapper runs NO pre-flight `/check` and performs NO
      // enforcement. The pre-flight (`_runAsyncCheck`) is async and a sync
      // context cannot `await` it, so a synchronous LLM method cannot be
      // blocked here — this branch does composition-capture + attempt-context
      // + call-outcome + response-composition + failure logging ONLY. This
      // Node-only asymmetry is intentional (Python's sync path DOES enforce via
      // `_run_sync_check`). This branch is reachable only through the public
      // `protect(..., isAsync:false, ...)` escape hatch; every internal
      // auto-instrumentation target passes `isAsync:true` and takes the async
      // branch above (which does enforce).

      // Capture prompt composition
      _capturePromptComposition(provider, args);

      const _session = getCurrentSession();
      _stashAttemptContext(
        _session, provider, args, undefined, undefined, _wireParseKey(provider),
      );

      // Call original
      const _callStart = Date.now();
      let result: any;
      try {
        result = original.apply(this, args);
      } catch (err) {
        const elapsedMs = Date.now() - _callStart;
        try {
          (_session as any)._call_outcome = buildCallOutcome(err, elapsedMs);
        } catch {
          // fail-safe
        }
        _emitCallFailureLog(getClient(), _session);
        throw err;
      }
      try {
        (_session as any)._call_outcome = buildCallOutcome(null, Date.now() - _callStart);
      } catch {
        // fail-safe
      }

      // Capture response composition
      _captureResponseComposition(provider, result);

      return result;
    };
  }

  // Anthropic: the async wrapper above collapses the vendor's APIPromise into
  // a plain Promise, breaking `messages.stream()` on SDK ≥0.35.0 (see
  // _preserveApiPromiseSurface). Re-attach the surface here — this covers the
  // isAsync and sync branches; the manualTelemetry branch returns earlier and
  // carries its own copy of this hook (see above). Gated on `provider` ONLY —
  // deliberately NOT on `target.method === "create"` — so the Beta.Messages
  // target is covered automatically. Pure surface decoration: zero
  // control-flow change inside the wrapper itself.
  if (provider === "anthropic") {
    obj[target.method] = _preserveApiPromiseSurface(obj[target.method]);
  }
}

/**
 * Maps npm module name to provider string for composition parsing.
 */
function _detectProvider(moduleName: string): string {
  const m = moduleName.toLowerCase();
  if (m.includes("openrouter")) return "openrouter";
  if (m.includes("cerebras")) return "cerebras";
  if (m.includes("together")) return "together"; // together-ai — OpenAI-compatible
  if (m.includes("groq")) return "groq"; // native groq-sdk — OpenAI-compatible; stream usage under x_groq.usage
  if (m.includes("openai")) return "openai";
  if (m.includes("anthropic")) return "anthropic";
  if (m.includes("google") || m.includes("generative")) return "google";
  if (m.includes("cohere")) return "cohere"; // Cohere v2 — OpenAI-compatible messages
  if (m.includes("huggingface")) return "huggingface"; // HuggingFace — OpenAI-compatible
  if (m.includes("mistral")) return "mistral"; // Mistral — OpenAI-compatible (wrapped stream chunks)
  if (m.includes("xai") || m.includes("x.ai") || m.includes("ai-sdk/xai")) return "xai"; // @ai-sdk/xai (Vercel AI SDK LanguageModelV2)
  if (m.includes("bedrock")) return "bedrock";
  if (m.includes("langchain")) return "langchain";
  return "";
}

// ── Serving-provider identification (host → provider) ────────────────────────
// Mirrors the server-side host→provider identity tables (HOST_EXACT / HOST_PATTERNS)
// (kept inline — SDKs do not read the shared test-only JSON). Used so the
// REROUTE cross-provider guard compares the *serving* host, not just the
// client SDK module.
const _HOST_EXACT: Record<string, string> = {
  "api.openai.com": "openai",
  "api.anthropic.com": "anthropic",
  "generativelanguage.googleapis.com": "google",
  "api.mistral.ai": "mistral",
  "codestral.mistral.ai": "mistral",
  "api.x.ai": "xai",
  "api.deepseek.com": "deepseek",
  "api.moonshot.ai": "moonshot",
  "api.moonshot.cn": "moonshot",
  "api.minimax.io": "minimax",
  "api.minimaxi.com": "minimax",
  "api.perplexity.ai": "perplexity",
  "api.cohere.com": "cohere",
  "api.cohere.ai": "cohere",
  "open.bigmodel.cn": "zhipu",
  "api.z.ai": "zhipu",
  "ai-gateway.vercel.sh": "vercel-gateway",
  "openrouter.ai": "openrouter",
  "api.together.xyz": "together",
  "api.together.ai": "together",
  "api.fireworks.ai": "fireworks",
  "api.deepinfra.com": "deepinfra",
  "api.novita.ai": "novita",
  "api.groq.com": "groq",
  "api.cerebras.ai": "cerebras",
  "api.studio.nebius.com": "nebius",
  "api.studio.nebius.ai": "nebius",
};

const _HOST_PATTERNS: Array<{ re: RegExp; provider: string }> = [
  { re: /\.openai\.azure\.com$/i, provider: "azure-openai" },
  { re: /\.services\.ai\.azure\.com$/i, provider: "azure-ai" },
  { re: /\.inference\.ai\.azure\.com$/i, provider: "azure-ai" },
  { re: /^bedrock-runtime\..*\.amazonaws\.com$/i, provider: "bedrock" },
  { re: /aiplatform\.googleapis\.com$/i, provider: "vertex-ai" },
  { re: /^(localhost|127\.0\.0\.1|0\.0\.0\.0)$/i, provider: "self_hosted" },
];

/**
 * Extract a lowercase hostname from a base URL. Fail-safe — returns "" on error.
 * Mirrors server extractHost (no credentials leave this helper either).
 */
function _extractHost(apiBase: string): string {
  if (!apiBase || typeof apiBase !== "string") return "";
  try {
    const withScheme = apiBase.includes("://") ? apiBase : `http://${apiBase}`;
    return new URL(withScheme).hostname.toLowerCase();
  } catch {
    try {
      let h = apiBase.replace(/^[a-z]+:\/\//i, "");
      h = h.split("/")[0].split("?")[0].split("@").pop() || "";
      return h.split(":")[0].toLowerCase();
    } catch {
      return "";
    }
  }
}

/**
 * Map a hostname to a canonical serving-provider slug, or null if unknown.
 */
function _matchHostToProvider(host: string): string | null {
  if (!host) return null;
  const exact = _HOST_EXACT[host];
  if (exact) return exact;
  for (const p of _HOST_PATTERNS) {
    if (p.re.test(host)) return p.provider;
  }
  return null;
}

/**
 * Resolve serving provider from a raw base-URL string.
 *
 * - absent/empty → { kind: "absent" } (keep module-derived provider)
 * - recognized host → { kind: "recognized", provider }
 * - non-empty base whose host is not in the map → { kind: "unrecognized" }
 * (REROUTE must refuse — we cannot prove the target model is servable)
 */
export function _resolveServingFromBaseUrl(
  apiBase: string,
):
  | { kind: "absent" }
  | { kind: "recognized"; provider: string }
  | { kind: "unrecognized" } {
  try {
    if (!apiBase || typeof apiBase !== "string" || !apiBase.trim()) {
      return { kind: "absent" };
    }
    const host = _extractHost(apiBase);
    if (!host) return { kind: "unrecognized" };
    const provider = _matchHostToProvider(host);
    if (provider) return { kind: "recognized", provider };
    return { kind: "unrecognized" };
  } catch {
    return { kind: "absent" };
  }
}

/**
 * Resolve serving identity for pre-flight check + REROUTE.
 *
 * - recognized host → provider remapped (minimax, openrouter, …); REROUTE OK
 * - absent base_url → module provider; REROUTE OK
 * - unrecognized custom host → **module provider kept** for matchConditions /
 * groupBy /check payload (check/log mirror), but `servingUnverified: true`
 * so the REROUTE guard refuses (cannot prove target is servable there)
 *
 * Fail-safe: any error → module provider, verified.
 *
 * Serving provider and wire/parse key are **two axes**. This function
 * returns the billing/rules vendor only. Never pass its result into
 * `_chunkHasUsage` / `_extractUsage` / stream accumulators / composition
 * parsers — those must use `_wireParseKey(moduleProvider)` (the client
 * library's wire shape), or OpenAI-compatible gateways silently drop streamed
 * spend (minimax/xai host remap selecting the wrong chunk parser).
 */
function _resolveServingProvider(
  moduleProvider: string,
  thisArg: any,
): { provider: string; servingUnverified: boolean } {
  if (!thisArg) return { provider: moduleProvider, servingUnverified: false };
  try {
    const baseUrl = _extractBaseURL(thisArg);
    const raw =
      baseUrl ||
      String(thisArg?._client?.baseURL ?? thisArg?.baseURL ?? "");
    const resolved = _resolveServingFromBaseUrl(raw);
    if (resolved.kind === "recognized") {
      return { provider: resolved.provider, servingUnverified: false };
    }
    if (resolved.kind === "unrecognized") {
      return { provider: moduleProvider, servingUnverified: true };
    }
  } catch {
    // fail-safe
  }
  return { provider: moduleProvider, servingUnverified: false };
}

/**
 * Wire/parse key for chunk usage, stream accumulation, and composition.
 *
 * Equals the **module** (client-library) provider slug — never the host-remapped
 * serving vendor. An OpenAI SDK client always produces OpenAI-shaped bytes
 * regardless of `baseURL` (api.minimax.io, api.x.ai, …). Using the serving
 * slug here re-opens (silent total loss of streamed telemetry).
 */
function _wireParseKey(moduleProvider: string): string {
  return moduleProvider;
}

/**
 * Resolves the provider actually being billed / serving the call.
 * See `_resolveServingProvider` for unrecognized-host semantics (returns the
 * module provider — never an empty sentinel that would blank match/groupBy).
 */
function _effectiveProvider(provider: string, thisArg: any): string {
  return _resolveServingProvider(provider, thisArg).provider;
}

/**
 * Records a provider override for the next LLM span on this session.
 *
 * The telemetry SpanProcessor derives the provider from the OpenLLMetry
 * `gen_ai.system` attribute, which only knows the physical SDK (`openai`).
 * For OpenRouter-routed calls the enforcer stashes the real provider here,
 * keyed by span order, and `onEnd` applies it.
 */
function _stashProviderOverride(provider: string, session: any, order?: number): void {
  try {
    // The wrapper reserves the span order for this call up front and passes it
    // in, so the stash lands on the exact span onStart will open. When no order
    // is supplied, fall back to peeking spanCounter (the value onStart would
    // consume via nextSpanOrder) — the pre-fix behavior.
    const key = order ?? session.spanCounter;
    const compKey = `${session.traceId}:${key}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    session._pendingCompositions[compKey].provider = provider;
  } catch {
    // fail-safe
  }
}

/**
 * Gateway-routed SDK call (e.g. the OpenAI SDK pointed at OpenRouter): records
 * the FULL request model slug and its vendor head for the next LLM span on this
 * session.
 *
 * Traceloop-JS's openai instrumentor strips the vendor prefix from
 * `gen_ai.request.model` ("openai/gpt-4.1-nano" -> "gpt-4.1-nano") before
 * telemetry onEnd sees it, so the instrumented-path row lost the gateway
 * identity (model stripped, `original_provider` absent). Stash the customer's
 * verbatim slug (`model`) plus its vendor head (`original_provider` — the
 * segment before "/", "" when un-prefixed) on the same span-order key
 * `_stashProviderOverride` uses; onEnd forwards them. Only called on the
 * detected-gateway path — non-gateway spans never carry these keys. Mirrors
 * Python `_stash_gateway_request_model`. Fail-safe.
 */
function _stashGatewayRequestModel(session: any, reqBody: any, order?: number): void {
  try {
    const model =
      reqBody && typeof reqBody === "object" ? (reqBody as any).model : undefined;
    if (typeof model !== "string" || !model) return;
    // Same keying as _stashProviderOverride: use the reserved order when the
    // wrapper supplies it, else peek spanCounter (the pre-fix behavior).
    const key = order ?? session.spanCounter;
    const compKey = `${session.traceId}:${key}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    session._pendingCompositions[compKey].model = model;
    session._pendingCompositions[compKey].original_provider = model.includes("/")
      ? model.split("/")[0].trim().toLowerCase()
      : "";
  } catch {
    // fail-safe
  }
}

/**
 * Extracts the bound client's serving endpoint as `host + path` (no userinfo,
 * no query, no hash) so the server can map it to a serving provider (e.g.
 * `api.minimax.io` → minimax). Strips embedded credentials — only host metadata
 * leaves the customer process. Fail-safe: returns `""` on any error.
 */
export function _extractBaseURL(thisArg: any): string {
  try {
    return _sanitizeBaseURLString(thisArg?._client?.baseURL ?? thisArg?.baseURL);
  } catch {
    return "";
  }
}

/**
 * Sanitizes a raw base-URL string to `protocol//host+path` (no userinfo, no
 * query, no hash) — the same privacy contract as _extractBaseURL, factored out
 * so callers that already hold the raw string (AI SDK `model.config.baseURL`)
 * can reuse it. Fail-safe: returns `""` on any error.
 */
export function _sanitizeBaseURLString(raw: unknown): string {
  try {
    const s = String(raw ?? "");
    if (!s) return "";
    const u = new URL(s);
    // protocol + host (incl. port, EXCL. user:pass) + path — drops credentials,
    // query, and hash. Only host metadata leaves the process.
    return `${u.protocol}//${u.host}${u.pathname}`;
  } catch {
    return "";
  }
}

/**
 * Stashes the serving endpoint for the next LLM span on this session — read back
 * by the telemetry SpanProcessor onEnd and forwarded as `model_extras.api_base`.
 * Mirrors `_stashProviderOverride` (same span-order keying). Fail-safe.
 */
function _stashApiBase(baseURL: string, session: any, order?: number): void {
  try {
    if (!baseURL) return;
    const key = order ?? session.spanCounter;
    const compKey = `${session.traceId}:${key}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    session._pendingCompositions[compKey].api_base = baseURL;
  } catch {
    // fail-safe — api_base is advisory
  }
}

/**
 * Provider-reported service tier → canonical tier name, or "" when the value
 * is the provider's default tier (or unrecognized — never guess a discount).
 * OpenAI reports `service_tier` on the response/chunk top level ("default" |
 * "flex" | "priority" | legacy "scale"); Anthropic inside the usage object
 * ("standard" | "batch" | "priority_tier"). The canonical names match the
 * price table's tier_modifiers keys / when:{tier} rules.
 */
const _SERVICE_TIER_CANONICAL: Record<string, string> = {
  flex: "flex",
  priority: "priority",
  priority_tier: "priority",
  scale: "priority",
  batch: "batch",
};

export function _extractServiceTier(result: any): string {
  try {
    const raw =
      result?.service_tier ??           // OpenAI chat completion / stream chunk
      result?.response?.service_tier ?? // OpenAI Responses terminal event
      result?.usage?.service_tier ??    // Anthropic messages
      result?.usageMetadata?.serviceTier; // Google genai (camelCase)
    if (typeof raw !== "string" || !raw) return "";
    return _SERVICE_TIER_CANONICAL[raw.toLowerCase()] ?? "";
  } catch {
    return "";
  }
}

/**
 * Stashes the response-reported service tier for the span the instrumentor
 * just ended — read back by telemetry onEnd and forwarded as `usage.tier` so
 * tier-specific pricing (batch/flex/priority) applies. Post-call keying
 * (spanCounter - 1), mirroring _captureResponseComposition. Fail-safe.
 */
function _stashServiceTier(result: any, session: any, order?: number): void {
  try {
    const tier = _extractServiceTier(result);
    if (!tier) return;
    // Use the reserved order threaded from the wrapper. Falling back to
    // spanCounter-1 (the value onStart just consumed) is the pre-fix behavior,
    // which mis-attributes under concurrent same-session calls.
    const key = order ?? Math.max(0, session.spanCounter - 1);
    const compKey = `${session.traceId}:${key}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    session._pendingCompositions[compKey].service_tier = tier;
  } catch {
    // fail-safe — tier is advisory
  }
}

/**
 * True when a usage object carries a prompt/completion detail sub-object
 * (snake or camel) — the signal that stashing it adds information the bare
 * Traceloop token attrs cannot express.
 */
function _hasUsageDetailObjects(usage: any): boolean {
  try {
    const isObj = (v: any): boolean => v != null && typeof v === "object";
    return (
      isObj(usage.prompt_tokens_details) ||
      isObj(usage.completion_tokens_details) ||
      isObj(usage.promptTokensDetails) ||
      isObj(usage.completionTokensDetails)
    );
  } catch {
    return false;
  }
}

/**
 * Stashes the provider's VERBATIM usage object for a NON-streaming
 * openai-wire call — read back by telemetry onEnd and forwarded as usage.raw
 * so token details (reasoning_tokens, cached_tokens) the Traceloop OpenAI
 * instrumentor never emits as span attrs reach the collector. Gated to
 * detail-bearing usage only, so plain calls keep the constructed raw
 * bit-identical. Deliberately a DISTINCT field from the stream tap's
 * `usage.raw` (that one pairs with stashed positional numbers); telemetry
 * re-checks this clone against the attr-derived counts before use.
 * Post-call keying mirrors _stashServiceTier. Fail-open.
 */
function _stashNonStreamVerbatimUsage(
  wireKey: string, result: any, session: any, order?: number,
): void {
  try {
    // Anthropic Messages.create reaches the same seam — module wire gate.
    if (wireKey !== "openai") return;
    const usage = result?.usage;
    if (usage == null || typeof usage !== "object" || Array.isArray(usage)) return;
    if (!_hasUsageDetailObjects(usage)) return;
    const p = Number(usage.prompt_tokens);
    const c = Number(usage.completion_tokens);
    if (!((Number.isFinite(p) && p > 0) || (Number.isFinite(c) && c > 0))) return;
    // Deep clone so later mutation of the response object can't change what
    // gets logged; a circular/broken usage object fails open (no stash).
    const clone = JSON.parse(JSON.stringify(usage));
    const key = order ?? Math.max(0, session.spanCounter - 1);
    const compKey = `${session.traceId}:${key}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    session._pendingCompositions[compKey].nonStreamVerbatimRawUsage = clone;
  } catch {
    // fail-open: no stash
  }
}

/**
 * Captures a REQUEST-config service tier for providers that accept the tier on
 * the request but never echo it in the response. Google GenAI takes
 * `service_tier` on the GenerateContentConfig and its response usageMetadata
 * carries no serviceTier field — so the response-side _extractServiceTier is
 * inert for Gemini, and priority traffic would otherwise arrive tier-less and
 * bill at standard rates. Stashed under the call's compKey as a FALLBACK:
 * _logManual reads the response-echoed tier first and only falls back to this
 * stash, so a response tier (should a provider ever start echoing one) still
 * wins. Values are canonicalized through the same map as the response path, so
 * default/standard/auto normalize to "" and are dropped. Google-only.
 * Fail-open — a hostile/throwing config getter can never reach customer code.
 */
export function _stashRequestServiceTier(
  provider: string, args: any[], session: any, order?: number,
): void {
  try {
    if (provider !== "google" && provider !== "gemini") return;
    const params = (args && typeof args[0] === "object" && args[0] !== null)
      ? args[0] : null;
    const cfg = params && typeof params.config === "object" && params.config !== null
      ? params.config : null;
    if (!cfg) return;
    const raw = cfg.serviceTier ?? cfg.service_tier;
    if (raw == null) return;
    let s: string;
    try { s = String(raw).toLowerCase().trim(); } catch { return; }
    if (!s) return;
    const tier = _SERVICE_TIER_CANONICAL[s] ?? ""; // standard/default/auto → "" (dropped)
    if (!tier) return;
    const key = order ?? session.spanCounter;
    const compKey = `${session.traceId}:${key}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    session._pendingCompositions[compKey].service_tier = tier;
  } catch {
    // fail-safe — tier is advisory
  }
}

/**
 * Request-side Gemini TTS / native-audio intent. True when
 * config.responseModalities (snake or camel) includes AUDIO, or speechConfig
 * is present. Walks request body + nested config. Fail-open — never throws.
 */
export function _wantsGoogleAudioOut(args: any[]): boolean {
  try {
    const body =
      args && typeof args[0] === "object" && args[0] !== null
        ? (args[0] as Record<string, any>)
        : null;
    if (!body) return false;
    const config =
      body.config && typeof body.config === "object" ? body.config : null;
    const mods = config
      ? (config.responseModalities ?? config.response_modalities)
      : (body.responseModalities ?? body.response_modalities);
    const speech = config
      ? (config.speechConfig ?? config.speech_config)
      : (body.speechConfig ?? body.speech_config);
    if (speech) return true;
    if (mods == null) return false;
    if (typeof mods === "string") return mods.toUpperCase() === "AUDIO";
    if (!Array.isArray(mods)) {
      try {
        return String((mods as any)?.value ?? mods).toUpperCase() === "AUDIO";
      } catch {
        return false;
      }
    }
    for (const m of mods) {
      const val =
        typeof m === "string" ? m : String((m as any)?.value ?? m ?? "");
      if (val.toUpperCase() === "AUDIO") return true;
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Response-side evidence that Gemini returned audio output. Looks at
 * usageMetadata.candidatesTokensDetails (snake + camel) for AUDIO with
 * tokenCount > 0, and falls back to response parts whose inlineData MIME is
 * audio/*. Fail-open — never throws.
 */
export function _isGoogleAudioOutput(result: any): boolean {
  try {
    if (result == null) return false;
    const um = result.usageMetadata ?? result.usage_metadata;
    if (um) {
      const dets =
        um.candidatesTokensDetails ?? um.candidates_tokens_details;
      if (Array.isArray(dets)) {
        for (const d of dets) {
          try {
            const mod = String(d?.modality ?? "").toUpperCase();
            const tc = Number(d?.tokenCount ?? d?.token_count ?? 0);
            if (mod === "AUDIO" && tc > 0) return true;
          } catch {
            // continue
          }
        }
      }
    }
    const cands = result.candidates;
    if (!Array.isArray(cands)) return false;
    for (const cand of cands) {
      try {
        const parts = cand?.content?.parts;
        if (!Array.isArray(parts)) continue;
        for (const part of parts) {
          const inline = part?.inlineData ?? part?.inline_data;
          if (!inline) continue;
          const mime = String(
            inline.mimeType ?? inline.mime_type ?? "",
          ).toLowerCase();
          if (mime.includes("audio")) return true;
        }
      } catch {
        // continue
      }
    }
    return false;
  } catch {
    return false;
  }
}

/** Resolve log operation for Google generateContent: audio_tts or chat. */
function _resolveGoogleLogOperation(
  provider: string,
  baseOperation: string,
  args: any[],
  result: any,
): string {
  try {
    const op = baseOperation || "chat";
    if (op !== "chat") return op;
    const p = (provider || "").toLowerCase();
    if (p !== "google" && p !== "gemini") return op;
    if (_wantsGoogleAudioOut(args) || _isGoogleAudioOutput(result)) {
      return "audio_tts";
    }
  } catch {
    // fail-open
  }
  return baseOperation || "chat";
}

/**
 * Captures prompt composition from call args and stores on session.
 */
function _capturePromptComposition(provider: string, args: any[], order?: number): void {
  try {
    const session = getCurrentSession();
    // First arg for most SDKs is the options object containing messages
    const kwargs = (typeof args[0] === "object" && args[0] !== null) ? args[0] : {};
    const comp = buildPromptComposition(provider, kwargs);
    if (comp.length > 0) {
      const key = order ?? session.spanCounter;
      const compKey = `${session.traceId}:${key}`;
      if (!session._pendingCompositions[compKey]) {
        session._pendingCompositions[compKey] = {};
      }
      session._pendingCompositions[compKey].prompt = comp;
    }
  } catch {
    // fail-safe: composition is best-effort
  }
}

/**
 * Captures response composition and stores on session.
 */
function _captureResponseComposition(provider: string, response: any, order?: number): void {
  try {
    const session = getCurrentSession();
    const comp = buildResponseComposition(provider, response);
    if (comp.length > 0) {
      const key = order ?? Math.max(0, session.spanCounter - 1);
      const compKey = `${session.traceId}:${key}`;
      if (!session._pendingCompositions[compKey]) {
        session._pendingCompositions[compKey] = {};
      }
      session._pendingCompositions[compKey].response = comp;
    }
    // Stash the response's tool-call ids (ordered, id+name only) so the manual
    // toolSpan()/tool() path can auto-correlate them by name. REPLACE on every
    // capture (even with []) so a no-tool response clears stale ids from a prior
    // agent-loop iteration. Best-effort; independently fail-open.
    try {
      session.setPendingToolCalls(extractPendingToolCalls(provider, response));
    } catch {
      /* fail-open */
    }
  } catch {
    // fail-safe: composition is best-effort
  }
}

// ── Manual telemetry path ───────────────────────────────────────────────
// For SDKs with no OpenLLMetry instrumentor — currently @google/genai and the
// native OpenRouter SDK (@openrouter/sdk). The wrapper extracts token usage
// from the response itself and logs it directly via tp.log().

/**
 * Unwraps the request options object from a wrapped method's args.
 * Handles both flat requests (`{ messages, model }`) and Speakeasy-style
 * envelopes (`{ chatRequest: { messages, model } }`).
 */
function _requestKwargs(args: any[]): Record<string, any> {
  const raw = args && args[0];
  if (!raw || typeof raw !== "object") return {};
  if (raw.chatRequest && typeof raw.chatRequest === "object") {
    return raw.chatRequest;
  }
  return raw;
}

/**
 * Extracts normalized token usage + model name from a manual-telemetry
 * response (or streaming chunk). Provider-specific because the native SDKs
 * expose different usage shapes.
 */
export function _extractUsage(
  provider: string,
  obj: any,
  args: any[],
): {
  model: string;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
} {
  const kwargs = _requestKwargs(args);

  if (provider === "anthropic") {
    // Anthropic Messages usage (manual path — Traceloop-bypassed streaming).
    // Anthropic's input_tokens EXCLUDES cache reads/writes (separate buckets),
    // so no subtraction — mirrors how telemetry.onEnd reads the Traceloop
    // span's gen_ai.usage.* for the non-bypassed path. Cache-creation tokens
    // bill at a different rate and flow via usage.raw (anthropic_messages
    // shape), not the legacy 3-number fields.
    const u = (obj && obj.usage) || {};
    const inputTokens = Number(u.input_tokens ?? 0);
    const outputTokens = Number(u.output_tokens ?? 0);
    const cachedTokens = Number(u.cache_read_input_tokens ?? 0);
    const model = String(obj?.model ?? kwargs.model ?? "unknown");
    return { model, inputTokens, outputTokens, cachedTokens };
  }

  if (provider === "cohere") {
    // Cohere v2 — usage is on `response.usage` (non-stream) or on the
    // `message-end` stream event at `chunk.delta.usage`; Cohere v1 nests it on
    // `response.meta`. The cohere-ai TS SDK camelCases the wire format
    // (billedUnits.inputTokens); raw-HTTP/REST responses keep snake_case
    // (billed_units.input_tokens). Read snake_case first (raw REST via
    // protect()), fall back to camelCase (SDK objects) for both the container
    // and the fields — without the fallback, raw-REST calls land 0/0 tokens and
    // cost goes unmeasured. Prefer billed_units (what Cohere charges); fall back
    // to raw token counts. No cached-token concept in Cohere's usage object.
    const u = (obj && (obj.usage ?? obj.delta?.usage ?? obj.meta)) || {};
    const counts = u.billed_units ?? u.billedUnits ?? u.tokens ?? {};
    const inputTokens = Number(counts.input_tokens ?? counts.inputTokens ?? 0) || 0;
    const outputTokens = Number(counts.output_tokens ?? counts.outputTokens ?? 0) || 0;
    const model = String(kwargs.model ?? "unknown");
    return { model, inputTokens, outputTokens, cachedTokens: 0 };
  }

  if (provider === "bedrock") {
    // AWS Bedrock Converse response: { output, stopReason, usage: {
    // inputTokens, outputTokens, cacheReadInputTokens, cacheWriteInputTokens } }.
    // ConverseStream delivers the SAME usage object on the terminal stream
    // event, nested under `metadata.usage` — read that first, then fall back to
    // the non-stream top-level `.usage` (regression-safe for ConverseCommand).
    // `cacheWriteInputTokens` is intentionally NOT mapped to cachedTokens (it
    // bills at the cache-write rate). The model id is read straight from the
    // request — no instrumentor in the path to strip its vendor prefix.
    const u = (obj && (obj.metadata?.usage ?? obj.usage)) || {};
    const inputTokens = Number(u.inputTokens ?? 0);
    const outputTokens = Number(u.outputTokens ?? 0);
    const cachedTokens = Number(u.cacheReadInputTokens ?? 0);
    const model = String(kwargs.modelId ?? kwargs.model ?? "unknown");
    return { model, inputTokens, outputTokens, cachedTokens };
  }

  if (provider === "openrouter") {
    // OpenRouter (native SDK) — OpenAI-shaped usage. `promptTokens` already
    // includes the cached prompt tokens, so subtract them to avoid
    // double-counting cost. `completionTokens` already includes
    // `completionTokensDetails.reasoningTokens`, so they must not be added again.
    const u = (obj && obj.usage) || {};
    const promptTokens = Number(u.promptTokens ?? 0);
    const cachedTokens = Number(u.promptTokensDetails?.cachedTokens ?? 0);
    const inputTokens = Math.max(0, promptTokens - cachedTokens);
    const outputTokens = Number(u.completionTokens ?? 0);
    const model = String(obj?.model ?? kwargs.model ?? "unknown");
    return { model, inputTokens, outputTokens, cachedTokens };
  }

  if (provider === "mistral") {
    // Mistral — non-streaming responses are OpenAI-shaped at the top level
    // (`obj.usage` with snake_case fields). Streaming chunks are wrapped in a
    // `data` field (CompletionEvent.data == CompletionChunk), so the final
    // usage-bearing chunk has its usage at `obj.data.usage`. Handle both.
    const u = (obj && (obj.usage ?? obj.data?.usage)) || {};
    const promptTokens = Number(u.prompt_tokens ?? u.promptTokens ?? 0);
    const completionTokens = Number(u.completion_tokens ?? u.completionTokens ?? 0);
    const model = String(obj?.model ?? obj?.data?.model ?? kwargs.model ?? "unknown");
    return { model, inputTokens: promptTokens, outputTokens: completionTokens, cachedTokens: 0 };
  }

  if (_isAiSdkParse(provider)) {
    // Vercel AI SDK LanguageModel (any provider — @ai-sdk/*, gateway,
    // community). Two spec generations of the `usage` object:
    // V2 (ai v5, specificationVersion "v2") — FLAT numbers:
    // { inputTokens, outputTokens, totalTokens, reasoningTokens?,
    // cachedInputTokens? } (ai v4 used promptTokens/completionTokens).
    // V3 (ai v6, specificationVersion "v3") — NESTED detail objects:
    // { inputTokens: { total, noCache, cacheRead, cacheWrite },
    // outputTokens: { total, text, reasoning } }.
    // `Number(nestedObject)` is NaN, so the V3 shape MUST be branch-detected
    // (typeof === "object") before the flat read. doGenerate carries `usage`
    // on the result; doStream's terminal `finish` part has the same shape.
    // The model id is bound at provider construction time and surfaced as
    // `this.modelId`; the instrumenter stashes it onto args[0] via the
    // `__tpXaiModel` key so this branch can recover it.
    let usageSource: any = obj?.usage;
    if (!usageSource && obj?.type === "finish") usageSource = obj.usage;
    const u = usageSource || {};
    let inputT: number;
    let outputT: number;
    let cachedTokens: number;
    if (
      (u.inputTokens !== null && typeof u.inputTokens === "object") ||
      (u.outputTokens !== null && typeof u.outputTokens === "object")
    ) {
      // V3 nested. Legacy 3-number fields can't express cacheWrite — it stays
      // inside inputT; the server-side `vercel_ai` shape mapper bills it
      // precisely from usage.raw.
      const it =
        u.inputTokens !== null && typeof u.inputTokens === "object"
          ? u.inputTokens
          : { total: u.inputTokens };
      const ot =
        u.outputTokens !== null && typeof u.outputTokens === "object"
          ? u.outputTokens
          : { total: u.outputTokens };
      inputT = Number(it.total) || 0;
      outputT = Number(ot.total) || 0;
      cachedTokens = Number(it.cacheRead) || 0;
    } else {
      inputT = Number(u.inputTokens ?? u.promptTokens) || 0;
      outputT = Number(u.outputTokens ?? u.completionTokens) || 0;
      cachedTokens = Number(u.cachedInputTokens ?? u.cacheReadInputTokens) || 0;
    }
    // Anthropic-compatible servers that defer input_tokens to message_delta
    // (e.g. MiniMax sends input_tokens=0 in message_start with the real count
    // in message_delta): @ai-sdk/anthropic builds its finish usage from
    // message_start only, so inputTokens lands 0 — but it forwards the MERGED
    // raw usage (message_start ⊕ message_delta) on
    // providerMetadata.anthropic.usage. Gated: engages only when the spec
    // usage shows zero input.
    if (inputT === 0) {
      try {
        const rawAnthropic: any = (obj as any)?.providerMetadata?.anthropic?.usage;
        if (rawAnthropic && typeof rawAnthropic === "object") {
          const rawIn = Number(rawAnthropic.input_tokens) || 0;
          if (rawIn > 0) {
            inputT = rawIn;
            if (!cachedTokens) {
              cachedTokens = Number(rawAnthropic.cache_read_input_tokens) || 0;
            }
          }
        }
      } catch {
        // fail-safe: keep the spec-usage zeros
      }
    }
    const inputTokens = Math.max(0, inputT - cachedTokens);
    const model = String(
      (kwargs as any)?.__tpXaiModel ?? kwargs.modelId ?? kwargs.model ?? "unknown",
    );
    return { model, inputTokens, outputTokens: outputT, cachedTokens };
  }

  if (provider === "openai_responses") {
    // OpenAI Responses API. Non-streaming responses carry `usage` directly on
    // the Response object. Streaming delivers usage ONLY on the
    // `response.completed` event — unwrap once via `.response`. The Node SDK
    // (openai >= 5) uses snake_case at runtime — input_tokens /
    // output_tokens / input_tokens_details.cached_tokens /
    // output_tokens_details.reasoning_tokens — so we read snake_case first and
    // fall back to camelCase for forward-compatibility. input_tokens already
    // INCLUDES the cached portion, so subtract to avoid double-counting cost.
    // Reasoning tokens are already part of output_tokens.
    let resp = obj;
    if (resp && resp.type === "response.completed" && resp.response) {
      resp = resp.response;
    }
    const u = (resp && resp.usage) || {};
    const inputT = Number(u.input_tokens ?? u.inputTokens ?? 0);
    const outputT = Number(u.output_tokens ?? u.outputTokens ?? 0);
    const cachedTokens = Number(
      u.input_tokens_details?.cached_tokens ??
        u.inputTokensDetails?.cachedTokens ??
        0,
    );
    const inputTokens = Math.max(0, inputT - cachedTokens);
    const model = String(resp?.model ?? kwargs.model ?? "unknown");
    return { model, inputTokens, outputTokens: outputT, cachedTokens };
  }

  if (
    provider === "cerebras" ||
    provider === "huggingface" ||
    provider === "together" ||
    provider === "openai" ||
    provider === "litellm" ||
    provider === "groq"
  ) {
    // OpenAI-shaped snake_case usage. `prompt_tokens` already includes any
    // cached prompt tokens, so subtract them to avoid double-counting cost.
    // `completion_tokens` already includes
    // `completion_tokens_details.reasoning_tokens`, so they must not be added
    // again. Some providers don't
    // emit prompt_tokens_details / completion_tokens_details — those read 0.
    // ("openai" and "litellm" added for raw-HTTP wrappers registered via
    // tp.protect({manual:true, provider:"openai"|"litellm"}) — both producers
    // return OpenAI Chat Completions shape directly.)
    // Groq streaming final chunks nest usage under `chunk.x_groq.usage`
    // (top-level `chunk.usage` absent); `x_groq` is a groq-only field so this
    // fallback never affects the other OpenAI-compatible providers above.
    const u = (obj && (obj.usage || obj.x_groq?.usage)) || {};
    const promptTokens = Number(u.prompt_tokens ?? 0);
    const cachedTokens = Number(u.prompt_tokens_details?.cached_tokens ?? 0);
    const inputTokens = Math.max(0, promptTokens - cachedTokens);
    const outputTokens = Number(u.completion_tokens ?? 0);
    const model = String(obj?.model ?? kwargs.model ?? "unknown");
    return { model, inputTokens, outputTokens, cachedTokens };
  }

  // Default: @google/genai usageMetadata shape.
  // Gemini's `promptTokenCount` already includes cached tokens, so subtract
  // them to avoid double-counting cost. `toolUsePromptTokenCount` is separate
  // input-side usage (tool results fed back to the model) — count it as input.
  // "Thoughts" (thinking) tokens are billed at the output rate.
  const um = (obj && obj.usageMetadata) || {};
  const promptTokens = Number(um.promptTokenCount ?? 0);
  const cachedTokens = Number(um.cachedContentTokenCount ?? 0);
  const toolUseTokens = Number(um.toolUsePromptTokenCount ?? 0);
  const inputTokens = Math.max(0, promptTokens - cachedTokens) + toolUseTokens;
  const outputTokens =
    Number(um.candidatesTokenCount ?? 0) + Number(um.thoughtsTokenCount ?? 0);
  const model = String(obj?.modelVersion ?? kwargs.model ?? "unknown");
  return { model, inputTokens, outputTokens, cachedTokens };
}

/** True if a streaming chunk carries token usage for this provider. */
function _chunkHasUsage(provider: string, chunk: any): boolean {
  if (!chunk) return false;
  // OpenAI Responses API — usage arrives ONLY on the `response.completed`
  // event, nested under event.response.usage.
  if (provider === "openai_responses")
    return chunk?.type === "response.completed" && chunk?.response?.usage != null;
  // OpenRouter, Cerebras, HuggingFace, Together, OpenAI, LiteLLM —
  // all OpenAI-shaped with usage on the final chunk (when the request asks
  // for it via stream_options.include_usage).
  if (
    provider === "openrouter" ||
    provider === "cerebras" ||
    provider === "huggingface" ||
    provider === "together" ||
    provider === "openai" ||
    provider === "litellm"
  )
    return chunk.usage != null;
  // Groq — OpenAI-shaped, but the native groq-sdk delivers streaming usage on
  // the FINAL chunk under `chunk.x_groq.usage` (top-level `chunk.usage` is
  // absent on the chunk type). Accept either; `x_groq` is a groq-only field.
  if (provider === "groq")
    return chunk.usage != null || chunk?.x_groq?.usage != null;
  // Anthropic — our raw-HTTP streaming wrapper synthesises a terminal chunk
  // with `usage.input_tokens` + `usage.output_tokens`. Detect either field.
  if (provider === "anthropic")
    return (
      chunk.usage != null &&
      (chunk.usage.input_tokens != null || chunk.usage.output_tokens != null)
    );
  // Mistral wraps each stream chunk as `CompletionEvent = { data: CompletionChunk }`;
  // the usage-bearing final chunk has `chunk.data.usage`.
  if (provider === "mistral")
    return chunk?.data?.usage != null || chunk?.usage != null;
  // Cohere v2 — the `message-end` stream event carries usage at delta.usage.
  if (provider === "cohere")
    return chunk.delta?.usage != null || chunk.usage != null;
  // Vercel AI SDK — `doStream` emits LanguageModel V2/V3 parts; usage rides on
  // the terminal `finish` event part. (Some providers also emit `usage` on a
  // top-level field — accept either.)
  if (_isAiSdkParse(provider))
    return (chunk?.type === "finish" && chunk?.usage != null) || chunk?.usage != null;
  // AWS Bedrock ConverseStream — usage rides the terminal metadata event at
  // `chunk.metadata.usage` (inputTokens / outputTokens / cacheReadInputTokens).
  if (provider === "bedrock")
    return chunk?.metadata?.usage != null;
  return chunk.usageMetadata != null;
}

/**
 * Captures prompt OR response composition for a manual-telemetry call at an
 * explicit span order. When `result` is undefined the prompt is captured from
 * args[0]; otherwise the response is captured. Stored on the session keyed by
 * `traceId:order` so _logManual can attach it to the logged span.
 *
 * `usageShape` (response side only) is the authoritative server-side shape
 * enum — passed by modality wrappers so the composition builder can short-
 * circuit binary/image/video bodies into a non-text entry. Prevents the TTS
 * misclassification bug where a binary audio body's `.text`/`text()` accessor
 * was read as a multi-KB "assistant text" entry.
 */
function _captureCompositionAt(
  provider: string,
  args: any[],
  result: any,
  order: number,
  usageShape?: string,
  operation?: string,
  // When true, capture composition as usual but DO NOT touch the pending
  // tool-call stash. The ai_sdk streaming path stashes tool-call ids
  // incrementally (mid-stream, before each execute() fires) via
  // session.appendPendingToolCall; letting the end-of-stream finalize also
  // REPLACE the stash from the full accumulator would resurrect ids already
  // consumed by those executions. Defaults false → unchanged for every other
  // caller.
  skipToolCallStash?: boolean,
): void {
  try {
    const session = getCurrentSession();
    const compKey = `${session.traceId}:${order}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    if (result === undefined) {
      const comp = buildPromptComposition(provider, _requestKwargs(args), operation);
      if (comp.length > 0) session._pendingCompositions[compKey].prompt = comp;
    } else {
      const comp = buildResponseComposition(provider, result, usageShape);
      if (comp.length > 0) {
        session._pendingCompositions[compKey].response = comp;
      }
      // Stash tool-call ids (ordered, id+name only) so manual toolSpan()/tool()
      // can auto-correlate by name. REPLACE on every response capture (even [])
      // so a no-tool / embedding response clears stale ids from a prior agent
      // loop — matches Mode-A _captureResponseComposition + Python. Without this
      // AI-SDK extract (content[] type==="tool-call") was dead code. Prompt-only
      // captures (result === undefined) do not touch the stash. Fail-open.
      if (!skipToolCallStash) {
        try {
          session.setPendingToolCalls(extractPendingToolCalls(provider, result));
        } catch {
          /* fail-open */
        }
      }
    }
  } catch {
    // fail-safe: composition is best-effort
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Raw provider usage + shape resolution.
//
// The server owns the (shape, raw) → billable-units mapping. Here we only
// identify the provider's shape and forward the verbatim usage object the
// provider returned. Streaming wrappers stash the final-chunk usage
// onto the session's `_pending_usage[traceId:order]` slot, so the manual log
// path picks it up if present; otherwise we extract directly from the result.
// ────────────────────────────────────────────────────────────────────────────
function _resolveUsageShape(provider: string, args: any[]): string {
  switch ((provider || "").toLowerCase()) {
    case "anthropic":      return "anthropic_messages";
    case "openai":         return "openai_chat";
    case "openai_responses": return "openai_responses";
    case "google":
    case "gemini":         return "google_genai";
    case "bedrock":        return "bedrock_converse";
    case "cohere":         return "cohere_chat";
    case "mistral":        return "mistral_chat";
    case "xai":            return "xai_chat";
    // Generic Vercel AI SDK path — usage.raw is the spec V2/V3 usage object;
    // the server-side vercel_ai mapper handles both generations (incl. V3
    // nested cache read/write + reasoning detail).
    case "ai_sdk":         return "vercel_ai";
    case "together":       return "together_chat";
    case "cerebras":       return "cerebras_chat";
    case "groq":           return "groq_chat";
    case "huggingface":    return "huggingface_chat";
    case "openrouter":     return "openrouter_routed";
    case "litellm":        return "openai_compatible_chat";
    default:               return "openai_compatible_chat";
  }
}

// Bedrock model-id prefixes that identify an embedding call. Used by the
// InvokeModel filter to route embedding calls through the manual log path.
const BEDROCK_EMBEDDING_MODEL_PREFIXES = [
  "amazon.titan-embed",
  "cohere.embed",
  "voyage.voyage",
];

function _isBedrockEmbeddingInvoke(cmd: any): boolean {
  try {
    if (!cmd || typeof cmd !== "object") return false;
    const cmdName = cmd?.constructor?.name ?? "";
    if (cmdName !== "InvokeModelCommand" && cmdName !== "InvokeModelWithResponseStreamCommand") {
      return false;
    }
    const modelId = String(cmd?.input?.modelId ?? "");
    return BEDROCK_EMBEDDING_MODEL_PREFIXES.some((p) => modelId.startsWith(p));
  } catch {
    return false;
  }
}

/**
 * Map a parsed Bedrock InvokeModel response body → (inputTokens, rawUsage, shape).
 * Per-provider response shapes for Bedrock-hosted embedding models.
 */
function _extractBedrockEmbeddingUsage(
  modelId: string,
  parsed: any,
): { inputTokens: number; rawUsage: Record<string, unknown> | null; shape: string } {
  if (!parsed || typeof parsed !== "object") {
    return { inputTokens: 0, rawUsage: null, shape: "openai_embeddings" };
  }
  const mid = (modelId || "").toLowerCase();
  if (mid.startsWith("amazon.titan-embed")) {
    const tokens = Number(parsed.inputTextTokenCount ?? 0) || 0;
    // Forward Titan's native field verbatim under its own shape — the server
    // owns the (shape, raw) → billable-units mapping; the SDK only identifies
    // the shape.
    return {
      inputTokens: tokens,
      rawUsage: { inputTextTokenCount: tokens },
      shape: "bedrock_titan_embed",
    };
  }
  if (mid.startsWith("cohere.embed")) {
    const billed = parsed?.meta?.billed_units ?? {};
    const tokens = Number(billed.input_tokens ?? 0) || 0;
    const images = Number(billed.images ?? 0) || 0;
    return {
      inputTokens: tokens,
      rawUsage: { meta: { billed_units: { input_tokens: tokens, images } } },
      shape: "cohere_embed",
    };
  }
  if (mid.startsWith("voyage.voyage")) {
    const tokens = Number(parsed?.usage?.total_tokens ?? 0) || 0;
    return {
      inputTokens: tokens,
      rawUsage: { total_tokens: tokens },
      shape: "voyage_embed",
    };
  }
  return { inputTokens: 0, rawUsage: null, shape: "openai_embeddings" };
}

function _bedrockEmbeddingOriginalProvider(modelId: string): string {
  const mid = (modelId || "").toLowerCase();
  if (mid.startsWith("amazon.titan-embed")) return "amazon";
  if (mid.startsWith("cohere.embed")) return "cohere";
  if (mid.startsWith("voyage.voyage")) return "voyage";
  return "bedrock";
}

/**
 * Decode the Bedrock InvokeModel request body (Uint8Array | string) into a
 * synthetic kwargs object the embedding composition parser can read. Best-
 * effort — failure to decode is non-fatal.
 */
function _bedrockEmbeddingKwargs(cmdInput: any): Record<string, any> {
  const out: Record<string, any> = { model: cmdInput?.modelId };
  try {
    let body = cmdInput?.body;
    if (body instanceof Uint8Array) {
      body = new TextDecoder().decode(body);
    }
    if (typeof body === "string") {
      const parsed = JSON.parse(body);
      if (parsed && typeof parsed === "object") {
        for (const k of ["inputText", "texts", "input", "inputs"]) {
          if (k in parsed) out[k] = parsed[k];
        }
        if ("inputText" in parsed && !("input" in out)) {
          out.input = parsed.inputText;
        }
      }
    }
  } catch {
    // best-effort
  }
  return out;
}

async function _handleBedrockEmbeddingInvoke(
  this_: any,
  original: Function,
  args: any[],
): Promise<any> {
  // 1. Pre-flight check (embedding intent). A block decision surfaces as a
  // TokenPoliceBlockedError here, which must propagate to the caller — so
  // this call is intentionally NOT wrapped: any error (BlockedError or a
  // fail-open internal error _runAsyncCheck already swallows) is handled at
  // the source, and re-throwing it verbatim would be a no-op.
  await _runAsyncCheck(
    { model: args[0]?.input?.modelId },
    "bedrock",
    { kind: "embedding" },
    // The check body here is a throwaway { model } object; the real
    // InvokeModel call re-reads `args` unchanged, so a reroute swap could
    // never reach the provider. Suppress reroute (block/allow still enforce)
    // to avoid stashing a phantom _tp_routing / misreporting savings.
    false,
  );

  // 1b. Pre-call telemetry setup is best-effort. Session resolution or the
  // attempt-context stash throwing must degrade to an un-instrumented call
  // (response still returned, just un-logged) — never stop the customer's
  // embedding request. Only TokenPoliceBlockedError may escape this path.
  let session: any = null;
  let startTime = new Date();
  let spanName: string | null = null;
  let order = 0;
  let _callStart = Date.now();
  try {
    session = getCurrentSession();
    startTime = new Date();
    spanName = consumePendingSpanName();
    try {
      order = session.nextSpanOrder();
    } catch {
      // fail-safe
    }
    _callStart = Date.now();
    _stashAttemptContext(session, "bedrock", [{ model: args[0]?.input?.modelId }], "embedding");
  } catch {
    // fail-open: proceed with the call, un-instrumented.
  }

  let response: any;
  try {
    response = await original.apply(this_, args);
  } catch (err) {
    // The real provider call failed — this is the customer's error and must
    // propagate. Emit a best-effort failure log first, but never let telemetry
    // errors mask or replace the original error.
    const elapsedMs = Date.now() - _callStart;
    try {
      if (session) (session as any)._call_outcome = buildCallOutcome(err, elapsedMs);
      _emitCallFailureLog(getClient(), session);
    } catch {
      // fail-safe
    }
    throw err;
  }

  // 2. Decode response body (Uint8Array — already buffered, safe to re-read).
  let parsed: any = null;
  try {
    let body: any = response?.body;
    if (body instanceof Uint8Array) {
      const text = new TextDecoder().decode(body);
      parsed = JSON.parse(text);
    } else if (typeof body === "string") {
      parsed = JSON.parse(body);
    }
  } catch {
    parsed = null;
  }

  // 3. Manual log with operation=embedding.
  try {
    const tp = getClient();
    if (tp && session) {
      const cmdInput = args[0]?.input ?? {};
      const modelId = String(cmdInput.modelId ?? "");
      const extracted = _extractBedrockEmbeddingUsage(modelId, parsed);
      let inputTokens = extracted.inputTokens;
      let rawUsage = extracted.rawUsage;
      const { shape } = extracted;
      const originalProvider = _bedrockEmbeddingOriginalProvider(modelId);

      // Bedrock-Cohere quirk (AWS-documented): the Embed route strips
      // Cohere's native meta.billed_units block, so the extractor lands at
      // inputTokens=0 even on successful calls (verified against
      // docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html).
      // Approximate from the decoded request body — same rule-of-thumb the
      // HF and Google-mldev embedding paths use. Applied generically (any
      // shape) so a future Bedrock route with the same gap would also recover.
      if (inputTokens === 0 && rawUsage && typeof rawUsage === "object") {
        const approx = _approximateBedrockEmbeddingTokens(cmdInput);
        if (approx > 0) {
          inputTokens = approx;
          rawUsage = { ...rawUsage, approx_input_tokens: approx, approximated: true };
        }
      }

      const spanObj = {
        ...manualSpanIds(session),
        span_kind: "llm" as const,
        span_name: spanName ?? modelId,
        span_order: order,
        start_time: startTime.toISOString(),
        end_time: new Date().toISOString(),
      };

      const metadata: Record<string, unknown> = { workflow_name: session.workflowName };
      if (session.sessionId) metadata.session_id = session.sessionId;
      // B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
      // session metadata WITHOUT it, then re-add it only when this row belongs
      // to the call that was actually rerouted (exact obs-key match).
      if (session.metadata) {
        copySessionMetadata(metadata, session.metadata);
      }
      stampRoutingMarker(metadata, session, _currentObsKey());

      // Build composition from the decoded request body.
      const compKwargs = _bedrockEmbeddingKwargs(cmdInput);
      let promptComp: unknown[] = [];
      try {
        promptComp = buildPromptComposition(originalProvider, compKwargs, "embedding");
      } catch {
        promptComp = [];
      }

      tp.log(
        session.userId,
        session.paidPlan,
        session.workflowName,
        session.sessionId,
        modelId,
        "bedrock",
        inputTokens,
        0, 0,
        metadata,
        spanObj,
        promptComp,
        [],
        {
          usage: { shape, raw: rawUsage },
          operation: "embedding",
          model_extras: { original_provider: originalProvider },
          planSource: session.planSource,
        },
      );
    }
  } catch {
    // fail-open
  }

  return response;
}

/**
 * Per-provider default embedding shape. Used when an embedding wrapper's
 * registry entry omits an explicit `shape` override. Mirrors the Python
 * `_EMBEDDING_SHAPE_BY_PROVIDER` table.
 */
function _resolveEmbeddingShape(provider: string): string {
  switch ((provider || "").toLowerCase()) {
    case "openai":      return "openai_embeddings";
    case "google":
    case "gemini":      return "google_genai_embeddings";
    case "cohere":      return "cohere_embed";
    case "mistral":     return "mistral_embed";
    case "voyage":      return "voyage_embed";
    case "huggingface": return "huggingface_embed";
    case "together":    return "together_embed";
    case "litellm":     return "openai_embeddings";
    // Vercel AI SDK EmbeddingModel — usage is { tokens } per the spec.
    case "ai_sdk":      return "vercel_ai_embed";
    default:            return "openai_embeddings";
  }
}

/**
 * Extract (model, inputTokens, rawUsage) from an embeddings response. Mirrors
 * the Python `_extract_embedding_usage` per-provider response-shape table.
 * Fail-safe: any failure returns ("unknown", 0, null) so the SDK still emits
 * a log row.
 */
export function _extractEmbeddingUsage(
  result: any,
  provider: string,
): { model: string; inputTokens: number; rawUsage: Record<string, unknown> | null } {
  if (result == null) return { model: "unknown", inputTokens: 0, rawUsage: null };
  const p = (provider || "").toLowerCase();
  try {
    if (p === "openai" || p === "together" || p === "litellm") {
      const u = result.usage ?? {};
      const prompt = Number(u.prompt_tokens ?? u.total_tokens ?? 0) || 0;
      return {
        model: String(result.model ?? "unknown"),
        inputTokens: prompt,
        rawUsage: { prompt_tokens: prompt, total_tokens: Number(u.total_tokens ?? prompt) || prompt },
      };
    }
    if (p === "google" || p === "gemini") {
      // `EmbedContentResponse` has NO usage_metadata (that's only on
      // generate_content). Walk the three actual sources, in priority:
      // 1. Vertex per-embedding: result.embeddings[i].statistics.tokenCount
      // 2. Vertex char-billed: result.metadata.billableCharacterCount → chars/4
      // 3. mldev / Gemini API: nothing — caller approximates from request
      const embeddings = Array.isArray(result.embeddings) ? result.embeddings : [];
      let vertexTokens = 0;
      for (const emb of embeddings) {
        const stats = emb?.statistics;
        if (stats) {
          vertexTokens += Number(stats.tokenCount ?? stats.token_count ?? 0) || 0;
        }
      }
      if (vertexTokens > 0) {
        return {
          model: "unknown",
          inputTokens: vertexTokens,
          rawUsage: {
            prompt_token_count: vertexTokens,
            total_token_count: vertexTokens,
          },
        };
      }
      const metadata = result.metadata;
      if (metadata) {
        const chars = Number(metadata.billableCharacterCount ?? metadata.billable_character_count ?? 0) || 0;
        if (chars > 0) {
          const approx = Math.max(1, Math.floor(chars / 4));
          return {
            model: "unknown",
            inputTokens: approx,
            rawUsage: {
              billable_character_count: chars,
              approx_input_tokens: approx,
              approximated: true,
            },
          };
        }
      }
      // mldev/Gemini API — no response-side usage. Return null so the
      // caller (_logManual) approximates from the request kwargs.
      return { model: "unknown", inputTokens: 0, rawUsage: null };
    }
    if (p === "cohere") {
      // The cohere-ai TS SDK camelCases the wire format (meta.billedUnits.
      // inputTokens); raw-HTTP/REST responses keep snake_case. Read snake_case
      // first (raw REST), fall back to camelCase (SDK objects) — without the
      // fallback, SDK-client embeds land 0 tokens / cost unmeasured.
      const billed =
        result.meta?.billed_units ?? result.meta?.billedUnits ?? {};
      const inputTokens =
        Number(billed.input_tokens ?? billed.inputTokens ?? 0) || 0;
      const images = Number(billed.images ?? 0) || 0;
      return {
        model: "unknown",
        inputTokens,
        rawUsage: { meta: { billed_units: { input_tokens: inputTokens, images } } },
      };
    }
    if (p === "mistral") {
      const u = result.usage ?? {};
      const prompt = Number(u.prompt_tokens ?? u.total_tokens ?? 0) || 0;
      return {
        model: String(result.model ?? "unknown"),
        inputTokens: prompt,
        rawUsage: { prompt_tokens: prompt, total_tokens: Number(u.total_tokens ?? prompt) || prompt },
      };
    }
    if (p === "voyage") {
      const u = result.usage ?? {};
      // Raw REST returns snake_case `total_tokens`; the Fern-generated `voyageai`
      // client (>=0.4.0) returns camelCase `totalTokens`. Read both — without
      // the fallback every SDK-client embed billed 0 tokens.
      const total = Number(u.total_tokens ?? u.totalTokens ?? 0) || 0;
      return {
        model: String(result.model ?? "unknown"),
        inputTokens: total,
        rawUsage: { total_tokens: total },
      };
    }
    if (p === "huggingface") {
      // feature_extraction returns a raw vector — no usage object. The wrapper
      // populates an approximate count on the rawUsage object before this call.
      return { model: "unknown", inputTokens: 0, rawUsage: { approx_input_tokens: 0 } };
    }
    if (p === "ai_sdk") {
      // Vercel AI SDK EmbeddingModel doEmbed → { embeddings, usage?: { tokens } }.
      // The model id is instance-bound; _logManual recovers it from the kwargs
      // copy (the embed wrapper stashes `model` there).
      const tokens = Number(result.usage?.tokens ?? 0) || 0;
      return { model: "unknown", inputTokens: tokens, rawUsage: { tokens } };
    }
    const u = result.usage ?? {};
    const prompt = Number(u.prompt_tokens ?? u.total_tokens ?? 0) || 0;
    return {
      model: String(result.model ?? "unknown"),
      inputTokens: prompt,
      rawUsage: { prompt_tokens: prompt },
    };
  } catch {
    return { model: "unknown", inputTokens: 0, rawUsage: null };
  }
}

/**
 * Best-effort token count when Bedrock returns no usable usage. Reuses
 * _bedrockEmbeddingKwargs to decode the request body once, then sums
 * `max(1, len(t) / 4)` over the input strings — same heuristic as
 * _approximateHfEmbeddingTokens. Returns 0 on any failure so the caller
 * can fall through.
 *
 * Triggered by the AWS-documented gap where Bedrock's Cohere Embed route
 * strips Cohere's native meta.billed_units field (see
 * docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-embed-v3.html).
 */
function _approximateBedrockEmbeddingTokens(cmdInput: any): number {
  try {
    const decoded = _bedrockEmbeddingKwargs(cmdInput || {});
    const texts = (decoded as any).texts ?? (decoded as any).inputs;
    if (Array.isArray(texts)) {
      let total = 0;
      for (const t of texts) {
        if (t != null) total += Math.max(1, Math.floor(String(t).length / 4));
      }
      return total;
    }
    const single = (decoded as any).input ?? (decoded as any).inputText;
    if (typeof single === "string") return Math.max(1, Math.floor(single.length / 4));
  } catch {
    return 0;
  }
  return 0;
}

/**
 * Best-effort chars→tokens estimate (len/4, the canonical order-of-magnitude
 * rule also used by the HF embedding approximation below). Used by the G5-1
 * no-usage-chunk stream fallback; always paired with `approximated: true` in
 * the raw usage so the collector reports cost_status='approximated', never an
 * exact 'measured'. Fail-safe: non-string / hostile input → 0.
 */
function _approximateTokensFromChars(text: unknown): number {
  try {
    const len = typeof text === "string" ? text.length : 0;
    return len > 0 ? Math.ceil(len / 4) : 0;
  } catch {
    return 0;
  }
}

/**
 * Best-effort token count for HuggingFace feature_extraction. The Node
 * SDK's signature is `featureExtraction({ inputs, model, ... })`. With no
 * usage object on return, approximate via len(text) / 4 — the canonical
 * order-of-magnitude estimate.
 */
function _approximateHfEmbeddingTokens(args: any[]): number {
  try {
    const req = args?.[0];
    if (!req || typeof req !== "object") return 0;
    const text = (req as any).inputs ?? (req as any).text;
    if (typeof text === "string") return Math.max(1, Math.floor(text.length / 4));
    if (Array.isArray(text)) {
      let total = 0;
      for (const t of text) {
        if (typeof t === "string") total += Math.max(1, Math.floor(t.length / 4));
      }
      return total;
    }
  } catch {
    // fail-safe
  }
  return 0;
}

/**
 * Recursively sums character lengths in any Google `embed_content` `contents`
 * shape and divides by 4. Handles str / list[str] / Content / list[Content],
 * including Content objects with `.parts[i].text`. Returns 0 on any failure.
 */
function _approxCharsToTokens(payload: any): number {
  try {
    if (payload == null) return 0;
    if (typeof payload === "string") return Math.max(1, Math.floor(payload.length / 4));
    if (Array.isArray(payload)) {
      let total = 0;
      for (const item of payload) total += _approxCharsToTokens(item);
      return total;
    }
    if (typeof payload === "object") {
      if ("parts" in payload && payload.parts != null) {
        return _approxCharsToTokens(payload.parts);
      }
      if (typeof payload.text === "string") {
        return Math.max(1, Math.floor(payload.text.length / 4));
      }
    }
  } catch {
    // fail-safe
  }
  return 0;
}

/**
 * Approximate input tokens for Google embed_content when the response
 * carries no usage telemetry (mldev/Gemini API path). The Node @google/genai
 * SDK's `embedContent({ model, contents })` keeps the input under
 * `args[0].contents`. Walks the structure, sums character lengths / 4.
 * Mirrors `_approximateHfEmbeddingTokens`. Fail-safe.
 */
function _approximateGoogleEmbeddingTokens(args: any[]): number {
  try {
    const req = args?.[0];
    if (!req || typeof req !== "object") return 0;
    return _approxCharsToTokens((req as any).contents);
  } catch {
    // fail-safe
  }
  return 0;
}

/**
 * Returns the verbatim provider usage object pulled off the response. Mirrors
 * the access patterns in _extractUsage() so providers with non-standard shapes
 * (Cohere `meta.billed_units`, Bedrock `usage`, Google `usageMetadata`, OpenAI
 * Responses `response.completed` event) all surface a single object.
 */
function _extractRawUsage(provider: string, result: any): unknown {
  if (!result || typeof result !== "object") return null;
  const p = (provider || "").toLowerCase();

  if (p === "google" || p === "gemini") {
    return result.usageMetadata ?? null;
  }
  if (p === "bedrock") {
    return result.usage ?? null;
  }
  if (p === "cohere") {
    // OpenLLMetry Cohere v2 path may surface `delta.usage` on stream chunks.
    return result.usage ?? result.delta?.usage ?? null;
  }
  if (p === "openai_responses") {
    // The streaming `response.completed` event nests usage at .response.usage.
    // Part 2: do NOT inject image_output_count into the chat-span usage.
    // Built-in image_generation_call items are billed on child `image` spans
    // via `_logResponsesImageChildren` (openai_images shape + gpt-image-*).
    // Folding the count into the mainline chat row left text-only pricing
    // with unpriced image units (Part 1 demoted to `partial`; dollars still
    // ~30× under). Chat usage stays text/cache tokens only.
    return result.response?.usage ?? result.usage ?? null;
  }
  if (p === "mistral") {
    // Mistral wraps non-stream responses in `.data` when called from the SDK helper.
    return result.usage ?? result.data?.usage ?? null;
  }
  if (_isAiSdkParse(p)) {
    // Vercel AI SDK — usage sits at .usage on both the doGenerate result and
    // the stream's terminal `finish` part (which is what stream wrappers pass
    // in as `result`). Forward the spec V2/V3 object verbatim — except when
    // the spec usage shows zero input while providerMetadata.anthropic.usage
    // carries the real count (MiniMax defers input_tokens to message_delta;
    // @ai-sdk/anthropic builds spec usage from message_start only). The
    // server-side vercel_ai mapper bills from this raw object, so the
    // recovery in _extractUsage alone would be overridden — patch the
    // forwarded raw too. Gated: any non-zero spec input passes verbatim.
    const u: any = result.usage ?? null;
    try {
      if (u && typeof u === "object") {
        const nested = u.inputTokens !== null && typeof u.inputTokens === "object";
        const specIn = nested
          ? Number(u.inputTokens.total) || 0
          : Number(u.inputTokens ?? u.promptTokens) || 0;
        if (specIn === 0) {
          const rawAnthropic = (result as any)?.providerMetadata?.anthropic?.usage;
          const recovered = Number(rawAnthropic?.input_tokens) || 0;
          if (recovered > 0) {
            if (nested) {
              const cacheRead = Number(u.inputTokens.cacheRead) || 0;
              return {
                ...u,
                inputTokens: {
                  ...u.inputTokens,
                  total: recovered,
                  ...(u.inputTokens.noCache != null
                    ? { noCache: Math.max(0, recovered - cacheRead) }
                    : {}),
                },
                __tp_recovered_input: true,
              };
            }
            return { ...u, inputTokens: recovered, __tp_recovered_input: true };
          }
        }
      }
    } catch {
      // fail-safe: forward verbatim
    }
    return u;
  }
  // Default: OpenAI-compatible — usage sits at .usage on the response.
  return result.usage ?? null;
}

// ────────────────────────────────────────────────────────────────────────────
// Modality (image / audio / video / OCR) extraction registry.
//
// Each handler returns:
// intent(args) → forwarded to /check so rules can match on
// intent.kind / intent.count / etc.
// extract(args, result) → {items, duration, raw} fed into the same
// usage = {shape, raw, items, duration} contract
// used for text calls.
//
// Direct port of the Python SDK's `_MODALITY_HANDLERS`.
// ────────────────────────────────────────────────────────────────────────────

type ModalityKey = "image_gen" | "audio_tts" | "audio_stt" | "video_gen" | "ocr";

interface ModalityExtract {
  items?: Record<string, number | string>;
  duration?: Record<string, number>;
  raw?: unknown;
}

interface ModalityHandler {
  intent: (args: any[]) => Record<string, unknown>;
  extract: (args: any[], result: any) => ModalityExtract;
}

function _safeInt(v: unknown, fallback = 0): number {
  if (v == null) return fallback;
  const n = typeof v === "number" ? v : Number(v);
  return Number.isFinite(n) ? Math.floor(n) : fallback;
}

/** Format positive width×height as "WxH" for items.image_size; "" if unknown. */
function _imageSizeFromDims(w: unknown, h: unknown): string {
  const width = _safeInt(w, 0);
  const height = _safeInt(h, 0);
  if (width > 0 && height > 0) return `${width}x${height}`;
  return "";
}

// ── image size resolution (request → response → binary → default) ──────
// Mapper only multiplies parseable "WxH" × count. Request dims are often
// omitted (provider defaults). Resolve from response metadata / in-memory
// image headers before falling back to documented API defaults. Never fetch
// URLs. Fail-open always — GOLDEN RULE.
const _IMAGE_SIZE_PARSE_RE = /^(\d+)\s*(?:[x×*]|-x-)\s*(\d+)$/i;

function _parseImageSizeStr(sizeStr: unknown): string {
  if (sizeStr == null) return "";
  try {
    const s = String(sizeStr).trim();
    if (!s) return "";
    const m = _IMAGE_SIZE_PARSE_RE.exec(s);
    if (!m) return "";
    const w = Number(m[1]);
    const h = Number(m[2]);
    if (!Number.isFinite(w) || !Number.isFinite(h) || w <= 0 || h <= 0) return "";
    if (w > 1e7 || h > 1e7) return "";
    return `${Math.floor(w)}x${Math.floor(h)}`;
  } catch {
    return "";
  }
}

function _asUint8(data: unknown): Uint8Array | null {
  try {
    if (data == null) return null;
    if (data instanceof Uint8Array) return data;
    if (typeof Buffer !== "undefined" && Buffer.isBuffer(data)) {
      return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    }
    if (data instanceof ArrayBuffer) return new Uint8Array(data);
    if (ArrayBuffer.isView(data)) {
      const v = data as ArrayBufferView;
      return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
    }
  } catch {
    /* fall through */
  }
  return null;
}

/** Read width×height from PNG / JPEG / WEBP headers. In-memory only. */
function _imageDimsFromBinary(data: unknown): string {
  try {
    const buf = _asUint8(data);
    if (!buf || buf.length < 24) return "";
    // PNG
    if (
      buf[0] === 0x89 &&
      buf[1] === 0x50 &&
      buf[2] === 0x4e &&
      buf[3] === 0x47 &&
      buf[12] === 0x49 &&
      buf[13] === 0x48 &&
      buf[14] === 0x44 &&
      buf[15] === 0x52
    ) {
      const w = (buf[16] << 24) | (buf[17] << 16) | (buf[18] << 8) | buf[19];
      const h = (buf[20] << 24) | (buf[21] << 16) | (buf[22] << 8) | buf[23];
      return _imageSizeFromDims(w >>> 0, h >>> 0);
    }
    // JPEG
    if (buf[0] === 0xff && buf[1] === 0xd8) {
      let i = 2;
      const limit = Math.min(buf.length, 65536);
      while (i + 9 < limit) {
        if (buf[i] !== 0xff) {
          i++;
          continue;
        }
        while (i < limit && buf[i] === 0xff) i++;
        if (i >= limit) break;
        const marker = buf[i++];
        if (marker === 0xd8 || marker === 0xd9 || (marker >= 0xd0 && marker <= 0xd7)) {
          continue;
        }
        if (i + 1 >= limit) break;
        const segLen = (buf[i] << 8) | buf[i + 1];
        if (segLen < 2) break;
        if ((marker === 0xc0 || marker === 0xc2) && i + 7 < limit) {
          const h = (buf[i + 3] << 8) | buf[i + 4];
          const w = (buf[i + 5] << 8) | buf[i + 6];
          return _imageSizeFromDims(w, h);
        }
        i += segLen;
      }
      return "";
    }
    // WEBP
    if (
      buf.length >= 30 &&
      buf[0] === 0x52 &&
      buf[1] === 0x49 &&
      buf[2] === 0x46 &&
      buf[3] === 0x46 &&
      buf[8] === 0x57 &&
      buf[9] === 0x45 &&
      buf[10] === 0x42 &&
      buf[11] === 0x50
    ) {
      const fourcc = String.fromCharCode(buf[12], buf[13], buf[14], buf[15]);
      if (fourcc === "VP8X" && buf.length >= 30) {
        const w = 1 + (buf[24] | (buf[25] << 8) | (buf[26] << 16));
        const h = 1 + (buf[27] | (buf[28] << 8) | (buf[29] << 16));
        return _imageSizeFromDims(w, h);
      }
      if (fourcc === "VP8 " && buf.length >= 30) {
        const w = (buf[26] | (buf[27] << 8)) & 0x3fff;
        const h = (buf[28] | (buf[29] << 8)) & 0x3fff;
        return _imageSizeFromDims(w, h);
      }
      if (fourcc === "VP8L" && buf.length >= 25) {
        const bits =
          buf[21] | (buf[22] << 8) | (buf[23] << 16) | (buf[24] << 24);
        const w = (bits & 0x3fff) + 1;
        const h = ((bits >> 14) & 0x3fff) + 1;
        return _imageSizeFromDims(w, h);
      }
    }
    return "";
  } catch {
    return "";
  }
}

function _b64PrefixToBytes(s: unknown, maxDecoded = 96): Uint8Array | null {
  // Only touch a small prefix — never strip/allocate the full multi-MB b64_json.
  try {
    if (typeof s !== "string" || !s) return null;
    let str = s;
    if (str.slice(0, 80).includes("base64,")) {
      str = str.split("base64,")[1] ?? "";
    }
    const want = Math.floor(((maxDecoded + 3) * 4) / 3);
    // Extra headroom so whitespace/newlines in wrapped b64 still yield enough chars.
    let prefix = str.slice(0, want + 64).replace(/\s+/g, "");
    if (!prefix) return null;
    let nChars = Math.min(prefix.length, want);
    nChars -= nChars % 4;
    if (nChars < 24) return null;
    const chunk = prefix.slice(0, nChars);
    if (typeof Buffer !== "undefined") {
      return new Uint8Array(Buffer.from(chunk, "base64"));
    }
    // Browser / edge fallback
    const bin = atob(chunk);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  } catch {
    return null;
  }
}

function _attrOrKey(obj: any, name: string): unknown {
  if (obj == null) return undefined;
  try {
    if (typeof obj === "object" && !Array.isArray(obj) && name in obj) {
      return (obj as any)[name];
    }
    if (typeof obj === "object") return (obj as any)[name];
  } catch {
    return undefined;
  }
  return undefined;
}

type _ImagePayload =
  | { kind: "dims"; w: unknown; h: unknown }
  | { kind: "size_str"; s: unknown }
  | { kind: "bytes"; data: unknown }
  | { kind: "b64"; s: string };

function *_iterImagePayloads(result: any): Generator<_ImagePayload> {
  if (result == null) return;
  try {
    // PIL-like / Blob.size is a number (byte length) — only treat tuple/array as dims
    const sz = _attrOrKey(result, "size");
    if (Array.isArray(sz) && sz.length >= 2) {
      yield { kind: "dims", w: sz[0], h: sz[1] };
    } else if (sz && typeof sz === "object" && !Array.isArray(sz)) {
      yield {
        kind: "dims",
        w: (sz as any).width,
        h: (sz as any).height,
      };
    }
  } catch {
    /* continue */
  }
  if (
    result instanceof Uint8Array ||
    (typeof Buffer !== "undefined" && Buffer.isBuffer(result)) ||
    result instanceof ArrayBuffer
  ) {
    yield { kind: "bytes", data: result };
    return;
  }
  let data = _attrOrKey(result, "data");
  if (data == null && Array.isArray(result)) data = result;
  try {
    if (Array.isArray(data)) {
      for (const item of data) {
        for (const key of ["b64_json", "base64", "b64"] as const) {
          const v = _attrOrKey(item, key);
          if (typeof v === "string" && v) {
            yield { kind: "b64", s: v };
            break;
          }
        }
        for (const key of ["bytes", "image_bytes", "imageBytes", "content"] as const) {
          const v = _attrOrKey(item, key);
          if (v != null && _asUint8(v)) {
            yield { kind: "bytes", data: v };
            break;
          }
        }
        const img = _attrOrKey(item, "image");
        if (img != null) {
          for (const key of ["image_bytes", "imageBytes", "data", "bytes"] as const) {
            const v = _attrOrKey(img, key);
            if (v != null && _asUint8(v)) {
              yield { kind: "bytes", data: v };
              break;
            }
            if (typeof v === "string" && v) {
              yield { kind: "b64", s: v };
              break;
            }
          }
        }
        const w = _attrOrKey(item, "width");
        const h = _attrOrKey(item, "height");
        if (w != null && h != null) yield { kind: "dims", w, h };
        const isz = _attrOrKey(item, "size");
        if (typeof isz === "string" && isz) yield { kind: "size_str", s: isz };
        else if (Array.isArray(isz) && isz.length >= 2) {
          yield { kind: "dims", w: isz[0], h: isz[1] };
        }
      }
    }
  } catch {
    /* continue */
  }
  try {
    const b64 = _attrOrKey(result, "base64");
    if (typeof b64 === "string" && b64) yield { kind: "b64", s: b64 };
  } catch {
    /* continue */
  }
  for (const collName of ["generatedImages", "generated_images", "images"] as const) {
    const coll = _attrOrKey(result, collName);
    if (!Array.isArray(coll)) continue;
    try {
      for (const item of coll) {
        const img = _attrOrKey(item, "image") ?? item;
        for (const key of ["imageBytes", "image_bytes", "data", "bytes"] as const) {
          const v = _attrOrKey(img, key);
          if (v != null && _asUint8(v)) {
            yield { kind: "bytes", data: v };
            break;
          }
          if (typeof v === "string" && v) {
            yield { kind: "b64", s: v };
            break;
          }
        }
        // AI SDK: images may be Uint8Array directly
        if (item instanceof Uint8Array || (typeof Buffer !== "undefined" && Buffer.isBuffer(item))) {
          yield { kind: "bytes", data: item };
        }
      }
    } catch {
      /* continue */
    }
  }
  try {
    const tsz = _attrOrKey(result, "size");
    if (typeof tsz === "string" && tsz) yield { kind: "size_str", s: tsz };
    const w = _attrOrKey(result, "width");
    const h = _attrOrKey(result, "height");
    if (w != null && h != null) yield { kind: "dims", w, h };
  } catch {
    /* continue */
  }
}

function _imageSizeFromResult(result: any): string {
  try {
    for (const p of _iterImagePayloads(result)) {
      if (p.kind === "dims") {
        const s = _imageSizeFromDims(p.w, p.h);
        if (s) return s;
      } else if (p.kind === "size_str") {
        const s = _parseImageSizeStr(p.s);
        if (s) return s;
      } else if (p.kind === "bytes") {
        const s = _imageDimsFromBinary(p.data);
        if (s) return s;
      } else if (p.kind === "b64") {
        const raw = _b64PrefixToBytes(p.s);
        if (raw) {
          const s = _imageDimsFromBinary(raw);
          if (s) return s;
        }
      }
    }
  } catch {
    /* fall through */
  }
  return "";
}

function _imageSizeProviderDefault(
  provider: string | undefined,
  model: string | undefined,
  requestSize?: unknown,
): string {
  try {
    const p = (provider || "").toLowerCase();
    const req = requestSize != null ? String(requestSize).trim().toLowerCase() : "";
    if (req && req !== "auto" && req !== "null" && req !== "none") {
      const parsed = _parseImageSizeStr(requestSize);
      if (parsed) return parsed;
      // Explicit non-pixel intent (aspect ratio / tier) — do not invent.
      if (req.includes(":") || req === "1k" || req === "2k" || req === "4k") {
        return "";
      }
    }
    if (p === "together") return "1024x1024";
    if (p === "openai") return "1024x1024";
    if (p === "huggingface" || p === "hf") return "1024x1024";
    if (p === "google" || p === "gemini") return "1024x1024";
    // xai / ai_sdk / unknown: no invent
    return "";
  } catch {
    return "";
  }
}

/**
 * Cascade: request WxH → request dims → response/binary → provider default.
 * Never throws. Never fetches URLs.
 */
function _resolveImageSize(opts: {
  requestSize?: unknown;
  requestW?: unknown;
  requestH?: unknown;
  result?: any;
  provider?: string;
  model?: string;
  allowDefault?: boolean;
}): string {
  try {
    const s1 = _parseImageSizeStr(opts.requestSize);
    if (s1) return s1;
    const s2 = _imageSizeFromDims(opts.requestW, opts.requestH);
    if (s2) return s2;
    const s3 = _imageSizeFromResult(opts.result);
    if (s3) return s3;
    if (opts.allowDefault !== false) {
      return _imageSizeProviderDefault(opts.provider, opts.model, opts.requestSize);
    }
    return "";
  } catch {
    return "";
  }
}

function _dumpRawUsage(obj: any): Record<string, unknown> {
  if (!obj) return {};
  try {
    if (typeof obj.toJSON === "function") return obj.toJSON();
    if (typeof obj === "object") return { ...obj };
  } catch {
    // fail-safe
  }
  return {};
}

function _audioFileSeconds(handle: unknown): number {
  // Best-effort duration extraction from an audio file handle, mirroring the
  // Python SDK's `_audio_file_seconds` (which reads the file via mutagen/wave).
  // Node has no binary audio dependency, but a canonical PCM WAV header is
  // trivially parseable in pure JS — read `byteRate` + the `data` chunk size
  // and divide. Any failure (non-WAV, unreadable, non-path handle) returns 0,
  // and the server falls back to per-token billing. (Customers can pass
  // `duration` in the request payload to override.)
  try {
    let path: string | null = null;
    if (typeof handle === "string") path = handle;
    else if (
      handle &&
      typeof (handle as any).name === "string" &&
      (handle as any).name
    ) {
      path = (handle as any).name;
    }
    if (!path) return 0;
    // Builtin — SDK-relative rung only (a bare require() would throw in the
    // ESM dist; core modules can't be shadowed, so no app-first pass needed).
    // A throwing _sdkRequire is caught by the enclosing try → returns 0.
    const fs = _sdkRequire?.("fs");
    if (!fs) return 0;
    const fd = fs.openSync(path, "r");
    try {
      const stat = fs.fstatSync(fd);
      const header = Buffer.alloc(Math.min(64, Number(stat.size) || 0));
      if (header.length < 44) return 0;
      fs.readSync(fd, header, 0, header.length, 0);
      if (
        header.toString("ascii", 0, 4) !== "RIFF" ||
        header.toString("ascii", 8, 12) !== "WAVE"
      ) {
        return 0;
      }
      const byteRate = header.readUInt32LE(28);
      // Canonical PCM layout: 44-byte header, `data` size at offset 40.
      let dataSize = 0;
      if (header.toString("ascii", 36, 40) === "data") {
        dataSize = header.readUInt32LE(40);
      } else {
        // Fallback: total file size minus the standard header.
        dataSize = Math.max(0, (Number(stat.size) || 0) - 44);
      }
      if (!byteRate) return 0;
      return dataSize / byteRate;
    } finally {
      fs.closeSync(fd);
    }
  } catch {
    return 0;
  }
}

// ────────────────────────────────────────────────────────────────────────────
// Best-effort OpenAI TTS SSE tap.
//
// Customer-driven: when the caller passes `stream_format="sse"` to
// `audio.speech.create()`, OpenAI returns text/event-stream where the
// terminal `speech.audio.done` event carries the `usage` object. We never
// auto-inject sse — silently rewriting the request would break customers
// who expect binary audio (see plan §"Why the obvious answer is dangerous").
//
// When the customer DID opt in: we drain the SSE bytes, parse usage,
// reassemble the audio bytes, and hand back a fresh Response wrapping the
// audio. Customer downstream code that calls `.arrayBuffer()` / `.body`
// now sees normal binary mp3 instead of raw SSE text — a transparent
// upgrade. Mid-stream chunked playback IS lost (we buffer fully); without
// our SDK the customer would still have had to parse the SSE bytes
// themselves, so this is no worse off in practice.
//
// TODO: source this model pattern from a server-delivered capability list so
// new TTS models are recognized without an SDK release.
// ────────────────────────────────────────────────────────────────────────────
const OPENAI_TTS_SSE_CAPABLE_MODELS = /^gpt-4o(-mini)?-tts/;
const TP_CAPTURED_USAGE = Symbol.for("tokenpolice.capturedUsage");

function _parseOpenAITtsSseBuffer(buf: Buffer): {
  audio: Buffer;
  usage: Record<string, unknown> | null;
} {
  const audioChunks: Buffer[] = [];
  let usage: Record<string, unknown> | null = null;
  // SSE permits either LF or CRLF frame delimiters; normalize to LF.
  const text = buf.toString("utf-8").replace(/\r\n/g, "\n");
  for (const frame of text.split("\n\n")) {
    for (const line of frame.split("\n")) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trimStart();
      if (payload === "[DONE]") continue;
      let obj: any;
      try {
        obj = JSON.parse(payload);
      } catch {
        continue;
      }
      if (typeof obj?.audio === "string") {
        try {
          audioChunks.push(Buffer.from(obj.audio, "base64"));
        } catch {
          /* ignore */
        }
      }
      if (obj?.type === "speech.audio.done" && obj?.usage) {
        usage = obj.usage as Record<string, unknown>;
      }
    }
  }
  return { audio: Buffer.concat(audioChunks), usage };
}

function _looksLikeFetchResponse(r: any): boolean {
  // Duck-type instead of `instanceof Response`. The OpenAI Node SDK
  // constructs its Response through `_shims/`, which can produce an
  // instance whose [[Prototype]] chain does NOT include the global
  // `Response` class — `instanceof Response` is unreliable in practice.
  return (
    r != null &&
    typeof r === "object" &&
    typeof r.arrayBuffer === "function" &&
    typeof r.headers?.get === "function"
  );
}

async function _maybeTapOpenAITtsSse(
  provider: string,
  target: TargetMethod,
  args: any[],
  result: any,
): Promise<any> {
  // All checks fail-safe — any negative answer returns `result` unchanged so
  // the customer's audio call is never harmed by an SDK-side mistake.
  try {
    if (provider !== "openai" || target.modality !== "audio_tts") return result;
    if (!_looksLikeFetchResponse(result)) return result;
    const body = args && args[0] && typeof args[0] === "object" ? args[0] : null;
    if (!body) return result;
    if ((body as any).stream_format !== "sse") return result;
    const model =
      typeof (body as any).model === "string" ? (body as any).model : "";
    if (!OPENAI_TTS_SSE_CAPABLE_MODELS.test(model)) return result;
    if (!result.body) return result;

    const sseBytes = Buffer.from(await result.arrayBuffer());
    const { audio, usage } = _parseOpenAITtsSseBuffer(sseBytes);

    // Invariant: never hand the customer an empty rebuilt body when the
    // provider actually sent bytes. If audio extraction came up empty but the
    // provider's response was non-empty — e.g. the per-frame audio field was
    // renamed/moved — do NOT fabricate an empty audio/mpeg body (silent data
    // loss, worse than fail-open). Fall back to the original response verbatim:
    // original bytes, original content-type. The customer opted into SSE, so
    // this hands them exactly what the provider sent and they can parse it
    // themselves. Metering is unaffected — usage rides its own frame and is
    // still attached below. Only extraction that produced real audio earns the
    // audio/mpeg upgrade.
    // Covered by tests: "falls back to original SSE bytes verbatim when the
    // audio field moved" and "rebuilds an audio/mpeg body on well-formed frames".
    const extractionEmpty = audio.length === 0 && sseBytes.length > 0;
    const bodyBytes = extractionEmpty ? sseBytes : audio;

    // Build a replacement Response that looks like a normal binary mp3 body.
    const headers = new Headers();
    try {
      result.headers?.forEach?.((v: string, k: string) => {
        const lower = k.toLowerCase();
        if (lower === "content-type" || lower === "content-length") return;
        headers.set(k, v);
      });
    } catch {
      /* ignore */
    }
    if (extractionEmpty) {
      // Fallback path: preserve the provider's original content-type rather
      // than forcing audio/mpeg over bytes we could not turn into audio.
      let origCT: string | null = null;
      try {
        origCT = result.headers?.get?.("content-type") ?? null;
      } catch {
        origCT = null;
      }
      if (origCT) headers.set("content-type", origCT);
    } else {
      headers.set("content-type", "audio/mpeg");
    }
    const replacement: any = new Response(bodyBytes, {
      status: result.status,
      statusText: result.statusText,
      headers,
    });
    if (usage) {
      Object.defineProperty(replacement, TP_CAPTURED_USAGE, {
        value: usage,
        enumerable: false,
        configurable: true,
        writable: false,
      });
    }
    return replacement;
  } catch {
    // Fail-safe: never let the SSE tap break the customer's audio call.
    return result;
  }
}

const MODALITY_HANDLERS: Partial<Record<string, ModalityHandler>> = {
  "openai:image_gen": {
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      return {
        kind: "image_generation",
        count: _safeInt(body.n, 1),
        size: typeof body.size === "string" ? body.size : undefined,
        quality: typeof body.quality === "string" ? body.quality : undefined,
      };
    },
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const raw = _dumpRawUsage(result?.usage);
      const data = Array.isArray(result?.data) ? result.data : [];
      const actual = data.length || _safeInt(body.n, 1);
      const model = String(body.model ?? "");
      const image_size = _resolveImageSize({
        requestSize: body.size,
        result,
        provider: "openai",
        model,
      });
      return {
        items: {
          images_generated: actual,
          image_size,
          image_quality: String(body.quality ?? ""),
          image_model: model,
        },
        duration: {},
        raw,
      };
    },
  },
  "openai:audio_tts": {
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const text = typeof body.input === "string" ? body.input : "";
      return { kind: "audio_speech", character_count: codePointLength(text), voice: body.voice };
    },
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const text = typeof body.input === "string" ? body.input : "";
      // When the caller opted into stream_format=sse, _maybeTapOpenAITtsSse
      // attached the captured `usage` object here. Otherwise (binary path)
      // the response has no usage — the call's cost is reported as unmeasured.
      const captured =
        result && (result as any)[TP_CAPTURED_USAGE]
          ? ((result as any)[TP_CAPTURED_USAGE] as Record<string, unknown>)
          : null;
      return {
        items: {
          tts_characters: codePointLength(text),
          audio_model: String(body.model ?? ""),
          voice: String(body.voice ?? ""),
        },
        duration: {},
        raw: captured ?? _dumpRawUsage(result?.usage),
      };
    },
  },
  "openai:audio_stt": {
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      return {
        kind: "audio_transcription",
        expected_seconds: _audioFileSeconds(body.file) || undefined,
      };
    },
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const raw = _dumpRawUsage(result?.usage);
      const seconds =
        Number((raw as Record<string, unknown>)?.duration ?? 0) ||
        _audioFileSeconds(body.file);
      return {
        items: { audio_model: String(body.model ?? "") },
        duration: { audio_seconds: Number(seconds) || 0 },
        raw,
      };
    },
  },
  "google:image_gen": {
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const config = (body.config ?? {}) as Record<string, unknown>;
      const n =
        _safeInt((config as Record<string, unknown>).numberOfImages, 0) ||
        _safeInt(body.numberOfImages, 1);
      return { kind: "image_generation", count: n };
    },
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const config = (body.config ?? {}) as Record<string, unknown>;
      const images =
        (Array.isArray(result?.generatedImages) && result.generatedImages) ||
        (Array.isArray(result?.images) && result.images) ||
        [];
      const count =
        images.length ||
        _safeInt((config as Record<string, unknown>).numberOfImages, 0) ||
        _safeInt(body.numberOfImages, 1);
      const model = String(body.model ?? "");
      // Aspect / tier strings are not pixel dims — pass through so the default
      // path refuses to invent over an explicit non-pixel size intent.
      const aspect =
        config.aspectRatio ??
        config.aspect_ratio ??
        body.aspectRatio ??
        body.aspect_ratio;
      const tierSize =
        config.imageSize ?? config.image_size ?? body.imageSize ?? body.image_size;
      const requestSize =
        aspect != null
          ? String(aspect)
          : tierSize != null
            ? String(tierSize)
            : undefined;
      const image_size = _resolveImageSize({
        requestSize,
        requestW: config.width ?? body.width,
        requestH: config.height ?? body.height,
        result,
        provider: "google",
        model,
      });
      return {
        items: {
          images_generated: count,
          image_model: model,
          image_size,
        },
        duration: {},
        raw: {},
      };
    },
  },
  "google:video_gen": {
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const config = (body.config ?? {}) as Record<string, unknown>;
      const secs =
        _safeInt((config as Record<string, unknown>).durationSeconds, 0) ||
        _safeInt(body.durationSeconds, 0);
      return { kind: "video_generation", expected_seconds: secs || undefined };
    },
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const config = (body.config ?? {}) as Record<string, unknown>;
      let secs =
        _safeInt((config as Record<string, unknown>).durationSeconds, 0) ||
        _safeInt(body.durationSeconds, 0);
      const opMeta = (result?.metadata ?? {}) as Record<string, unknown>;
      if (!secs && opMeta) {
        secs =
          _safeInt(opMeta.videoDurationSeconds, 0) ||
          _safeInt(opMeta.durationSeconds, 0);
      }
      return {
        items: { video_model: String(body.model ?? "") },
        duration: { video_seconds: secs || 0 },
        raw: {},
      };
    },
  },
  "together:image_gen": {
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      return { kind: "image_generation", count: _safeInt(body.n, 1) };
    },
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const data = Array.isArray(result?.data) ? result.data : [];
      const actual = data.length || _safeInt(body.n, 1);
      const model = String(body.model ?? "");
      // Cascade: request dims → response b64 header → API default 1024×1024.
      const image_size = _resolveImageSize({
        requestW: body.width,
        requestH: body.height,
        result,
        provider: "together",
        model,
      });
      return {
        items: {
          images_generated: actual,
          image_model: model,
          image_size,
        },
        duration: {},
        raw: _dumpRawUsage(result?.usage),
      };
    },
  },
  // ── Mistral non-text modalities ─────────────────────────────────────────────
  // Direct port of Python's `_intent/_extract_mistral_ocr` +
  // `_intent/_extract_mistral_audio_stt`. Node folds the request into args[0]
  // (Speakeasy client passes a single request object), so `model`/`file` are
  // read from args[0] — NOT a positional args[1]. Mistral responses carry
  // duration/page counts under `usage_info` (falling back to `usage`).
  "mistral:ocr": {
    intent: () => ({ kind: "ocr" }),
    extract: (_args, result) => {
      const raw = _dumpRawUsage(result?.usage_info ?? result?.usage);
      const pages = _safeInt((raw as Record<string, unknown>)?.pages_processed, 0);
      return { items: { ocr_pages: pages }, duration: {}, raw };
    },
  },
  "mistral:audio_stt": {
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const secs = _audioFileSeconds(body.file ?? body.audio);
      return { kind: "audio_transcription", expected_seconds: secs || undefined };
    },
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const raw = _dumpRawUsage(result?.usage_info ?? result?.usage);
      const seconds =
        Number((raw as Record<string, unknown>)?.duration ?? 0) ||
        _audioFileSeconds(body.file ?? body.audio);
      return {
        items: { audio_model: String(body.model ?? "") },
        duration: { audio_seconds: Number(seconds) || 0 },
        raw,
      };
    },
  },
  // ── HuggingFace non-text modalities ─────────────────────────────────────────
  // Direct port of Python's `_intent/_extract_hf_image|audio_tts|audio_stt`.
  // Python reads the text/audio at positional `args[1]`; the @huggingface/inference
  // task functions take a single merged request object, so Node reads
  // `inputs`/`text` (text) and `data`/`inputs` (audio) from args[0].
  "huggingface:image_gen": {
    intent: () => ({ kind: "image_generation", count: 1 }),
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const params =
        body.parameters !== null && typeof body.parameters === "object"
          ? (body.parameters as Record<string, unknown>)
          : {};
      const model = String(body.model ?? "");
      // Cascade: request dims → in-memory bytes (Buffer/Uint8Array; Blob skipped
      // because extract is sync) → default 1024×1024.
      const image_size = _resolveImageSize({
        requestW: params.width ?? body.width,
        requestH: params.height ?? body.height,
        result,
        provider: "huggingface",
        model,
      });
      return {
        items: {
          images_generated: 1,
          image_model: model,
          image_size,
        },
        duration: {},
        raw: {},
      };
    },
  },
  "huggingface:audio_tts": {
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const text =
        typeof body.inputs === "string"
          ? body.inputs
          : typeof body.text === "string"
            ? body.text
            : "";
      return { kind: "audio_speech", character_count: codePointLength(text) };
    },
    extract: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const text =
        typeof body.inputs === "string"
          ? body.inputs
          : typeof body.text === "string"
            ? body.text
            : "";
      return {
        items: { tts_characters: codePointLength(text), audio_model: String(body.model ?? "") },
        duration: {},
        raw: {},
      };
    },
  },
  "huggingface:audio_stt": {
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const secs = _audioFileSeconds(body.data ?? body.inputs);
      return { kind: "audio_transcription", expected_seconds: secs || undefined };
    },
    extract: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const secs = _audioFileSeconds(body.data ?? body.inputs);
      return {
        items: { audio_model: String(body.model ?? "") },
        duration: { audio_seconds: Number(secs) || 0 },
        raw: {},
      };
    },
  },
  // ── Vercel AI SDK modality models ──────────────────────────────────────────
  // args[0] is the kwargs copy built by _aiSdkKwargs: the spec call options
  // plus `__tpXaiModel` (the instance-bound model id — call options carry no
  // model field). The *_model items below feed _logModality's model resolution.
  "ai_sdk:image_gen": {
    // ImageModelV2/V3 doGenerate({ prompt, n, size, aspectRatio, ... }) →
    // { images: [], usage?: { inputTokens?, outputTokens?, totalTokens? } }.
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      return {
        kind: "image_generation",
        count: _safeInt(body.n, 1),
        size: typeof body.size === "string" ? body.size : undefined,
      };
    },
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const images = Array.isArray(result?.images) ? result.images : [];
      const model = String(body.__tpXaiModel ?? "");
      // Prefer parseable request size; then binary from images[]; aspect-only
      // stays empty (no invent for ai_sdk — provider-specific defaults unknown).
      const requestSize =
        typeof body.size === "string" && body.size
          ? body.size
          : typeof body.aspectRatio === "string"
            ? body.aspectRatio
            : undefined;
      const image_size = _resolveImageSize({
        requestSize,
        result,
        provider: "ai_sdk",
        model,
        allowDefault: false,
      });
      return {
        items: {
          images_generated: images.length || _safeInt(body.n, 1),
          image_size,
          image_model: model,
        },
        duration: {},
        raw: _dumpRawUsage(result?.usage),
      };
    },
  },
  "ai_sdk:audio_tts": {
    // SpeechModelV2/V3 doGenerate({ text, voice, ... }) → { audio } — the spec
    // result carries NO usage object, so billing is character-based.
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const text = typeof body.text === "string" ? body.text : "";
      return { kind: "audio_speech", character_count: codePointLength(text), voice: body.voice };
    },
    extract: (args, _result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const text = typeof body.text === "string" ? body.text : "";
      return {
        items: {
          tts_characters: codePointLength(text),
          audio_model: String(body.__tpXaiModel ?? ""),
          voice: String(body.voice ?? ""),
        },
        duration: {},
        raw: {},
      };
    },
  },
  "ai_sdk:audio_stt": {
    // TranscriptionModelV2/V3 doGenerate({ audio, mediaType }) →
    // { text, segments: [{ startSecond, endSecond }], durationInSeconds? }.
    intent: () => ({ kind: "audio_transcription" }),
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      let seconds = Number(result?.durationInSeconds ?? 0) || 0;
      if (!seconds && Array.isArray(result?.segments) && result.segments.length > 0) {
        seconds = Number(result.segments[result.segments.length - 1]?.endSecond ?? 0) || 0;
      }
      return {
        items: { audio_model: String(body.__tpXaiModel ?? "") },
        duration: { audio_seconds: seconds },
        raw: {},
      };
    },
  },
  "ai_sdk:video_gen": {
    // VideoModelV3 doGenerate({ prompt, n, duration, resolution, ... }) →
    // { videos: [] }. `duration` is seconds PER video — billable seconds are
    // duration × actual video count.
    intent: (args) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const secs = _safeInt(body.duration, 0);
      return {
        kind: "video_generation",
        count: _safeInt(body.n, 1),
        expected_seconds: secs || undefined,
      };
    },
    extract: (args, result) => {
      const body = (args[0] ?? {}) as Record<string, unknown>;
      const videos = Array.isArray(result?.videos) ? result.videos : [];
      const count = videos.length || _safeInt(body.n, 1);
      const perVideo = _safeInt(body.duration, 0);
      return {
        items: {
          videos_generated: count,
          video_model: String(body.__tpXaiModel ?? ""),
        },
        duration: { video_seconds: perVideo * count },
        raw: {},
      };
    },
  },
};

function _buildIntent(
  provider: string,
  modality: ModalityKey | undefined,
  args: any[],
): Record<string, unknown> | null {
  if (!modality) return null;
  const handler = MODALITY_HANDLERS[`${provider}:${modality}`];
  if (!handler) return { kind: modality };
  try {
    const intent = handler.intent(args) || {};
    if (!("kind" in intent)) (intent as any).kind = modality;
    return intent as Record<string, unknown>;
  } catch {
    return { kind: modality };
  }
}

const _logModality = failSafeSync(function (
  provider: string,
  modality: ModalityKey,
  shape: string,
  args: any[],
  result: any,
  order: number,
  spanName: string | null,
  startTime: Date,
  elapsedSeconds: number,
): void {
  const tp = getClient();
  if (!tp) return;
  const handler = MODALITY_HANDLERS[`${provider}:${modality}`];
  if (!handler) return;

  let extracted: ModalityExtract = {};
  try {
    extracted = handler.extract(args, result) || {};
  } catch {
    extracted = {};
  }

  const items = (extracted.items ?? {}) as Record<string, number | string>;
  const duration = (extracted.duration ?? {}) as Record<string, number>;
  const raw = extracted.raw ?? {};
  if (
    elapsedSeconds > 0 &&
    !duration.audio_seconds &&
    !duration.video_seconds
  ) {
    duration.compute_seconds = Number(elapsedSeconds);
  }

  const body = (args && args[0] && typeof args[0] === "object" ? args[0] : {}) as Record<
    string,
    unknown
  >;
  const model =
    (typeof body.model === "string" && body.model) ||
    (items.image_model as string) ||
    (items.audio_model as string) ||
    (items.video_model as string) ||
    "unknown";

  const session = getCurrentSession();
  const spanObj = {
    ...manualSpanIds(session),
    span_kind: "llm" as const,
    span_name: spanName ?? String(model),
    span_order: order,
    start_time: startTime.toISOString(),
    end_time: new Date().toISOString(),
  };

  const metadata: Record<string, unknown> = {
    workflow_name: session.workflowName,
  };
  if (session.sessionId) metadata.session_id = session.sessionId;
  // B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
  // session metadata WITHOUT it, then re-add it only when this row belongs
  // to the call that was actually rerouted (exact obs-key match).
  if (session.metadata) {
    copySessionMetadata(metadata, session.metadata);
  }
  stampRoutingMarker(metadata, session, _currentObsKey());
  metadata.modality = modality;

  let promptComp: unknown[] = [];
  let responseComp: unknown[] = [];
  const compKey = `${session.traceId}:${order}`;
  const compData = session._pendingCompositions[compKey];
  if (compData) {
    promptComp = compData.prompt ?? [];
    responseComp = compData.response ?? [];
    delete session._pendingCompositions[compKey];
  }

  // Vercel AI SDK modality calls parse under the internal "ai_sdk" key but
  // REPORT the provider derived from the model instance (mirrors _logManual).
  // api_base flows from the kwargs-copy `baseURL` stash the same way.
  let reportProvider = provider;
  const modelExtras: Record<string, string> = {};
  try {
    const aiMeta = (args?.[0] as any)?.__tpAiSdk;
    if (aiMeta && typeof aiMeta.provider === "string" && aiMeta.provider) {
      reportProvider = aiMeta.provider;
    }
  } catch {
    // fail-safe
  }
  const apiBase = _extractBaseURL(args?.[0]);
  if (apiBase) modelExtras.api_base = apiBase;

  // Same success-path drain as _logManual — modality calls also run
  // preflight and may stash applied local_decision / reject observations.
  // Keyed: modality calls never stream, so this runs synchronously inside
  // the wrapper's obs scope and the ALS read is this call's own key.
  let localDecision: Record<string, unknown> | undefined;
  let observations: unknown[] = [];
  try {
    const obsKey = _currentObsKey();
    localDecision = claimLocalDecision(session, obsKey);
    observations = state.drainObservations(obsKey);
  } catch {
    localDecision = undefined;
    observations = [];
  }

  tp.log(
    session.userId,
    session.paidPlan,
    session.workflowName,
    session.sessionId,
    String(model),
    reportProvider,
    0,
    0,
    0,
    metadata,
    spanObj,
    promptComp,
    responseComp,
    {
      usage: {
        shape,
        raw,
        items: items as Record<string, number>,
        duration,
      },
      // modality (image_gen / audio_tts / audio_stt / video_gen / ocr) is a
      // canonical `operation` value. Passing it segments spend-by-operation
      // correctly AND lets the modality-aware span_kind derive (image/tts/...).
      operation: modality,
      ...(Object.keys(modelExtras).length > 0 ? { model_extras: modelExtras } : {}),
      ...(localDecision ? { local_decision: localDecision } : {}),
      ...(observations.length > 0 ? { observations } : {}),
      planSource: session.planSource,
    },
  );
});


/**
 * Extracts token usage from a manual-telemetry response and logs it via
 * tp.log(). Used for SDKs with no OpenLLMetry instrumentor (@google/genai,
 * @openrouter/sdk), so the telemetry SpanProcessor never observes these calls.
 */
// ── Part 2: Responses-API built-in image_generation → child image spans ─
//
// The Responses mainline model (gpt-4.1-mini, …) only returns text/cache usage.
// Image spend is on a GPT Image model (default gpt-image-1) invoked as a tool;
// the output item carries b64 only — size/quality/model live on the request
// tools[] config. Emit one child openai_images span per completed image so the
// dedicated image pricing path (count × quality × size) can fire. Fail-open.

// A usable image dimension, or "" when there is nothing to report. 'auto' is the
// provider choosing for us, not an observation, so it must never travel as a
// value — empty string means "not observed" — do not invent a default quality/size.
function _sanitizeImageDim(v: unknown): string {
  if (typeof v !== "string") return "";
  const s = v.trim();
  return s && s.toLowerCase() !== "auto" ? s : "";
}

function _extractResponsesImageToolConfig(args: any[]): {
  model: string;
  size: string;
  quality: string;
} {
  try {
    const body =
      args?.[0] && typeof args[0] === "object"
        ? (args[0] as Record<string, unknown>)
        : null;
    const tools = body?.tools;
    if (!Array.isArray(tools)) {
      return { model: "gpt-image-1", size: "", quality: "" };
    }
    for (const t of tools) {
      if (!t || typeof t !== "object") continue;
      const type = (t as any).type;
      if (type !== "image_generation") continue;
      const model =
        typeof (t as any).model === "string" && (t as any).model
          ? String((t as any).model)
          : "gpt-image-1";
      let size =
        typeof (t as any).size === "string" ? String((t as any).size) : "";
      if (size.toLowerCase() === "auto") size = "";
      let quality =
        typeof (t as any).quality === "string"
          ? String((t as any).quality)
          : "";
      if (quality.toLowerCase() === "auto") quality = "";
      return { model, size, quality };
    }
  } catch {
    // fail-open
  }
  return { model: "gpt-image-1", size: "", quality: "" };
}

/** Completed (or result-bearing) image_generation_call items from a Responses result. */
function _listResponsesImageCalls(result: any): any[] {
  try {
    let resp = result;
    if (resp && resp.type === "response.completed" && resp.response) {
      resp = resp.response;
    }
    const output = resp?.output ?? result?.response?.output ?? result?.output;
    if (!Array.isArray(output)) return [];
    const out: any[] = [];
    for (const it of output) {
      if (!it || typeof it !== "object") continue;
      if ((it as any).type !== "image_generation_call") continue;
      const status = (it as any).status;
      if (status === "failed") continue;
      const res = (it as any).result;
      // Prefer completed / unknown-status items; require a result for partial states.
      if (status === "in_progress" || status === "generating") {
        if (typeof res !== "string" || !res) continue;
      }
      out.push(it);
    }
    return out;
  } catch {
    return [];
  }
}

/**
 * Emit one child image span per Responses image_generation_call.
 * Must run after the chat `_logManual`: the chat row is the call's primary
 * log and owns the session `local_decision` stash plus the keyed
 * observations drain for this call's obs key — running first would let a
 * child image row consume them instead of the chat row. (Cross-CALL
 * observation theft is separately impossible now that drains are keyed;
 * this ordering is about which of the SAME call's rows carries the audit
 * fields.)
 * Never throws (failSafeSync).
 */
const _logResponsesImageChildren = failSafeSync(function (
  args: any[],
  result: any,
  startTime: Date,
): void {
  const tp = getClient();
  if (!tp) return;
  const images = _listResponsesImageCalls(result);
  if (images.length === 0) return;

  const session = getCurrentSession();
  const cfg = _extractResponsesImageToolConfig(args);
  const model = cfg.model || "gpt-image-1";

  for (const item of images) {
    try {
      let order = 0;
      try {
        order = session.nextSpanOrder();
      } catch {
        order = 0;
      }
      const b64 =
        typeof (item as any)?.result === "string" ? (item as any).result : "";
      // Reuse cascade: synthetic Images API shape so b64 header is read.
      const syntheticResult = b64 ? { data: [{ b64_json: b64 }] } : {};
      // Opportunistically read the dims off the output item itself. Today
      // the image_generation_call item ships only {id,result,status,type}, so
      // these are always absent and behavior is unchanged — but undeclared wire
      // fields survive JSON parsing, so if OpenAI starts echoing the
      // server-resolved quality/size we capture it for free. The item value wins
      // over the request tool config because it is what was actually produced
      // (the request may have said nothing, or 'auto'). Never trusted to exist:
      // empty string means "not observed" — leave unset rather than inventing a default.
      const itemSize = _sanitizeImageDim((item as any)?.size);
      const itemQuality = _sanitizeImageDim((item as any)?.quality);
      const image_size = _resolveImageSize({
        // Ahead of cfg.size deliberately: _resolveImageSize short-circuits on the
        // first usable source, and appending it after the b64/provider-default
        // steps would make it dead code.
        requestSize: itemSize || cfg.size || undefined,
        result: syntheticResult,
        provider: "openai",
        model,
      });
      const image_quality = itemQuality || cfg.quality || "";

      const spanObj = {
        ...manualSpanIds(session),
        span_kind: "llm" as const,
        span_name: model,
        span_order: order,
        start_time: startTime.toISOString(),
        end_time: new Date().toISOString(),
      };

      const metadata: Record<string, unknown> = {
        workflow_name: session.workflowName,
        modality: "image_gen",
        // Provenance: billed image was produced by Responses image_generation tool.
        tp_source: "openai_responses_image_generation",
      };
      if (session.sessionId) metadata.session_id = session.sessionId;
      // B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
      // session metadata WITHOUT it, then re-add it only when this row belongs
      // to the call that was actually rerouted (exact obs-key match).
      if (session.metadata) {
        copySessionMetadata(metadata, session.metadata);
      }
      stampRoutingMarker(metadata, session, _currentObsKey());

      const items: Record<string, number | string> = {
        images_generated: 1,
        image_model: model,
        image_size,
        image_quality,
      };

      tp.log(
        session.userId,
        session.paidPlan,
        session.workflowName,
        session.sessionId,
        model,
        "openai",
        0,
        0,
        0,
        metadata,
        spanObj,
        [],
        [{ type: "image", role: "assistant" }],
        {
          usage: {
            shape: "openai_images",
            raw: {},
            items: items as Record<string, number>,
            duration: {},
          },
          operation: "image_gen",
          planSource: session.planSource,
        },
      );
    } catch {
      // per-item fail-open: one bad item must not skip siblings
    }
  }
});

const _logManual = failSafeSync(function (
  provider: string,
  args: any[],
  result: any,
  order: number,
  spanName: string | null,
  startTime: Date,
  operation: string = "chat",
  shapeOverride: string | null = null,
  latency: TPLatency | null = null,
): void {
  const tp = getClient();
  if (!tp) return;

  const session = getCurrentSession();

  // ── Token usage + model name (provider-specific shapes) ──
  let model: string;
  let inputTokens: number;
  let outputTokens: number;
  let cachedTokens: number;
  let embeddingRawUsage: Record<string, unknown> | null = null;
  if (operation === "embedding") {
    const ext = _extractEmbeddingUsage(result, provider);
    model = ext.model;
    inputTokens = ext.inputTokens;
    outputTokens = 0;
    cachedTokens = 0;
    embeddingRawUsage = ext.rawUsage;
    if ((provider || "").toLowerCase() === "huggingface") {
      const approx = _approximateHfEmbeddingTokens(args);
      inputTokens = approx;
      embeddingRawUsage = { approx_input_tokens: approx, approximated: true };
    } else if (
      ((provider || "").toLowerCase() === "google" ||
       (provider || "").toLowerCase() === "gemini") &&
      inputTokens === 0
    ) {
      // mldev/Gemini API: embed_content returns no usage on the response.
      // Vertex paths inside _extractEmbeddingUsage already filled tokens
      // from per-embedding stats or billable_character_count; if we still
      // have 0 here we're on mldev and need to approximate from kwargs.
      const approx = _approximateGoogleEmbeddingTokens(args);
      if (approx > 0) {
        inputTokens = approx;
        embeddingRawUsage = { approx_input_tokens: approx, approximated: true };
      }
    }
    if (!model || model === "unknown") {
      const req = args?.[0];
      if (req && typeof req === "object" && typeof (req as any).model === "string") {
        model = (req as any).model;
      } else {
        model = "unknown";
      }
    }
  } else {
    ({ model, inputTokens, outputTokens, cachedTokens } = _extractUsage(
      provider,
      result,
      args,
    ));
  }

  // The auto span path reports the requested model, this path reads the
  // provider's echo — so a dated snapshot echo (`…-4-5-20251001`) split one
  // model across two rows. Family-gated: only an echo that is the request plus
  // a date suffix collapses back; a genuinely different served model wins.
  const reqModel =
    args?.[0] && typeof (args[0] as any).model === "string"
      ? (args[0] as any).model
      : undefined;
  model = preferRequestedModel(reqModel, model);

  // ── Span hierarchy ──
  const spanObj = {
    ...manualSpanIds(session),
    span_kind: "llm" as const,
    span_name: spanName ?? model,
    span_order: order,
    start_time: startTime.toISOString(),
    end_time: new Date().toISOString(),
  };

  // ── Metadata ──
  const metadata: Record<string, unknown> = {
    workflow_name: session.workflowName,
  };
  if (session.sessionId) metadata.session_id = session.sessionId;
  // B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
  // session metadata WITHOUT it, then re-add it only when this row belongs
  // to the call that was actually rerouted (exact obs-key match).
  if (session.metadata) {
    copySessionMetadata(metadata, session.metadata);
  }
  stampRoutingMarker(metadata, session, _currentObsKey());

  // ── Composition (captured into the session by _captureCompositionAt) ──
  let promptComp: unknown[] = [];
  let responseComp: unknown[] = [];
  // A request-config service tier stashed pre-call (Google GenAI takes the tier
  // on the request and never echoes it in the response). Captured here before
  // compData is dropped; used only as a fallback when the response carries none.
  let stashedServiceTier = "";
  const compKey = `${session.traceId}:${order}`;
  const compData = session._pendingCompositions[compKey];
  if (compData) {
    promptComp = compData.prompt ?? [];
    responseComp = compData.response ?? [];
    if (typeof compData.service_tier === "string") stashedServiceTier = compData.service_tier;
    delete session._pendingCompositions[compKey];
  }

  // ── Raw provider usage ──
  // Stream wrappers stash the final-chunk usage at session._pending_usage; non-
  // streaming wrappers extract from `result` directly. Embedding branch
  // already populated rawUsage from _extractEmbeddingUsage.
  let rawUsage: unknown = null;
  if (operation === "embedding") {
    rawUsage = embeddingRawUsage;
  } else {
    try {
      const stash = (session as any)._pending_usage;
      const stashKey = compKey;
      if (stash && stash[stashKey] !== undefined) {
        rawUsage = stash[stashKey];
        delete stash[stashKey];
      } else {
        rawUsage = _extractRawUsage(provider, result);
      }
    } catch {
      rawUsage = null;
    }
  }

  // Provider-reported service tier (OpenAI response top level / Anthropic
  // usage object) → usage.tier so tier-specific (batch/flex/priority) pricing
  // applies on the manual-telemetry path too. Falls back to a request-config
  // tier stashed pre-call (Google GenAI never echoes `service_tier` in the
  // response) — the response value still wins when present.
  const serviceTier = _extractServiceTier(result) || stashedServiceTier;
  const usageBlock = {
    shape: shapeOverride || (operation === "embedding"
      ? _resolveEmbeddingShape(provider)
      : _resolveUsageShape(provider, args)),
    raw: rawUsage,
    ...(serviceTier ? { tier: serviceTier } : {}),
  };

  // Endpoint hint for server-side disambiguation (Batch API vs Chat etc.).
  const modelExtras: Record<string, string> = {};
  // Serving endpoint (host+path, no credentials/query) → the server maps it to a
  // serving provider (e.g. api.minimax.io → minimax). Shared helper strips
  // userinfo so only host metadata leaves the process. Fail-safe ("" on error).
  const apiBase = _extractBaseURL(args?.[0]);
  if (apiBase) modelExtras.api_base = apiBase;
  const baseURL = (args?.[0]?.baseURL ?? args?.[0]?._client?.baseURL ?? "") as string;
  if (provider === "openai" && /openrouter/i.test(baseURL)) {
    // On the OpenRouter-via-OpenAI-SDK path the request model carries the
    // routed vendor head as a slug prefix ("anthropic/claude-...",
    // "openai/gpt-4o-mini"). `original_provider` means that vendor head — the
    // deployer the model routes to — NOT the physical SDK provider. Mirror the
    // auto/gateway path (see the gateway request-model stash), which derives it
    // from the model slug. When the model has no routed "vendor/model" form,
    // OMIT original_provider entirely — its absence is handled server-side.
    const routedHead =
      typeof model === "string" && model.includes("/")
        ? model.split("/")[0].trim().toLowerCase()
        : "";
    if (routedHead) modelExtras.original_provider = routedHead;
    modelExtras.deployment = "routed";
  }

  // Generic Vercel AI SDK calls parse under the internal "ai_sdk" key, but the
  // REPORTED provider is the one the wrapper derived from the model instance
  // (`model.provider` head — e.g. "minimax", "vercel-gateway"), stashed on the
  // kwargs copy as `__tpAiSdk`. Fail-safe: fall back to the parsing key.
  let reportProvider = provider;
  try {
    const aiMeta = (args?.[0] as any)?.__tpAiSdk;
    if (aiMeta && typeof aiMeta.provider === "string" && aiMeta.provider) {
      reportProvider = aiMeta.provider;
    }
  } catch {
    // fail-safe
  }

  // Success-call duration. Streaming paths pass a monotonic latency block
  // (total_ms spans request→last chunk); non-streaming falls back to the
  // wall-clock around the call. The server maps call_outcome.duration_ms →
  // duration_ms and defaults status to 'success', so this never flips a row
  // to failed (failures log via _emitCallFailureLog, not here).
  let durationMs = 0;
  try {
    durationMs = latency?.total_ms ?? Math.max(0, Date.now() - startTime.getTime());
  } catch {
    durationMs = 0;
  }

  // Drain applied local_decision + shadow/reject observations so
  // stream + manual success paths confirm REQUEST_REROUTED / REROUTE_REJECTED
  // on /log (previously only failure/block paths drained these).
  // Keyed: non-streaming callers run this synchronously inside the wrapper's
  // obs scope; stream wrappers re-enter their captured scope (runWithObsKey)
  // around this call — either way the ALS read is the call's own key. Null
  // (no scope) claims only untagged + stale entries.
  let localDecision: Record<string, unknown> | undefined;
  let observations: unknown[] = [];
  try {
    const obsKey = _currentObsKey();
    localDecision = claimLocalDecision(session, obsKey);
    observations = state.drainObservations(obsKey);
  } catch {
    localDecision = undefined;
    observations = [];
  }

  tp.log(
    session.userId,
    session.paidPlan,
    session.workflowName,
    session.sessionId,
    model,
    reportProvider,
    inputTokens,
    outputTokens,
    cachedTokens,
    metadata,
    spanObj,
    promptComp,
    responseComp,
    {
      usage: usageBlock,
      operation,
      ...(Object.keys(modelExtras).length > 0 ? { model_extras: modelExtras } : {}),
      ...(latency ? { latency } : {}),
      ...(durationMs > 0
        ? { call_outcome: { status: "success", duration_ms: durationMs } }
        : {}),
      ...(localDecision ? { local_decision: localDecision } : {}),
      ...(observations.length > 0 ? { observations } : {}),
      planSource: session.planSource,
    },
  );

  // Part 2: after the chat row is written, emit child image spans for
  // Responses built-in image_generation tool calls (stream + non-stream).
  if ((provider || "").toLowerCase() === "openai_responses") {
    try {
      _logResponsesImageChildren(args, result, startTime);
    } catch {
      // fail-open: never surface from manual log path
    }
  }
});

/**
 * Creates a per-stream accumulator for building response composition from
 * streamed chunks. Returns null for providers whose streaming shape isn't
 * accumulated here (the wrapper then falls back to the final usage chunk).
 */
function _newStreamAccumulator(provider: string): any {
  if (provider === "cohere") {
    return {
      textParts: [] as string[],
      toolPlanParts: [] as string[],
      toolCalls: {} as Record<number, any>,
    };
  }
  // Cerebras is OpenAI-compatible for stream deltas; usage already handles it —
  // membership here is composition-only (must stay in lockstep with accumulate
  // + toResponse branches below). Do not add openrouter without Mode-A gate work.
  if (
    provider === "huggingface" ||
    provider === "mistral" ||
    provider === "together" ||
    provider === "openai" ||
    provider === "litellm" ||
    provider === "groq" ||
    provider === "cerebras"
  ) {
    return {
      textParts: [] as string[],
      toolCalls: {} as Record<number, any>,
    };
  }
  if (provider === "openai_responses") {
    // OpenAI Responses API stream — accumulate by output_item index.
    // items[idx] = { type: "message", textParts: [...] }
    // | { type: "function_call", name, callId, argParts: [...] }
    // | { type: "reasoning" }
    return {
      items: {} as Record<number, any>,
      order: [] as number[],
    };
  }
  if (_isAiSdkParse(provider)) {
    // Vercel AI SDK LanguageModel V2/V3 stream parts. Accumulate text-delta by
    // id (so concurrent text streams stay separate) and tool-call by toolCallId
    // (tool-input-delta fragments). Final part is `finish` with usage.
    return {
      textParts: {} as Record<string, string[]>,
      textOrder: [] as string[],
      toolCalls: {} as Record<string, { toolName: string; argParts: string[]; complete?: any }>,
      reasoningParts: [] as string[],
    };
  }
  if (provider === "anthropic") {
    // Anthropic Messages raw event stream — `create({stream:true})` on an SDK
    // whose root exports `APIPromise` (≥0.35), i.e. the Traceloop-instrumented
    // path (`anthropicStreamBypass` is false there). The instrumentor's own
    // accumulator ignores `input_json_delta` (tool inputs stay `{}`) and never
    // reads `message_delta.delta.stop_reason`, and telemetry.ts's
    // `gen_ai.output.messages` fallback reads text parts only — so a streamed
    // tool_use turn lost its tool_call composition entry and a tool-only turn
    // logged an EMPTY composition. Content blocks keyed by stream index; the
    // block map + rebuild mirror `_tapAnthropicStreamBypass` (the ≤0.34 path)
    // so both client generations fingerprint identically. Composition-only:
    // usage still comes from the instrumentor span / `_chunkHasUsage` stash.
    return { blocks: {} as Record<number, any> };
  }
  return null;
}

/** Folds one streamed chunk into the accumulator (provider-aware). */
function _accumulateStreamChunk(provider: string, acc: any, chunk: any): void {
  if (!acc || !chunk) return;

  if (provider === "openai_responses") {
    // OpenAI Responses API stream events. Relevant types:
    // response.output_item.added — register new item under output_index
    // response.output_text.delta — append text to a message item
    // response.function_call_arguments.delta — append args fragment
    // (response.completed is read separately for usage.) The Node SDK uses
    // snake_case on the wire (output_index, call_id) — check both for safety.
    try {
      const etype = chunk.type ?? "";
      const idx = chunk.output_index ?? chunk.outputIndex;
      if (etype === "response.output_item.added") {
        const item = chunk.item;
        if (item == null || idx == null) return;
        const itype = item.type ?? "";
        let slot: any;
        if (itype === "function_call") {
          slot = {
            type: "function_call",
            name: item.name ?? "",
            callId: item.call_id ?? item.callId ?? "",
            argParts: [] as string[],
          };
        } else if (itype === "message") {
          slot = { type: "message", textParts: [] as string[] };
        } else if (itype === "reasoning") {
          slot = { type: "reasoning" };
        } else if (itype === "image_generation_call") {
          // Part 2: retain id/status/result when present so stream
          // composition + child image spans can see the billed image.
          slot = {
            type: "image_generation_call",
            id: item.id ?? "",
            status: item.status ?? "",
            result: item.result ?? null,
          };
        } else {
          slot = { type: itype };
        }
        acc.items[idx] = slot;
        if (!acc.order.includes(idx)) acc.order.push(idx);
        return;
      }
      // Full image item (incl. b64 result) often arrives on output_item.done.
      if (etype === "response.output_item.done") {
        const item = chunk.item;
        if (item == null || idx == null) return;
        if ((item.type ?? "") === "image_generation_call") {
          acc.items[idx] = {
            type: "image_generation_call",
            id: item.id ?? "",
            status: item.status ?? "completed",
            result: item.result ?? null,
          };
          if (!acc.order.includes(idx)) acc.order.push(idx);
        }
        return;
      }
      if (etype === "response.output_text.delta") {
        const delta = chunk.delta;
        if (idx == null || typeof delta !== "string" || !delta) return;
        if (!acc.items[idx]) {
          acc.items[idx] = { type: "message", textParts: [] };
          if (!acc.order.includes(idx)) acc.order.push(idx);
        }
        (acc.items[idx].textParts ??= []).push(delta);
        return;
      }
      if (etype === "response.function_call_arguments.delta") {
        const delta = chunk.delta;
        if (idx == null || typeof delta !== "string" || !delta) return;
        if (!acc.items[idx]) {
          acc.items[idx] = {
            type: "function_call",
            name: "",
            callId: "",
            argParts: [],
          };
          if (!acc.order.includes(idx)) acc.order.push(idx);
        }
        (acc.items[idx].argParts ??= []).push(delta);
        return;
      }
    } catch {
      // fail-open: composition is best-effort
    }
    return;
  }

  // Vercel AI SDK — LanguageModel V2/V3 stream parts:
  // { type: "text-start", id }
  // { type: "text-delta", id, delta }
  // { type: "text-end", id }
  // { type: "tool-input-start", id, toolName }
  // { type: "tool-input-delta", id, delta }
  // { type: "tool-input-end", id }
  // { type: "tool-call", toolCallId, toolName, input }
  // { type: "reasoning-start" | "reasoning-delta" | "reasoning-end", ... }
  // { type: "finish", finishReason, usage }
  // We accumulate text by id, tool-input fragments by id (mapped to a
  // toolCall slot), and prefer the consolidated `tool-call` part when emitted.
  if (_isAiSdkParse(provider)) {
    try {
      const t = chunk?.type;
      if (t === "text-delta" && typeof chunk.delta === "string" && chunk.delta) {
        const id = chunk.id ?? "_default";
        if (!acc.textParts[id]) {
          acc.textParts[id] = [];
          acc.textOrder.push(id);
        }
        acc.textParts[id].push(chunk.delta);
        return;
      }
      if (t === "tool-input-start") {
        const id = chunk.id ?? chunk.toolCallId ?? "";
        if (!id) return;
        if (!acc.toolCalls[id]) {
          acc.toolCalls[id] = { toolName: chunk.toolName ?? "", argParts: [] };
        } else if (chunk.toolName) {
          acc.toolCalls[id].toolName = chunk.toolName;
        }
        return;
      }
      if (t === "tool-input-delta" && typeof chunk.delta === "string" && chunk.delta) {
        const id = chunk.id ?? chunk.toolCallId ?? "";
        if (!id) return;
        if (!acc.toolCalls[id]) acc.toolCalls[id] = { toolName: "", argParts: [] };
        acc.toolCalls[id].argParts.push(chunk.delta);
        return;
      }
      if (t === "tool-call") {
        // Consolidated tool call — overrides any accumulated argParts.
        const id = chunk.toolCallId ?? chunk.id ?? "";
        if (!id) return;
        if (!acc.toolCalls[id]) acc.toolCalls[id] = { toolName: "", argParts: [] };
        acc.toolCalls[id].toolName = chunk.toolName ?? acc.toolCalls[id].toolName;
        // `input` is already the parsed JSON object — serialize back for the
        // composition hash so callers see the same payload they would in
        // doGenerate's response.
        try {
          acc.toolCalls[id].complete = JSON.stringify(chunk.input ?? {});
        } catch {
          acc.toolCalls[id].complete = String(chunk.input ?? "");
        }
        return;
      }
      if (t === "reasoning-delta" && typeof chunk.delta === "string" && chunk.delta) {
        acc.reasoningParts.push(chunk.delta);
        return;
      }
    } catch {
      // fail-open: composition is best-effort
    }
    return;
  }

  // Mistral — OpenAI-shaped delta chunks but in camelCase, and each event is
  // wrapped as `{ data: CompletionChunk }`. Unwrap once; the delta carries
  // `content` + `toolCalls[]` (each with `id`, `function.{name,arguments}`,
  // `index`).
  if (provider === "mistral") {
    const inner = chunk.data ?? chunk;
    const delta = inner.choices?.[0]?.delta;
    if (!delta) return;
    if (typeof delta.content === "string" && delta.content) {
      acc.textParts.push(delta.content);
    }
    const toolCalls = delta.toolCalls ?? delta.tool_calls;
    if (Array.isArray(toolCalls)) {
      for (const tc of toolCalls) {
        const idx = tc.index ?? 0;
        if (!acc.toolCalls[idx]) {
          acc.toolCalls[idx] = {
            id: "",
            type: "function",
            function: { name: "", arguments: "" },
          };
        }
        const slot = acc.toolCalls[idx];
        if (tc.id) slot.id = tc.id;
        const fn = tc.function;
        if (fn) {
          if (fn.name) slot.function.name = fn.name;
          if (fn.arguments) slot.function.arguments += fn.arguments;
        }
      }
    }
    return;
  }

  // HuggingFace + Together + OpenAI + LiteLLM + Groq + Cerebras — OpenAI-shaped
  // delta chunks: choices[0].delta.content and choices[0].delta.tool_calls
  // (each delta tool call carries an `index`).
  if (
    provider === "huggingface" ||
    provider === "together" ||
    provider === "openai" ||
    provider === "litellm" ||
    provider === "groq" ||
    provider === "cerebras"
  ) {
    const delta = chunk.choices?.[0]?.delta;
    if (!delta) return;
    if (delta.content) acc.textParts.push(delta.content);
    const toolCalls = delta.tool_calls;
    if (Array.isArray(toolCalls)) {
      for (const tc of toolCalls) {
        const idx = tc.index ?? 0;
        if (!acc.toolCalls[idx]) {
          acc.toolCalls[idx] = {
            id: "",
            type: "function",
            function: { name: "", arguments: "" },
          };
        }
        const slot = acc.toolCalls[idx];
        if (tc.id) slot.id = tc.id;
        const fn = tc.function;
        if (fn) {
          if (fn.name) slot.function.name = fn.name;
          if (fn.arguments) slot.function.arguments += fn.arguments;
        }
      }
    }
    return;
  }

  if (provider === "anthropic") {
    // Same event walk as `_tapAnthropicStreamBypass`: content_block_start
    // opens a text / tool_use slot; text_delta and input_json_delta append.
    // Other block types (thinking, …) are kept as opaque slots and dropped at
    // rebuild — parity with the non-streamed anthropic response branch.
    try {
      const etype = chunk?.type;
      if (etype === "content_block_start") {
        const idx = Number(chunk.index ?? 0);
        const cb: any = chunk.content_block ?? {};
        acc.blocks[idx] =
          cb.type === "tool_use"
            ? { type: "tool_use", id: cb.id ?? "", name: cb.name ?? "", inputJson: "" }
            : cb.type === "text"
              ? { type: "text", text: typeof cb.text === "string" ? cb.text : "" }
              : { type: String(cb.type ?? "unknown") };
      } else if (etype === "content_block_delta") {
        const idx = Number(chunk.index ?? 0);
        const d: any = chunk.delta ?? {};
        // Lazy slot for defensive robustness (delta without a start).
        const slot =
          acc.blocks[idx] ??
          (acc.blocks[idx] =
            d.type === "input_json_delta"
              ? { type: "tool_use", id: "", name: "", inputJson: "" }
              : { type: "text", text: "" });
        if (d.type === "text_delta" && typeof d.text === "string") {
          slot.text = (slot.text ?? "") + d.text;
        } else if (d.type === "input_json_delta" && typeof d.partial_json === "string") {
          slot.inputJson = (slot.inputJson ?? "") + d.partial_json;
        }
      }
    } catch {
      // fail-open: composition is best-effort
    }
    return;
  }

  if (provider !== "cohere") return;
  // Cohere v2 stream events: content-delta / tool-plan-delta /
  // tool-call-start / tool-call-delta. The TS SDK camelCases keys
  // (toolCalls/toolPlan); raw-HTTP SSE consumers (tp.protect manual REST)
  // feed the wire-format snake_case (tool_calls/tool_plan) — read both, or
  // tool-call-only streamed turns register no content (TTFT never marked,
  // composition falls back).
  const etype = chunk.type;
  const message = chunk.delta?.message;
  if (etype === "content-delta") {
    const frag = message?.content?.text;
    if (frag) acc.textParts.push(frag);
  } else if (etype === "tool-plan-delta") {
    const frag = message?.toolPlan ?? message?.tool_plan;
    if (frag) acc.toolPlanParts.push(frag);
  } else if (etype === "tool-call-start") {
    const idx = chunk.index ?? 0;
    const tc = message?.toolCalls ?? message?.tool_calls;
    if (tc) {
      acc.toolCalls[idx] = {
        id: tc.id ?? "",
        type: "function",
        function: {
          name: tc.function?.name ?? "",
          arguments: tc.function?.arguments ?? "",
        },
      };
    }
  } else if (etype === "tool-call-delta") {
    const idx = chunk.index ?? 0;
    const tcd = message?.toolCalls ?? message?.tool_calls;
    const frag = tcd?.function?.arguments;
    if (frag && acc.toolCalls[idx]) {
      acc.toolCalls[idx].function.arguments += frag;
    }
  }
}

/**
 * Builds a synthetic response object from the accumulator so buildResponse
 * composition can parse it. Returns null when nothing was accumulated.
 */
function _streamAccumulatorToResponse(provider: string, acc: any): any {
  if (!acc) return null;

  if (provider === "openai_responses") {
    const out: any[] = [];
    const indices: number[] =
      acc.order.length > 0
        ? acc.order
        : Object.keys(acc.items).map(Number).sort((a, b) => a - b);
    for (const idx of indices) {
      const slot = acc.items[idx];
      if (!slot) continue;
      if (slot.type === "message") {
        const text = (slot.textParts ?? []).join("");
        if (text) {
          out.push({
            type: "message",
            content: [{ type: "output_text", text }],
          });
        }
      } else if (slot.type === "function_call") {
        out.push({
          type: "function_call",
          name: slot.name ?? "",
          call_id: slot.callId ?? "",
          arguments: (slot.argParts ?? []).join(""),
        });
      } else if (slot.type === "reasoning") {
        out.push({ type: "reasoning" });
      } else if (slot.type === "image_generation_call") {
        // Part 2 / composition: surface built-in image tool output so
        // stream composition matches non-stream (assistant/image entry).
        out.push({
          type: "image_generation_call",
          id: slot.id ?? "",
          status: slot.status ?? "completed",
          result: slot.result ?? null,
        });
      }
    }
    if (out.length === 0) return null;
    return { output: out };
  }

  // HuggingFace / Mistral / Together / OpenAI / LiteLLM / Groq / Cerebras —
  // build a synthetic OpenAI-shaped response so buildResponseComposition's
  // `response.choices` branch can parse it. Mistral stream chunks are wrapped
  // in `.data` but the synthesized response we hand to composition is plain
  // OpenAI-shape.
  if (
    provider === "huggingface" ||
    provider === "mistral" ||
    provider === "together" ||
    provider === "openai" ||
    provider === "litellm" ||
    provider === "groq" ||
    provider === "cerebras"
  ) {
    const message: any = { role: "assistant" };
    const text = acc.textParts.join("");
    if (text) message.content = text;
    const toolCalls = Object.keys(acc.toolCalls)
      .map(Number)
      .sort((a, b) => a - b)
      .map((k) => acc.toolCalls[k]);
    if (toolCalls.length > 0) message.tool_calls = toolCalls;
    if (message.content === undefined && message.tool_calls === undefined) {
      return null;
    }
    return { choices: [{ message }] };
  }

  if (_isAiSdkParse(provider)) {
    // Build a synthetic LanguageModel doGenerate-shaped response so the AI SDK
    // branch in buildResponseComposition can parse it uniformly.
    const content: any[] = [];
    if (acc.reasoningParts.length > 0) {
      const text = acc.reasoningParts.join("");
      if (text) content.push({ type: "reasoning", text });
    }
    for (const id of acc.textOrder) {
      const text = (acc.textParts[id] || []).join("");
      if (text) content.push({ type: "text", text });
    }
    for (const id of Object.keys(acc.toolCalls)) {
      const slot = acc.toolCalls[id];
      const args = slot.complete ?? slot.argParts.join("");
      content.push({
        type: "tool-call",
        toolCallId: id,
        toolName: slot.toolName,
        input: args,
      });
    }
    if (content.length === 0) return null;
    return { content };
  }

  if (provider === "anthropic") {
    // Rebuild anthropic-shaped content blocks in stream order so
    // buildResponseComposition's anthropic branch (text → assistant entry,
    // tool_use → tool_call entry with canonical-JSON input) and
    // extractPendingToolCalls (tool_use ids) treat it like a non-streamed
    // Message. Null while no renderable block exists (keeps
    // `_streamAccHasContent` / TTFT semantics: first text or tool_use block).
    const content: any[] = [];
    for (const k of Object.keys(acc.blocks ?? {})
      .map(Number)
      .sort((a, b) => a - b)) {
      const b = acc.blocks[k];
      if (b?.type === "text" && typeof b.text === "string" && b.text) {
        content.push({ type: "text", text: b.text });
      } else if (b?.type === "tool_use") {
        let input: unknown = {};
        try {
          input = b.inputJson ? JSON.parse(b.inputJson) : {};
        } catch {
          input = b.inputJson; // unparsable partial JSON — keep raw string
        }
        content.push({ type: "tool_use", id: b.id ?? "", name: b.name ?? "", input });
      }
    }
    if (content.length === 0) return null;
    return { content };
  }

  if (provider !== "cohere") return null;
  const message: any = {};
  const text = acc.textParts.join("");
  if (text) message.content = [{ type: "text", text }];
  const toolPlan = acc.toolPlanParts.join("");
  if (toolPlan) message.toolPlan = toolPlan;
  const toolCalls = Object.keys(acc.toolCalls)
    .map(Number)
    .sort((a, b) => a - b)
    .map((k) => acc.toolCalls[k]);
  if (toolCalls.length > 0) message.toolCalls = toolCalls;
  if (Object.keys(message).length === 0) return null;
  return { message };
}

// ── Latency capture (TTFT + streaming throughput) ──────────────────────────
// Mirrors the Python SDK. The stream wrapper timestamps the first *content*
// chunk to mark Time-To-First-Token, anchored at the monotonic provider-call
// start. All deltas use performance.now() (monotonic, ms) and are clamped
// non-negative so a clock anomaly can never produce a negative value.

interface TPLatency {
  is_streaming: boolean;
  ttft_ms: number | null;
  total_ms: number;
  generation_ms: number | null;
  output_tokens: number | null;
  clock: "monotonic";
}

/**
 * True once the accumulator holds renderable content (assistant text or a
 * tool-call delta). Reuses the accumulator→response builder so it can never
 * drift from the per-provider parsing. When the provider isn't accumulated
 * (acc == null) we best-effort treat the first chunk as first content.
 */
function _streamAccHasContent(provider: string, acc: any): boolean {
  if (acc == null) return true;
  try {
    return _streamAccumulatorToResponse(provider, acc) != null;
  } catch {
    return false;
  }
}

/**
 * Assemble the SDK latency primitives from monotonic timestamps (ms). Returns
 * null on failure or a missing anchor (telemetry loss, never a thrown error).
 * output_tokens is left null — the server prefers its own mapped count.
 */
function _buildStreamLatency(
  reqStartMono: number | undefined,
  ttftMono: number | null,
  endMono: number,
): TPLatency | null {
  try {
    if (reqStartMono == null) return null;
    const totalMs = Math.round(Math.max(0, endMono - reqStartMono));
    let ttftMs: number | null = null;
    let generationMs: number | null = null;
    if (ttftMono != null) {
      ttftMs = Math.round(Math.max(0, ttftMono - reqStartMono));
      generationMs = Math.round(Math.max(0, endMono - ttftMono));
    }
    return {
      is_streaming: true,
      ttft_ms: ttftMs,
      total_ms: totalMs,
      generation_ms: generationMs,
      output_tokens: null,
      clock: "monotonic",
    };
  } catch {
    return null;
  }
}

/**
 * Wraps a manual-telemetry streaming response (an async iterable of chunks) so
 * the final chunk's token usage is captured and logged once the consumer
 * finishes iterating. Response composition is accumulated from the streamed
 * chunks (the final usage chunk alone does not carry the message content).
 * Fully fail-open — yields every original chunk untouched.
 */
async function _wrapManualStream(
  stream: any,
  provider: string,
  args: any[],
  order: number,
  spanName: string | null,
  startTime: Date,
  reqStartMono?: number,
  // True when the wrapper injected stream_options.include_usage on this call
  // (the customer did NOT ask): the synthetic usage-only terminal chunk is
  // tapped for metering but stripped from the customer-visible iteration so
  // their code sees exactly the chunks it asked for.
  suppressUsageChunk = false,
): Promise<any> {
  if (!stream || typeof stream[Symbol.asyncIterator] !== "function") {
    // Not iterable — nothing to wrap; pass through.
    return stream;
  }
  try {
    // Preserve the native provider-stream surface. For Stainless `Stream`
    // objects (OpenAI Responses / Groq / Cerebras / Together) the value carries
    // `.tee()`, `.controller`, `.toReadableStream()` and class private (`#`)
    // fields — returning a bare async generator (the old behavior) dropped all
    // of them, and a Proxy would break the private-field brand check. So we
    // override ONLY `Symbol.asyncIterator` IN PLACE on the real object and hand
    // the SAME object back (identity + every member intact) — mirroring the
    // proven `_tapStreamUsageForOnEnd`.
    //
    // Capture the ORIGINAL iterator BEFORE installing the override, then drive
    // it via `.next()` — never `for await … of` it and never re-read
    // `stream[Symbol.asyncIterator]` from INSIDE the override. For a bare async
    // generator (HF/Gemini) `getOrig()` returns the generator ITSELF (its
    // `[Symbol.asyncIterator]()` yields `this`), so a `for await … of getOrig()`
    // would re-read the now-overridden `Symbol.asyncIterator` and self-recurse
    // infinitely. Advancing the iterator directly (mirror the proven
    // `_tapStreamUsageForOnEnd`: bind BEFORE install, `getOrig()` once, then
    // `inner.next()`) is the ONLY safe chunk source.
    const getOrig = stream[Symbol.asyncIterator].bind(stream);
    // Captured obs key for the drain-time log calls below: the generator body
    // is resumed from the CONSUMER's async context (ALS does not survive
    // external async-generator resumption), so the key is captured here —
    // still inside the wrapper's per-call obs scope — and re-entered around
    // the log calls via runWithObsKey.
    const _obsKey = _currentObsKey();
    // Snapshot the attempt context at wrap time (mirrors _wrapBedrockConverseStream).
    // _stashAttemptContext ran for THIS call just before the provider call, but
    // the _attempted_* slots are session-global — a later call on the same
    // session may overwrite them before this stream fails.
    let _attemptSnap: {
      model: unknown;
      provider: unknown;
      operation: unknown;
      shape: unknown;
      wireKey: unknown;
    } | null = null;
    try {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const s = getCurrentSession() as any;
      _attemptSnap = {
        model: s._attempted_model,
        provider: s._attempted_provider,
        operation: s._attempted_operation,
        shape: s._attempted_shape,
        wireKey: s._attempted_wire_key,
      };
    } catch {
      // fail-open: no snapshot → the catch below emits off the live slots,
      // exactly the pre-snapshot behavior.
    }
    Object.defineProperty(stream, Symbol.asyncIterator, {
      configurable: true,
      writable: true,
      value: async function* () {
        let lastUsageChunk: any;
        const acc = _newStreamAccumulator(provider);
        const streamStartMono = performance.now();
        let ttftMono: number | null = null;
        let streamFailed = false;
        const inner: AsyncIterator<any> = getOrig();
        try {
          for (;;) {
            const r = await inner.next();
            if (r.done) break;
            const chunk = r.value;
            // Tap is best-effort: a malformed chunk (provider chunk-shape drift)
            // must cost telemetry, never abort the customer's iteration. (Python
            // already guards this per-chunk; Node previously did not — this closes
            // that gap so a tap failure degrades to telemetry loss.)
            try {
              if (_chunkHasUsage(provider, chunk)) lastUsageChunk = chunk;
              _accumulateStreamChunk(provider, acc, chunk);
              // Mark TTFT at the first chunk carrying renderable content.
              if (ttftMono === null && _streamAccHasContent(provider, acc)) {
                ttftMono = performance.now();
              }
            } catch {
              // fail-open: tap failure never breaks the stream
            }
            // The SDK injected include_usage (the customer didn't ask) — strip
            // the synthetic usage-only terminal chunk (usage set, empty
            // choices) after tapping it, so the customer's iteration is
            // byte-identical to what they requested. A usage payload riding a
            // CONTENT chunk (what some Together replicas send unasked) is
            // never stripped. Guarded: a check failure yields the chunk.
            if (suppressUsageChunk) {
              let _skip = false;
              try {
                _skip = _isUsageOnlyChunk(chunk);
              } catch {
                _skip = false;
              }
              if (_skip) continue;
            }
            yield chunk;
          }
        } catch (err) {
          streamFailed = true;
          const session = getCurrentSession();
          const elapsedMs = Math.round(Math.max(0, performance.now() - streamStartMono));
          try {
            (session as any)._call_outcome = buildCallOutcome(err, elapsedMs);
          } catch {
            // fail-safe
          }
          // Restore this call's attempt context if a later same-session call
          // overwrote the session-global slots. Save the current values first:
          // when they differ from the snapshot they belong to a newer in-flight
          // call — hand them back after the emit (which clears the slots) so
          // that call's failure row isn't degraded to model="unknown".
          let cur: typeof _attemptSnap = null;
          let overwritten = false;
          try {
            if (_attemptSnap) {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              const s = session as any;
              cur = {
                model: s._attempted_model,
                provider: s._attempted_provider,
                operation: s._attempted_operation,
                shape: s._attempted_shape,
                wireKey: s._attempted_wire_key,
              };
              overwritten =
                cur.model !== _attemptSnap.model ||
                cur.provider !== _attemptSnap.provider ||
                cur.operation !== _attemptSnap.operation ||
                cur.shape !== _attemptSnap.shape ||
                cur.wireKey !== _attemptSnap.wireKey;
              s._attempted_model = _attemptSnap.model;
              s._attempted_provider = _attemptSnap.provider;
              s._attempted_operation = _attemptSnap.operation;
              s._attempted_shape = _attemptSnap.shape;
              s._attempted_wire_key = _attemptSnap.wireKey;
            }
          } catch {
            // fail-open: emit off the live slots, same as before the snapshot
          }
          // Re-enter the captured obs scope so the failure drain claims this
          // call's own observations (see _obsKey capture above).
          _reenterObsScope(_obsKey, () =>
            _emitCallFailureLog(getClient(), session),
          );
          // _emitCallFailureLog cleared the _attempted_* slots; hand a newer
          // call's context back if we borrowed them.
          try {
            if (overwritten && cur) {
              // eslint-disable-next-line @typescript-eslint/no-explicit-any
              const s = session as any;
              s._attempted_model = cur.model;
              s._attempted_provider = cur.provider;
              s._attempted_operation = cur.operation;
              s._attempted_shape = cur.shape;
              s._attempted_wire_key = cur.wireKey;
            }
          } catch {
            // fail-open: failure logging must never affect customer error propagation
          }
          throw err;
        } finally {
          if (!streamFailed) {
            // Propagate an early customer break/return to the underlying
            // provider iterator so it can release resources — the manual
            // `.next()` loop doesn't auto-close the way `for await … of` did.
            // Idempotent on a fully-drained iterator; fail-open on any throw.
            try {
              await inner.return?.();
            } catch {
              // fail-open: closing the source stream must never throw out
            }
          }
          if (!streamFailed) {
            try {
              if (lastUsageChunk) {
                const latency = _buildStreamLatency(reqStartMono, ttftMono, performance.now());
                // Response composition comes from the accumulated stream; usage and
                // model still come from the final usage chunk.
                const responseForComp =
                  _streamAccumulatorToResponse(provider, acc) ?? lastUsageChunk;
                _captureCompositionAt(provider, args, responseForComp, order);
                // Gemini TTS stream: same reclass as non-stream (was hardcoded
                // "chat" here and left TTS rows misclassified).
                const streamOp = _resolveGoogleLogOperation(
                  provider,
                  "chat",
                  args,
                  lastUsageChunk,
                );
                // Re-enter the captured obs scope so _logManual's keyed drain
                // claims this call's own observations (the finally runs in
                // the consumer's context, outside the original scope).
                _reenterObsScope(_obsKey, () =>
                  _logManual(
                    provider,
                    args,
                    lastUsageChunk,
                    order,
                    spanName,
                    startTime,
                    streamOp,
                    null,
                    latency,
                  ),
                );
              } else if (provider === "together") {
                // G5-1 fallback: the stream drained cleanly but carried NO
                // usage payload at all (replica-dependent on Together, and
                // still reachable with injection on: captureStreamUsage=false,
                // the strip-and-retry path, or a replica that accepts the
                // option and omits the chunk anyway). Losing the row silently
                // is the worst outcome — a customer under-sees real spend
                // with no signal anywhere. Log the row with token counts
                // approximated from the request messages + accumulated
                // response (chars/4, the SDK's established estimator) and
                // raw.approximated=true, so the collector prices it as
                // cost_status='approximated' — never a fake exact 'measured',
                // never a silent zero. Gated to together: other manual
                // providers' usage extraction is not OpenAI-shaped, and only
                // together has evidenced this omission class.
                const latency = _buildStreamLatency(reqStartMono, ttftMono, performance.now());
                const responseForComp = _streamAccumulatorToResponse(provider, acc);
                if (responseForComp) {
                  _captureCompositionAt(provider, args, responseForComp, order);
                }
                const synthetic = {
                  ...(responseForComp && typeof responseForComp === "object"
                    ? responseForComp
                    : {}),
                  usage: {
                    prompt_tokens: _approximateTokensFromChars(
                      JSON.stringify(args?.[0]?.messages ?? "") ?? "",
                    ),
                    completion_tokens: _approximateTokensFromChars(
                      JSON.stringify(
                        (responseForComp as any)?.choices?.[0]?.message ?? "",
                      ) ?? "",
                    ),
                    approximated: true,
                  },
                };
                _reenterObsScope(_obsKey, () =>
                  _logManual(
                    provider,
                    args,
                    synthetic,
                    order,
                    spanName,
                    startTime,
                    "chat",
                    null,
                    latency,
                  ),
                );
              }
            } catch {
              // fail-open: never throw out of a wrapped stream
            }
          }
        }
      },
    });
    return stream;
  } catch {
    // fail-open: the override could not be installed — the object is
    // frozen/sealed or `Symbol.asyncIterator` is a non-configurable own
    // property. Hand back the original native stream untapped (surface intact,
    // iteration works) rather than a broken/half-tapped value or a bare
    // generator. Telemetry loss is acceptable; breaking the customer's stream
    // is not.
    return stream;
  }
}

/**
 * Wraps an AWS Bedrock `ConverseStream` output (`{ stream, $metadata }`) so the
 * streamed token usage — delivered on the terminal `{ metadata: { usage } }`
 * event nested inside `.stream` — is logged exactly once when the customer
 * finishes draining the stream, via the SAME `_logManual` the non-streaming
 * `ConverseCommand` path uses. The generic `_wrapManualStream` can't handle
 * this shape: the top-level output object is NOT async-iterable (only `.stream`
 * is), so it would never observe the usage event.
 *
 * Non-destructive: ONLY `result.stream` is replaced (with a lazy tap-and-log
 * async generator); `result.$metadata` + every other own key + the object
 * identity are preserved (mirrors `_tapStreamUsageForOnEnd`, never a bare
 * generator). Fully fail-open, exactly like `_wrapManualStream`: yields every
 * original chunk untouched and one at a time (no buffering), swallows any
 * tap/log failure, and re-throws ONLY a genuine provider error (after emitting
 * a failure log). Logs ZERO rows when no terminal usage event is seen (no
 * phantom zero-usage row).
 */
function _wrapBedrockConverseStream(
  result: any,
  provider: string,
  args: any[],
  order: number,
  spanName: string | null,
  startTime: Date,
  reqStartMono?: number,
): any {
  const source = result?.stream;
  if (!source || typeof source[Symbol.asyncIterator] !== "function") {
    // Not a stream — nothing to tap; hand the output back untouched.
    return result;
  }
  // Captured obs key for the drain-time log calls in the generator below —
  // resumed from the consumer's context, outside this call's obs scope
  // (mirrors _wrapManualStream).
  const _obsKey = _currentObsKey();
  // Snapshot the attempt context at wrap time (mirrors _tapStreamUsageForOnEnd).
  // _stashAttemptContext ran for THIS call just before the provider call, but
  // the _attempted_* slots are session-global — a later call on the same
  // session may overwrite them before this stream fails.
  let _attemptSnap: {
    model: unknown;
    provider: unknown;
    operation: unknown;
    shape: unknown;
    wireKey: unknown;
  } | null = null;
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const s = getCurrentSession() as any;
    _attemptSnap = {
      model: s._attempted_model,
      provider: s._attempted_provider,
      operation: s._attempted_operation,
      shape: s._attempted_shape,
      wireKey: s._attempted_wire_key,
    };
  } catch {
    // fail-open: no snapshot → the catch below emits off the live slots,
    // exactly the pre-snapshot behavior.
  }
  result.stream = (async function* () {
    let lastUsageChunk: any;
    const acc = _newStreamAccumulator(provider);
    const streamStartMono = performance.now();
    let ttftMono: number | null = null;
    let streamFailed = false;
    try {
      for await (const chunk of source) {
        // Tap is best-effort: a malformed event (Bedrock chunk-shape drift)
        // must cost telemetry, never abort the customer's iteration.
        try {
          if (_chunkHasUsage(provider, chunk)) lastUsageChunk = chunk;
          _accumulateStreamChunk(provider, acc, chunk);
          if (ttftMono === null && _streamAccHasContent(provider, acc)) {
            ttftMono = performance.now();
          }
        } catch {
          // fail-open: tap failure never breaks the stream
        }
        // Lazy passthrough: yield each chunk immediately, driving the source
        // exactly one chunk ahead of the consumer — never buffer/drain first.
        yield chunk;
      }
    } catch (err) {
      // Genuine provider error mid-stream — emit a failure log, then re-throw
      // the customer's OWN error unchanged (mirrors _wrapManualStream).
      streamFailed = true;
      const session = getCurrentSession();
      const elapsedMs = Math.round(Math.max(0, performance.now() - streamStartMono));
      try {
        (session as any)._call_outcome = buildCallOutcome(err, elapsedMs);
      } catch {
        // fail-safe
      }
      // Restore this call's attempt context if a later same-session call
      // overwrote the session-global slots. Save the current values first:
      // when they differ from the snapshot they belong to a newer in-flight
      // call — hand them back after the emit (which clears the slots) so
      // that call's failure row isn't degraded to model="unknown".
      let cur: typeof _attemptSnap = null;
      let overwritten = false;
      try {
        if (_attemptSnap) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const s = session as any;
          cur = {
            model: s._attempted_model,
            provider: s._attempted_provider,
            operation: s._attempted_operation,
            shape: s._attempted_shape,
            wireKey: s._attempted_wire_key,
          };
          overwritten =
            cur.model !== _attemptSnap.model ||
            cur.provider !== _attemptSnap.provider ||
            cur.operation !== _attemptSnap.operation ||
            cur.shape !== _attemptSnap.shape ||
            cur.wireKey !== _attemptSnap.wireKey;
          s._attempted_model = _attemptSnap.model;
          s._attempted_provider = _attemptSnap.provider;
          s._attempted_operation = _attemptSnap.operation;
          s._attempted_shape = _attemptSnap.shape;
          s._attempted_wire_key = _attemptSnap.wireKey;
        }
      } catch {
        // fail-open: emit off the live slots, same as before the snapshot
      }
      // Re-enter the captured obs scope so the failure drain claims this
      // call's own observations (see _obsKey capture above).
      _reenterObsScope(_obsKey, () =>
        _emitCallFailureLog(getClient(), session),
      );
      // _emitCallFailureLog cleared the _attempted_* slots; hand a newer
      // call's context back if we borrowed them.
      try {
        if (overwritten && cur) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const s = session as any;
          s._attempted_model = cur.model;
          s._attempted_provider = cur.provider;
          s._attempted_operation = cur.operation;
          s._attempted_shape = cur.shape;
          s._attempted_wire_key = cur.wireKey;
        }
      } catch {
        // fail-open: failure logging must never affect customer error propagation
      }
      throw err;
    } finally {
      if (!streamFailed) {
        try {
          // Only log when a terminal usage event was actually seen — never emit
          // a phantom zero-usage row.
          if (lastUsageChunk) {
            const latency = _buildStreamLatency(reqStartMono, ttftMono, performance.now());
            // Response composition comes from the accumulated stream (best-
            // effort); usage + model come from the terminal metadata event via
            // the bedrock branch of _extractUsage (reads metadata.usage).
            const responseForComp =
              _streamAccumulatorToResponse(provider, acc) ?? lastUsageChunk;
            _captureCompositionAt(provider, args, responseForComp, order);
            // Re-enter the captured obs scope for the keyed drain inside
            // _logManual (finally runs in the consumer's context).
            _reenterObsScope(_obsKey, () =>
              _logManual(
                provider,
                args,
                lastUsageChunk,
                order,
                spanName,
                startTime,
                "chat",
                null,
                latency,
              ),
            );
          }
        } catch {
          // fail-open: never throw out of a wrapped stream
        }
      }
    }
  })();
  return result;
}

/**
 * Non-destructively taps a streaming result to harvest token usage and stash it
 * for telemetry.onEnd. Used on the Traceloop (non-manual) path — primarily the
 * OpenAI SDK pointed at an OpenAI-compatible base_url — whose streaming spans
 * carry NO `gen_ai.usage.*` (the instrumentor's success hook fires before the
 * stream is consumed), so onEnd would drop the row.
 *
 * Unlike `_wrapManualStream` (which replaces the stream with a bare async
 * generator), this overrides ONLY `result[Symbol.asyncIterator]`: the Stream
 * object's identity and every other method (`.tee()`, `.toReadableStream()`,
 * `.controller`, …) are preserved, so customer code that uses them is
 * unaffected. Requires `stream_options.include_usage` so the provider emits a
 * final usage chunk.
 *
 * The stash is keyed by the SAME compKey `_stashApiBase` uses
 * (`traceId:spanCounter`) — the order the instrumentor span's onStart consumes —
 * so telemetry.onEnd finds it. Fail-open everywhere: any error returns the
 * stream untouched, and if usage never arrives onEnd simply skips the row
 * (status quo) — never worse than before.
 */
/**
 * Inject `stream_options.include_usage` into an OpenAI-wire chat-completions
 * stream request that didn't ask for it, so the provider emits the final usage
 * chunk and the streamed call's cost isn't silently lost. Scope: the openai SDK
 * parse key (custom base_urls — MiniMax, Groq, vLLM — ride the same client)
 * plus the native together-ai SDK's manual-telemetry path — Together honors the
 * option (their API is OpenAI-wire) and, without it, whether a stream carries a
 * usage payload at all is REPLICA-DEPENDENT on Together's side (G5-1: some
 * serving backends attach usage to the final chunk unasked, others omit it
 * entirely → the streamed row was silently lost 30–50% of the time). Only
 * `messages`-bearing bodies qualify (the Responses API streams usage without
 * any opt-in). The synthetic usage-only terminal chunk is stripped from the
 * customer-visible iterator (openai: _tapStreamUsageForOnEnd; together:
 * _wrapManualStream's suppressUsageChunk), so customer code sees exactly the
 * chunks it asked for. Disable with `captureStreamUsage: false` or
 * TP_CAPTURE_STREAM_USAGE=0.
 *
 * Returns a restore handle when injected (so the injected option can be undone
 * on retry and never leaks back into the customer's body), else null. The
 * handle is invoked on every exit of _callWithInjectedStreamUsage. Fail-safe.
 */
function _injectStreamUsageOption(
  provider: string,
  args: any[],
): { restore: () => void } | null {
  try {
    const tp = getClient();
    if (!tp || (tp as any).captureStreamUsage === false) return null;
    if (provider !== "openai" && provider !== "together") return null;
    const body = args && args[0];
    if (!body || typeof body !== "object") return null;
    if (body.stream !== true || !Array.isArray(body.messages)) return null;
    const prior = body.stream_options;
    if (prior && prior.include_usage === true) return null; // customer asked — chunk is theirs
    body.stream_options = { ...(prior || {}), include_usage: true };
    return {
      restore: () => {
        try {
          if (prior === undefined) delete body.stream_options;
          else body.stream_options = prior;
        } catch {
          // fail-safe
        }
      },
    };
  } catch {
    return null;
  }
}

/**
 * Calls the wrapped SDK method with the include_usage injection. Strip-and-retry
 * only when the rejection is plausibly caused by the injected option — a
 * strict-compat server answering 400/422 or naming the param — rejected before
 * any generation, so a retry cannot double-generate. A rate-limit or auth
 * failure must never trigger a second provider request. The customer's body is
 * restored on every exit (success, retry, rethrow) so the injected option never
 * leaks back into their object; the injection can never fail a call that would
 * otherwise have succeeded.
 */
/**
 * Manual telemetry for an Anthropic raw event stream on the Traceloop-bypass
 * path (see `anthropicStreamBypass` in _wrapMethod). No OTel span exists for
 * the call, so this tap accumulates usage from the stream events and logs a
 * row directly when the stream drains:
 * - `message_start` → model + usage (input_tokens, cache_read/_creation)
 * - `message_delta` → usage (cumulative output_tokens) + stop_reason
 * - `content_block_delta` → response text (composition)
 *
 * Non-destructive: only `[Symbol.asyncIterator]` is overridden — the genuine
 * `Stream` object (`.controller`, `.tee()`, …) is preserved, which
 * `MessageStream._createMessage` requires (it reads `stream.controller`).
 * Fail-open everywhere: a tap failure degrades to telemetry loss, never breaks
 * the customer's iteration. Early termination (`return()`) logs whatever usage
 * was seen, since those tokens were already billed.
 */
function _tapAnthropicStreamBypass(
  stream: any,
  thisArg: any,
  args: any[],
  order: number,
  spanName: string | null,
  startTime: Date,
  session: any,
  reqStartMono?: number,
): any {
  try {
    if (!stream || typeof stream[Symbol.asyncIterator] !== "function") return stream;
    // Kwargs copy carrying the serving endpoint so _logManual's
    // _extractBaseURL(args[0]) pickup emits model_extras.api_base (so
    // downstream pricing can identify the serving endpoint, e.g.
    // api.minimax.io → minimax). Same stash precedent as `__tpXaiModel` /
    // `__tpAiSdk`.
    const kwargsArgs = [{ ...(args[0] || {}), baseURL: _extractBaseURL(thisArg) }];
    // Capture the ambient `.stream()` log latch (if this `create({stream:true})`
    // was internally delegated from a `.stream()` call, that wrapper is ALSO
    // logging this same request — share its latch so exactly one row wins). Null
    // for an independent `create({stream:true})`. Captured here, at tap-setup
    // time, so it reflects the async context of the create call, not of the later
    // drain. See A7(b) / A7(b-reverse).
    let inheritedLatch: { logged: boolean } | null = null;
    try {
      inheritedLatch = _anthropicStreamLatchStorage.getStore() ?? null;
    } catch {
      inheritedLatch = null; // fail-open: treat as an independent call → log
    }
    // Captured obs key for the drain-time calls below (finalize/next run in
    // the consumer's context, outside this call's obs scope). Captured at
    // tap-setup time like the latch above.
    const _obsKey = _currentObsKey();
    const getOrig = stream[Symbol.asyncIterator].bind(stream);
    const acc = {
      model: "",
      usage: {} as Record<string, unknown>,
      // Content blocks keyed by stream index. Text AND tool_use blocks must
      // both be reconstructed: a tool-call-only response with an empty
      // `content` array makes buildResponseComposition's anthropic branch
      // return nothing and fall through to the Tier-3
      // `String(response)` → "[object Object]" fallback (bogus
      // `complete_response` rows). tool_use ids also feed
      // setPendingToolCalls so tp.toolSpan call-id correlation works.
      blocks: {} as Record<number, any>,
      stopReason: null as unknown,
      logged: false,
    };
    const streamStartMono = reqStartMono ?? performance.now();
    // TTFT — marked at the first content block (text OR tool_use). Mirrors
    // _wrapManualStream's "first renderable content" semantics.
    let ttftMono: number | null = null;
    const finalize = (): void => {
      if (acc.logged) return;
      acc.logged = true;
      try {
        const u: any = acc.usage;
        // Mirror the span-path gate: never log zero-usage rows.
        if (!(Number(u.input_tokens) > 0 || Number(u.output_tokens) > 0)) return;
        // Rebuild anthropic-shaped content blocks in stream order. Thinking /
        // unknown block types are dropped — parity with the non-streamed
        // anthropic response branch in buildResponseComposition.
        const content: any[] = [];
        for (const k of Object.keys(acc.blocks)
          .map(Number)
          .sort((a, b) => a - b)) {
          const b = acc.blocks[k];
          if (b?.type === "text" && typeof b.text === "string" && b.text) {
            content.push({ type: "text", text: b.text });
          } else if (b?.type === "tool_use") {
            let input: unknown = {};
            try {
              input = b.inputJson ? JSON.parse(b.inputJson) : {};
            } catch {
              input = b.inputJson; // unparsable partial JSON — keep raw string
            }
            content.push({ type: "tool_use", id: b.id ?? "", name: b.name ?? "", input });
          }
        }
        // Anthropic-Message-shaped synthetic so the existing composition
        // builder + usage extractor treat it like a non-streaming response.
        const synthetic = {
          model: acc.model || String((kwargsArgs[0] as any)?.model ?? "unknown"),
          stop_reason: acc.stopReason,
          usage: acc.usage,
          content,
        };
        // Empty content would fall through to the Tier-3 String(response)
        // fallback in buildResponseComposition — skip composition entirely
        // (empty beats "[object Object]"). Usage still logs below.
        if (content.length > 0) {
          _captureCompositionAt("anthropic", kwargsArgs, synthetic, order);
        }
        // When `client.messages.stream()` delegates to this patched
        // `create({stream:true})` (SDK-version dependent), the `.stream()`
        // context-manager wrapper is ALSO logging this same call. Share its
        // one-shot log latch (inherited via async context at tap-setup time) so
        // exactly one row is emitted (never 2). The take happens AFTER the
        // zero-usage guard above, so a path that skips logging never "takes" the
        // latch → the other path still logs (never 0). No-op for an independent
        // `create({stream:true})` call (no inherited latch).
        if (!_streamLatchTake({ __tpLogLatch: inheritedLatch })) return;
        // Latency primitives (is_streaming/ttft/total) — same builder as the
        // manual stream path; fail-safe (null on a missing anchor).
        const latency = _buildStreamLatency(streamStartMono, ttftMono, performance.now());
        // Re-enter the captured obs scope for _logManual's keyed drain
        // (finalize fires from the consumer's iteration context).
        _reenterObsScope(_obsKey, () =>
          _logManual(
            "anthropic", kwargsArgs, synthetic, order, spanName, startTime, "chat",
            null, latency,
          ),
        );
      } catch {
        // fail-open: telemetry loss only
      }
    };
    Object.defineProperty(stream, Symbol.asyncIterator, {
      configurable: true,
      writable: true,
      value: function (): AsyncIterator<any> {
        const inner: AsyncIterator<any> = getOrig();
        return {
          async next(): Promise<IteratorResult<any>> {
            let r: IteratorResult<any>;
            try {
              r = await inner.next();
            } catch (err) {
              try {
                (session as any)._call_outcome = buildCallOutcome(
                  err,
                  Math.round(Math.max(0, performance.now() - streamStartMono)),
                );
                // Re-enter the captured obs scope so the failure drain claims
                // this call's observations (next() runs in the consumer's
                // context).
                const _failureDispatched = _reenterObsScope(_obsKey, () =>
                  _emitCallFailureLog(getClient(), session),
                );
                // F-17-A: on the delegated-from-`.stream()` case the vendor
                // MessageStream re-delivers this rejection to the consumer's
                // iterator, whose catch would emit a degraded duplicate. Mark
                // the shared latch ONLY after a REAL dispatch so that layer
                // skips (see _streamLatchMarkFailure); no-op for an
                // independent `create({stream:true})` (inheritedLatch null).
                if (_failureDispatched) _streamLatchMarkFailure(inheritedLatch);
              } catch {
                // fail-safe
              }
              throw err;
            }
            try {
              if (!r.done) {
                const ev: any = r.value;
                if (ev?.type === "message_start" && ev.message) {
                  if (ev.message.model) acc.model = String(ev.message.model);
                  if (ev.message.usage && typeof ev.message.usage === "object") {
                    acc.usage = { ...acc.usage, ...ev.message.usage };
                  }
                } else if (ev?.type === "message_delta") {
                  if (ev.usage && typeof ev.usage === "object") {
                    acc.usage = { ...acc.usage, ...ev.usage };
                  }
                  if (ev.delta?.stop_reason) acc.stopReason = ev.delta.stop_reason;
                } else if (ev?.type === "content_block_start") {
                  if (ttftMono === null) ttftMono = performance.now();
                  const idx = Number(ev.index ?? 0);
                  const cb: any = ev.content_block ?? {};
                  acc.blocks[idx] =
                    cb.type === "tool_use"
                      ? {
                          type: "tool_use",
                          id: cb.id ?? "",
                          name: cb.name ?? "",
                          inputJson: "",
                        }
                      : cb.type === "text"
                        ? { type: "text", text: typeof cb.text === "string" ? cb.text : "" }
                        : { type: String(cb.type ?? "unknown") };
                } else if (ev?.type === "content_block_delta") {
                  if (ttftMono === null) ttftMono = performance.now();
                  const idx = Number(ev.index ?? 0);
                  // Lazy slot for defensive robustness (delta without a start).
                  const slot =
                    acc.blocks[idx] ??
                    (acc.blocks[idx] =
                      ev.delta?.type === "input_json_delta"
                        ? { type: "tool_use", id: "", name: "", inputJson: "" }
                        : { type: "text", text: "" });
                  if (ev.delta?.type === "text_delta" && typeof ev.delta.text === "string") {
                    slot.text = (slot.text ?? "") + ev.delta.text;
                  } else if (
                    ev.delta?.type === "input_json_delta" &&
                    typeof ev.delta.partial_json === "string"
                  ) {
                    slot.inputJson = (slot.inputJson ?? "") + ev.delta.partial_json;
                  }
                }
              } else {
                finalize();
              }
            } catch {
              // fail-open: a telemetry tap must never break customer iteration
            }
            return r;
          },
          return: inner.return
            ? (v?: any) => {
                try {
                  finalize();
                } catch {
                  // fail-open
                }
                return inner.return!(v as any);
              }
            : undefined,
          throw: inner.throw ? (e?: any) => inner.throw!(e) : undefined,
        } as AsyncIterator<any>;
      },
    });
  } catch {
    // fail-open: return the stream untouched (telemetry loss only)
  }
  return stream;
}

async function _callWithInjectedStreamUsage(
  original: any,
  thisArg: any,
  args: any[],
  provider: string,
): Promise<{ result: any; injected: boolean }> {
  const handle = _injectStreamUsageOption(provider, args);
  if (!handle) {
    return { result: await original.apply(thisArg, args), injected: false };
  }
  try {
    const result = await original.apply(thisArg, args);
    // The request (including any provider-SDK-internal retries) has been
    // serialized by now, so revert the customer's body while still reporting
    // injected:true — downstream usage-chunk stripping keys off that flag, not
    // the body object, so the option never leaks back into the customer's own
    // object (stored templates, retries, deep-equal assertions).
    handle.restore();
    return { result, injected: true };
  } catch (err: any) {
    if (_shouldRetryWithoutInjection(err)) {
      handle.restore();
      return { result: await original.apply(thisArg, args), injected: false };
    }
    // Non-retryable: restore the customer's body before the error propagates.
    handle.restore();
    throw err;
  }
}

/**
 * Strip-and-retry only when the rejection is plausibly caused by the injected
 * stream_options.include_usage — a strict-compat server rejecting an unknown
 * param answers 400/422 or names the param in its message. A rate-limit (429)
 * or auth failure (401/403), and any other status, is never caused by the
 * injection and must never trigger a second provider request. Fully guarded: a
 * hostile error object cannot make this throw, and a failure here returns false
 * (no retry → the provider's own error propagates, which is the customer's
 * error, not an SDK failure).
 */
function _shouldRetryWithoutInjection(err: any): boolean {
  try {
    const status = Number(err?.status ?? err?.response?.status);
    if (Number.isFinite(status) && (status === 400 || status === 422)) return true;
    let msg = "";
    try {
      msg = String(err?.message ?? err ?? "").toLowerCase();
    } catch {
      msg = "";
    }
    return msg.includes("stream_options") || msg.includes("include_usage");
  } catch {
    return false;
  }
}

/** Overrides a stream's async iterator to drop SDK-injected usage-only
 * terminal chunks (no usage stashing — used for `.tee()` branches whose
 * parent tap already stashed). Fail-open: returns the stream untouched. */
function _suppressUsageChunkOnly(stream: any): any {
  try {
    if (!stream || typeof stream[Symbol.asyncIterator] !== "function") return stream;
    const getOrig = stream[Symbol.asyncIterator].bind(stream);
    Object.defineProperty(stream, Symbol.asyncIterator, {
      configurable: true,
      writable: true,
      value: function (): AsyncIterator<any> {
        const inner: AsyncIterator<any> = getOrig();
        return {
          async next(): Promise<IteratorResult<any>> {
            for (;;) {
              const r = await inner.next();
              if (!r.done && _isUsageOnlyChunk(r.value)) continue;
              return r;
            }
          },
          return: inner.return ? (v?: any) => inner.return!(v as any) : undefined,
          throw: inner.throw ? (e?: any) => inner.throw!(e) : undefined,
        } as AsyncIterator<any>;
      },
    });
  } catch {
    // fail-open
  }
  return stream;
}

/** A usage-only terminal chunk (`usage` set, empty `choices`) — what
 * include_usage appends to an OpenAI-wire chat stream. */
function _isUsageOnlyChunk(chunk: any): boolean {
  try {
    return Boolean(
      chunk &&
        chunk.usage != null &&
        Array.isArray(chunk.choices) &&
        chunk.choices.length === 0,
    );
  } catch {
    return false;
  }
}

function _tapStreamUsageForOnEnd(
  provider: string,
  result: any,
  session: any,
  args: any[],
  order: number,
  suppressUsageChunk = false,
  reqStartMono?: number,
): any {
  try {
    if (!result || typeof result[Symbol.asyncIterator] !== "function") return result;
    // Key by the order captured BEFORE original.apply (= the order the
    // instrumentor span's onStart consumes), matching _stashApiBase. Using the
    // post-onStart counter would be off by one and land the usage on the NEXT
    // span — dropping this one and mis-attributing the next.
    const compKey = `${session.traceId}:${order}`;
    const getOrig = result[Symbol.asyncIterator].bind(result);
    // Latency anchors: the request-start mono comes from the wrapper (captured
    // before the provider call); fall back to tap creation. TTFT is marked at
    // the first chunk carrying renderable content (text OR tool-call delta).
    const tapStartMono = reqStartMono ?? performance.now();
    let ttftMono: number | null = null;
    // Response-composition accumulator (same machinery as _wrapManualStream).
    // Without it the raw Stream object would Tier-3 to "[object Object]".
    const acc = _newStreamAccumulator(provider);
    // ── Failure emission (mirrors the proven catch in _wrapManualStream) ──
    // On the traceloop-instrumented streaming path `await create()` resolves
    // BEFORE any HTTP happens — the request only fires on first next(). So
    // the wrapper's request-failure catch has already passed and this tap is
    // the ONLY place a provider failure (401/400/429 mid-stream) can be
    // observed and turned into a failure row.
    //
    // Captured obs key for the failure emit below: the iterator's next() runs
    // in the CONSUMER's async context (ALS does not survive), so the key is
    // captured here — still inside the wrapper's per-call obs scope — and
    // re-entered around the emit via runWithObsKey.
    const _obsKey = _currentObsKey();
    // F-17-A: inherited `.stream()` log latch, captured ONCE at tap-setup time
    // (like _obsKey above — the tap installs inside the create wrapper's async
    // context; the drain runs in the consumer's, where ALS doesn't survive).
    // Non-null ONLY when this create was internally delegated from a patched
    // `.stream()` call; this tap is generic across providers, so for every
    // other context the latch is null and the failure mark below is a
    // guaranteed no-op.
    let _inheritedStreamLatch: StreamLatch | null = null;
    try {
      _inheritedStreamLatch = _anthropicStreamLatchStorage.getStore() ?? null;
    } catch {
      _inheritedStreamLatch = null; // fail-open: treat as an independent call
    }
    // Snapshot the attempt context at install time. _stashAttemptContext ran
    // for THIS call just before the provider call, but the _attempted_* slots
    // are session-global — a later call on the same session may overwrite
    // them before this stream fails.
    const _attemptSnap = {
      model: session._attempted_model,
      provider: session._attempted_provider,
      operation: session._attempted_operation,
      shape: session._attempted_shape,
      wireKey: session._attempted_wire_key,
    };
    // Once-guard in the OUTER tap closure (deliberately NOT per-iterator
    // object): repeated Symbol.asyncIterator calls / repeated next() after a
    // rejection must not double-emit the failure row.
    let _streamFailureEmitted = false;
    // Entirely fail-open and fully synchronous (no await): a telemetry
    // failure here degrades to telemetry loss — it must never break the
    // customer's iteration or change the error they receive (golden rule).
    const _emitTapStreamFailure = (err: unknown): void => {
      try {
        if (_streamFailureEmitted) return;
        _streamFailureEmitted = true;
        // Mark this compKey failed FIRST and SYNCHRONOUSLY — telemetry's
        // deferred onEnd log (scheduled on nextTick from the span-end path)
        // must observe it, and microtask-before-nextTick ordering guarantees
        // this catch runs before that deferred log. The marker lives in a
        // dedicated session Set, NOT on _pendingCompositions[compKey]: the
        // emit below may consume+delete that entry (prompt attach in
        // _emitCallFailureLog), which would destroy an entry-resident marker
        // before telemetry reads it.
        ((session._failedStreamCompKeys ??= new Set()) as Set<string>).add(compKey);
        // Failed outcome — classifies error_kind / http_status.
        const elapsedMs = Math.round(Math.max(0, performance.now() - tapStartMono));
        session._call_outcome = buildCallOutcome(err, elapsedMs);
        // Restore this call's attempt context if a later same-session call
        // overwrote the session-global slots. Save the current values first:
        // when they differ from the snapshot they belong to a newer in-flight
        // call — hand them back after the emit (which clears the slots) so
        // that call's failure row isn't degraded to model="unknown".
        const cur = {
          model: session._attempted_model,
          provider: session._attempted_provider,
          operation: session._attempted_operation,
          shape: session._attempted_shape,
          wireKey: session._attempted_wire_key,
        };
        const overwritten =
          cur.model !== _attemptSnap.model ||
          cur.provider !== _attemptSnap.provider ||
          cur.operation !== _attemptSnap.operation ||
          cur.shape !== _attemptSnap.shape ||
          cur.wireKey !== _attemptSnap.wireKey;
        session._attempted_model = _attemptSnap.model;
        session._attempted_provider = _attemptSnap.provider;
        session._attempted_operation = _attemptSnap.operation;
        session._attempted_shape = _attemptSnap.shape;
        session._attempted_wire_key = _attemptSnap.wireKey;
        // Emit inside the captured obs scope so the failure drain claims this
        // call's own observations (same as _wrapManualStream's catch).
        const _failureDispatched = _reenterObsScope(_obsKey, () =>
          _emitCallFailureLog(getClient(), session),
        );
        // F-17-A: mark the shared `.stream()` latch ONLY after a REAL
        // dispatch, so the consumer-layer iterator catch skips its degraded
        // duplicate (see _streamLatchMarkFailure). No-op when no latch was
        // inherited (any non-delegated context).
        if (_failureDispatched) _streamLatchMarkFailure(_inheritedStreamLatch);
        // _emitCallFailureLog cleared the _attempted_* slots; hand a newer
        // call's context back if we borrowed them.
        if (overwritten) {
          session._attempted_model = cur.model;
          session._attempted_provider = cur.provider;
          session._attempted_operation = cur.operation;
          session._attempted_shape = cur.shape;
          session._attempted_wire_key = cur.wireKey;
        }
      } catch {
        // fail-open: failure logging must never affect customer error propagation
      }
    };
    Object.defineProperty(result, Symbol.asyncIterator, {
      configurable: true,
      writable: true,
      value: function (): AsyncIterator<any> {
        const inner: AsyncIterator<any> = getOrig();
        return {
          async next(): Promise<IteratorResult<any>> {
            for (;;) {
              // Only the pull is guarded: a rejection here IS the provider
              // failing the call (the request fires on first next() on this
              // path). Emit the failure row, then rethrow the ORIGINAL error
              // by identity so the app sees exactly the provider's error.
              let r: IteratorResult<any>;
              try {
                r = await inner.next();
              } catch (err) {
                _emitTapStreamFailure(err);
                throw err;
              }
              try {
                if (!r.done) {
                  _accumulateStreamChunk(provider, acc, r.value);
                  if (ttftMono === null && _streamAccHasContent(provider, acc)) {
                    ttftMono = performance.now();
                  }
                }
                // Stash usage the instant the include_usage chunk arrives. That
                // chunk PRECEDES the stream's `done`, and the instrumentor ends
                // its span at `done` (scheduling onEnd's log on nextTick) — so the
                // stash is already in place when that deferred log reads it.
                // Stashing at `done` instead would race and lose (nextTick runs
                // before the microtask continuation that observes `done`).
                if (!r.done && _chunkHasUsage(provider, r.value)) {
                  const u = _extractUsage(provider, r.value, args);
                  if (u && (u.inputTokens > 0 || u.outputTokens > 0)) {
                    const cd =
                      session._pendingCompositions[compKey] ??
                      (session._pendingCompositions[compKey] = {});
                    cd.usage = {
                      input_tokens: u.inputTokens,
                      output_tokens: u.outputTokens,
                      cached_tokens: u.cachedTokens || 0,
                    };
                    // Forward the provider's verbatim chunk usage alongside the
                    // netted positional counts. input_tokens above is
                    // cache-EXCLUSIVE (max(0, prompt_tokens - cached) per
                    // _extractUsage); the OpenAI-family shape mappers treat
                    // raw.prompt_tokens as cache-INCLUSIVE and subtract cached
                    // themselves, so the Mode-A stream fallback in telemetry.ts
                    // must emit the inclusive prompt_tokens this object carries.
                    // Deep-cloned so a later chunk mutation can't corrupt it; any
                    // failure skips the raw stash (positional counts remain).
                    try {
                      const rawUsage = r.value.usage ?? r.value.x_groq?.usage;
                      if (
                        rawUsage &&
                        typeof rawUsage === "object" &&
                        (Number(rawUsage.prompt_tokens) > 0 ||
                          Number(rawUsage.completion_tokens) > 0 ||
                          Number(rawUsage.total_tokens) > 0)
                      ) {
                        cd.usage.raw = JSON.parse(JSON.stringify(rawUsage));
                      }
                    } catch {
                      // fail-open: positional counts remain the fallback
                    }
                    // Chunks echo the response-level service tier (OpenAI) —
                    // stash it so streamed batch/flex/priority calls price right.
                    const tier = _extractServiceTier(r.value);
                    if (tier) cd.service_tier = tier;
                    // Latency primitives — the usage chunk is the stream's
                    // terminal content, so "now" approximates stream end.
                    const lat = _buildStreamLatency(tapStartMono, ttftMono, performance.now());
                    if (lat) cd.latency = lat;
                    // Streamed response composition from the accumulated
                    // chunks (the onEnd path otherwise has only the raw
                    // Stream object → "[object Object]"). Also refresh the
                    // pending tool-call ids so manual toolSpan correlation
                    // works on streamed tool turns.
                    try {
                      const respForComp = _streamAccumulatorToResponse(provider, acc);
                      if (respForComp) {
                        const comp = buildResponseComposition(provider, respForComp);
                        if (comp.length > 0) cd.response = comp;
                        session.setPendingToolCalls(
                          extractPendingToolCalls(provider, respForComp),
                        );
                      }
                    } catch {
                      // fail-open: composition is best-effort
                    }
                  }
                }
              } catch {
                // fail-open: a telemetry tap must never break customer iteration
              }
              // The SDK injected include_usage (the customer didn't ask) — strip
              // the synthetic usage-only terminal chunk so customer code sees
              // exactly the chunks it requested. Never strips content chunks
              // (usage attached to a chunk WITH choices passes through).
              if (suppressUsageChunk && !r.done && _isUsageOnlyChunk(r.value)) {
                continue;
              }
              return r;
            }
          },
          // (see tee() suppression below — teed branches bypass this iterator)
          return: inner.return ? (v?: any) => inner.return!(v as any) : undefined,
          throw: inner.throw ? (e?: any) => inner.throw!(e) : undefined,
        } as AsyncIterator<any>;
      },
    });
    // openai-node's Stream.tee() calls the internal iterator directly,
    // bypassing the patched Symbol.asyncIterator — so an SDK-injected usage
    // chunk would leak to teed consumers. Wrap tee() so each branch gets the
    // suppression (usage is already stashed by the parent tap above).
    if (suppressUsageChunk && typeof (result as any).tee === "function") {
      const origTee = (result as any).tee.bind(result);
      (result as any).tee = function (...teeArgs: any[]): any {
        const out = origTee(...teeArgs);
        try {
          if (Array.isArray(out)) return out.map((s) => _suppressUsageChunkOnly(s));
        } catch {
          // fail-open
        }
        return out;
      };
    }
  } catch {
    // fail-open: return the stream untouched
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════════
// Anthropic Message Batches — results-retrieval telemetry. Batch spend is
// otherwise invisible: the submit call carries no usage, and results never
// pass through the instrumented .create path. Wrapping .results() logs one
// llm row per succeeded entry with shape='anthropic_batch' + tier='batch'
// (the server applies the batch discount), yielding every entry verbatim to
// the customer.
// ═══════════════════════════════════════════════════════════════════

// In-process dedup: results pages are idempotent reads and may be re-read
// (retries, multiple consumers). Only the first read in this process logs a
// given (batch_id, custom_id). Cross-process re-reads produce rows with the
// SAME deterministic span_id, so they stay identifiable/dedupable downstream.
const _batchResultsLogged = new Set<string>();
const _BATCH_RESULTS_LOGGED_MAX = 50000;

function _batchResultAlreadyLogged(key: string): boolean {
  if (_batchResultsLogged.has(key)) return true;
  _batchResultsLogged.add(key);
  if (_batchResultsLogged.size > _BATCH_RESULTS_LOGGED_MAX) {
    _batchResultsLogged.delete(_batchResultsLogged.values().next().value as string);
  }
  return false;
}

const _logAnthropicBatchEntry = failSafeSync(function (
  entry: any,
  batchId: string,
  apiBase: string,
): void {
  const tp = getClient();
  if (!tp) return;
  const result = entry?.result;
  if (result?.type !== "succeeded") return; // errored/canceled/expired carry no usage
  const customId = String(entry?.custom_id ?? "");
  if (_batchResultAlreadyLogged(`${batchId}:${customId}`)) return;
  const message = result?.message;
  const usage = message?.usage;
  if (!usage) return;
  const model = String(message?.model ?? "unknown");

  const session = getCurrentSession();
  const order = session.nextSpanOrder();
  const nowISO = new Date().toISOString();
  const spanObj = {
    ...manualSpanIds(session),
    // Deterministic span id: a re-read from another process emits the SAME id
    // for the same batch entry, so duplicates are detectable downstream.
    span_id: createHash("sha1")
      .update(`anthropic_batch:${batchId}:${customId}`)
      .digest("hex")
      .slice(0, 16),
    span_kind: "llm" as const,
    span_name: `batch:${customId || batchId}`,
    span_order: order,
    start_time: nowISO,
    end_time: nowISO,
  };

  const metadata: Record<string, unknown> = {
    workflow_name: session.workflowName,
    batch_id: batchId,
    batch_custom_id: customId,
  };
  if (session.sessionId) metadata.session_id = session.sessionId;
  // B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
  // session metadata WITHOUT it, then re-add it only when this row belongs
  // to the call that was actually rerouted (exact obs-key match).
  if (session.metadata) {
    copySessionMetadata(metadata, session.metadata);
  }
  // ACCEPTED B4 RESIDUAL — batch reroute provenance is intentionally dropped
  // here. A reroute (if any) was applied on the batch CREATE call; this row is
  // emitted while RETRIEVING results, often in another process entirely (the
  // dedupe set below is result-time only — nothing correlates create back to
  // retrieve, by design, since results are re-readable). So the key read here
  // belongs to the retrieval call and never matches: the row simply carries no
  // `_tp_routing`. Omission only — a batch row can never show a stranger's
  // reroute, which is the property that matters. Do not "fix" this by falling
  // back to an unkeyed peek.
  stampRoutingMarker(metadata, session, _currentObsKey());

  let rawUsage: unknown = null;
  try {
    rawUsage = JSON.parse(JSON.stringify(usage));
  } catch {
    rawUsage = null;
  }

  tp.log(
    session.userId,
    session.paidPlan,
    session.workflowName,
    session.sessionId,
    model,
    "anthropic",
    Number(usage?.input_tokens ?? 0) || 0,
    Number(usage?.output_tokens ?? 0) || 0,
    Number(usage?.cache_read_input_tokens ?? 0) || 0,
    metadata,
    spanObj,
    [],
    [],
    {
      usage: { shape: "anthropic_batch", raw: rawUsage, tier: "batch" },
      model_extras: {
        endpoint: "messages/batches/results",
        ...(apiBase ? { api_base: apiBase } : {}),
      },
      planSource: session.planSource,
    },
  );
});

/** Overrides the resolved results page's async iterator in place — entries
 * pass through verbatim; logging rides alongside, fail-open. */
function _tapBatchResultsIterable(page: any, batchId: string, apiBase: string): any {
  try {
    if (!page || typeof page[Symbol.asyncIterator] !== "function") return page;
    const getOrig = page[Symbol.asyncIterator].bind(page);
    Object.defineProperty(page, Symbol.asyncIterator, {
      configurable: true,
      writable: true,
      value: function (): AsyncIterator<any> {
        const inner: AsyncIterator<any> = getOrig();
        return {
          async next(): Promise<IteratorResult<any>> {
            const r = await inner.next();
            if (!r.done) {
              try {
                _logAnthropicBatchEntry(r.value, batchId, apiBase);
              } catch {
                // fail-open: telemetry never breaks customer iteration
              }
            }
            return r;
          },
          return: inner.return ? (v?: any) => inner.return!(v as any) : undefined,
          throw: inner.throw ? (e?: any) => inner.throw!(e) : undefined,
        } as AsyncIterator<any>;
      },
    });
  } catch {
    // fail-open: return the page untouched
  }
  return page;
}

function _instrumentAnthropicBatches(anthropicEntry: any): void {
  try {
    const root = anthropicEntry?.Anthropic ?? anthropicEntry?.default ?? anthropicEntry;
    // Resolve the Batches class: the root namespace re-export first, then the
    // package subpath for SDK versions that don't re-export it.
    let BatchesCls = root?.Messages?.Batches ?? root?.Batches;
    if (!BatchesCls) {
      // App-first so the subpath comes from the same @anthropic-ai/sdk copy
      // the entry module resolved from. Not resolvable in this SDK version →
      // undefined → skip.
      const sub = resolveProviderModule(
        "@anthropic-ai/sdk/resources/messages/batches",
      );
      BatchesCls = sub?.Batches ?? sub?.default;
    }
    const proto = BatchesCls?.prototype;
    if (!proto || typeof proto.results !== "function" || (proto as any).__tpBatchesPatched) {
      return;
    }
    (proto as any).__tpBatchesPatched = true;
    const orig = proto.results;
    proto.results = function (this: any, messageBatchId: any, ...rest: any[]): any {
      const out = orig.call(this, messageBatchId, ...rest);
      try {
        const apiBase = _extractBaseURL(this);
        const batchId = String(messageBatchId ?? "");
        // .results() returns a LAZY APIPromise (the request fires on await).
        // Wrap .then instead of chaining on it, so laziness, identity, and
        // helper methods (.withResponse()) are all preserved; the resolved
        // page gets its iterator tapped in place.
        const origThen = out?.then?.bind(out);
        if (typeof origThen === "function") {
          out.then = (onFulfilled?: any, onRejected?: any) =>
            origThen((page: any) => {
              try {
                _tapBatchResultsIterable(page, batchId, apiBase);
              } catch {
                // fail-open
              }
              return onFulfilled ? onFulfilled(page) : page;
            }, onRejected);
        }
      } catch {
        // fail-open: hand back the original promise untouched
      }
      return out;
    };
    // Restore directly to this prototype (Batches isn't root-exported in every
    // SDK version) and clear the idempotency marker so a re-instrument re-wraps.
    _restoreThunks.push(() => {
      proto.results = orig;
      try {
        delete (proto as any).__tpBatchesPatched;
      } catch {
        // non-configurable — ignore
      }
    });
  } catch {
    // fail-open: batches stay uninstrumented
  }
}

// ═══════════════════════════════════════════════════════════════════
// Anthropic `Messages.stream()` context-manager instrumentation.
//
// `client.messages.stream()` is a distinct first-party streaming helper (the
// `.on('text')` / `for await` / `await .finalMessage()` API), not
// `create({stream:true})`. Node instruments only `.create` (via the
// target-method registry), so streamed context-manager calls would otherwise
// get no pre-flight `/check` and no `/log`. Ports the equivalent Anthropic
// stream instrumentation from the Python SDK.
//
// The wrapper is object-preserving (reuses the create-path stream-tap idiom):
// it returns the same MessageStream with only the gated members overridden in
// place — `Symbol.asyncIterator`, `finalMessage`, `on`, `once` — so
// `.controller`, `.tee()`, `.abort()`, etc. stay intact (never a bare async
// generator: replacing the object would drop those and break private-field
// brand checks).
//
// Fail-open contract: the async pre-flight is the reused check helper
// (fail-safe-wrapped, firewall-gated) — it rejects only with
// TokenPoliceBlockedError under a verified enforce block and swallows every
// other error. That block is gated at every instrumented delivery surface —
// the async iterator's first `next()`, `finalMessage()`, `finalText()`,
// `done()`, `emitted()`, and the `.on`/`.once` emitter — and a best-effort
// `abort()` is issued, so a customer using those surfaces receives no content
// on a block. Residual limitations: direct data reads on the preserved object
// (`receivedMessages` / `currentMessage`) are not interceptable, and a token
// emitted before the async check resolves can still slip through a degraded
// fail-open path. Every non-block path fails open (telemetry loss /
// pass-through), never a throw.
//
// Known limitation: unlike the create-path pre-check, the provider HTTP
// request may already be initiated inside the MessageStream constructor
// before the async check resolves — request-initiation suppression is a
// create-path-only property. The guarantee here is that no tokens are
// surfaced before a block. Covered by tests/anthropicStream.test.ts.
// ═══════════════════════════════════════════════════════════════════

// Token-bearing MessageStream event names. Listeners for these are buffered by
// the wrapper until the pre-flight check resolves (flushed on allow, dropped on
// enforce block). `error` / `abort` (and any other name) always pass through.
const _ANTHROPIC_TOKEN_EVENTS = new Set([
  "text",
  "message",
  "contentBlock",
  "streamEvent",
  "finalMessage",
  // Newer `@anthropic-ai/sdk` token-bearing events. `inputJson` is emitted on
  // 0.30.x; `thinking` / `citation` / `signature` land on later versions and are
  // a pure no-op where never emitted (forward-compat gating).
  "inputJson",
  "thinking",
  "citation",
  "signature",
]);

// Per-call async-context channel carrying a `.stream()` call's one-shot log latch
// to any `create({stream:true})` the SDK internally delegates to WITHIN that
// stream's construction window. Scoping it here (instead of on the long-lived
// session) means a later, independent `create({stream:true})` in the same
// session scope no longer inherits a stale latch and drops its row. A stub
// (run⇒fn(), getStore⇒undefined) keeps this a no-op on runtimes without
// AsyncLocalStorage — degrading to a possible dup row in the delegation case,
// never a dropped row or a throw.
// `logged` (success one-shot) and `failureLogged` (failure one-shot) are
// INDEPENDENT: a failure emit must never suppress a success row on another
// layer and vice versa (retries can produce a failure then a success across
// different layers of the same `.stream()` call).
// `checkPromise` + `checkedBody` are the single-flight pre-flight channel: the
// `.stream()` wrapper publishes its in-flight check promise and the body object
// that check evaluated (and, on an applied REROUTE, mutated in place); the
// internally-delegated `create({stream:true})` awaits the shared promise
// instead of issuing a second /check, and syncs the reroute model from
// `checkedBody` onto its own body copy. Set ONLY after a successful kickoff —
// when absent, the delegated layer runs its own layer-local check (degradation
// is "two checks", never "zero checks").
type StreamLatch = {
  logged: boolean;
  failureLogged?: boolean;
  observed?: Set<string>;
  checkPromise?: Promise<unknown>;
  checkedBody?: Record<string, any>;
};
const _anthropicStreamLatchStorage: {
  run<T>(store: StreamLatch, fn: () => T): T;
  getStore(): StreamLatch | undefined;
} = (() => {
  try {
    return new AsyncLocalStorage<StreamLatch>();
  } catch {
    return {
      run<T>(_store: StreamLatch, fn: () => T): T {
        return fn();
      },
      getStore(): StreamLatch | undefined {
        return undefined;
      },
    };
  }
})();

/**
 * One-shot manual-log latch. A streamed `.stream()` call may reach TWO manual-log
 * points for the SAME request: (a) this file's `.stream()` wrapper
 * (`finalMessage()` / iterator-drain), and (b) the create-path
 * `_tapAnthropicStreamBypass` when `.stream()` internally delegates to the
 * patched `create({stream:true})` (SDK-version dependent). Both share ONE latch —
 * the `.stream()` wrapper stamps it on the MessageStream (`__tpLogLatch`) for its
 * own log point, and propagates it to the delegated create via
 * `_anthropicStreamLatchStorage`. Whichever path reaches its log point FIRST
 * "takes" the latch and logs; the other sees it taken and SKIPS → exactly one row
 * when `.stream()` delegates to the patched create (constraint covered by A7(b) /
 * A7(b-reverse)). The take must happen only for a path about to log a valid
 * (non-zero) row, so the latch is never taken-then-skipped → never 0 rows.
 * Fail-open: any bookkeeping error degrades to "log anyway" (a possible dup) over
 * a lost row / a throw. `marker.__tpLogLatch` is the shared latch; a no-latch
 * marker means an independent call → log.
 */
function _streamLatchTake(marker?: any): boolean {
  try {
    const latch = marker && marker.__tpLogLatch;
    if (!latch) return true; // no latch in play → log
    if (latch.logged) return false; // another path already logged → skip
    latch.logged = true; // take it and log
    return true;
  } catch {
    return true; // fail-open: prefer a row (possible dup) over a lost row
  }
}

/**
 * FAILURE-side counterpart of the latch (F-17-A). A `.stream()` request that
 * fails at request time is observed FIRST at the delegated-create layer (a
 * good row — attempt context still stashed), then the vendor MessageStream
 * catches that rejection internally and RE-DELIVERS it to the consumer's
 * `for await`, whose wrapper catch would emit a SECOND, degraded row
 * (model='unknown'/provider='' — the first emit cleared `_attempted_*`).
 * Dedupe direction is deliberate: the SOURCE layer marks `failureLogged` ONLY
 * AFTER `_emitCallFailureLog` reported a REAL dispatch, and the CONSUMER layer
 * checks — so a skipped consumer emit always has a real row behind it, and
 * dedupe can never cost the only row. `failureLogged` is independent of
 * `logged` (see the StreamLatch type note). Both helpers swallow everything:
 * worst case is a duplicate row, never a lost row or a throw.
 */
function _streamLatchMarkFailure(latch: StreamLatch | null | undefined): void {
  try {
    if (latch && typeof latch === "object") latch.failureLogged = true;
  } catch {
    // fail-open: an unmarkable latch degrades to a possible dup row
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _streamLatchFailureTaken(mgr: any): boolean {
  try {
    return mgr?.__tpLogLatch?.failureLogged === true;
  } catch {
    return false; // fail-open: prefer a possible dup row over a lost row
  }
}

function _instrumentAnthropicStream(anthropicEntry: any): void {
  try {
    const root = anthropicEntry?.Anthropic ?? anthropicEntry?.default ?? anthropicEntry;
    _patchAnthropicStreamMethod(root?.Messages, false);
    _patchAnthropicStreamMethod(root?.AsyncMessages, true);
    // Beta.Messages is a SIBLING class (extends APIResource, not Messages), so
    // its `.stream()` needs its own patch. Optional-chained on purpose:
    // `Beta.Messages.prototype.stream` is ABSENT in 0.30.1 and present in
    // 0.90.0 — _patchAnthropicStreamMethod already no-ops when `.stream` is
    // missing. Beta.AsyncMessages mirrors the non-beta line above for SDK
    // builds that expose it.
    _patchAnthropicStreamMethod(root?.Beta?.Messages, false);
    _patchAnthropicStreamMethod(root?.Beta?.AsyncMessages, true);
  } catch {
    // fail-open: streaming stays uninstrumented (init must never throw)
  }
}

function _patchAnthropicStreamMethod(cls: any, isAsync: boolean): void {
  try {
    const proto = cls?.prototype;
    if (!proto || typeof proto.stream !== "function") return;
    // Idempotent — never double-wrap. OWN-property check on purpose: today
    // Beta.Messages is a sibling class, but if a future vendor made it
    // `extends Messages`, a plain truthy read would INHERIT the non-beta
    // marker and silently skip patching the beta prototype (lost telemetry).
    if (Object.prototype.hasOwnProperty.call(proto, "__tpStreamPatched")) return;
    const orig = proto.stream;
    (proto as any).__tpStreamPatched = true;
    _anthropicStreamRestore.push({ proto, method: "stream", original: orig });
    proto.stream = function (this: any, ...args: any[]): any {
      // One per-call obs scope around the WHOLE `.stream()` body — covering
      // the wrapper's pre-flight and the internal delegated
      // `create({stream:true})` (whose wrapper sees the existing scope and
      // reuses it). Normally only the wrapper's single-flight pre-flight runs;
      // the shared scope remains so the degraded two-check paths still carry
      // ONE key. The latch dedup (_pushObservationOnce) is unchanged and
      // orthogonal.
      return _runWithCallObsScope(() =>
        _anthropicStreamWrapper(orig, this, args),
      );
    };
  } catch {
    // fail-open: this variant stays uninstrumented
  }
}

/**
 * The `.stream()` replacement. Runs the framework-scope skip + async pre-flight,
 * constructs the real MessageStream, and returns it with the token-delivery
 * surfaces gated. NEVER throws synchronously except a customer-side construction
 * error (which is the customer's own, re-propagated unswallowed).
 */
function _anthropicStreamWrapper(orig: any, thisArg: any, args: any[]): any {
  // Framework scope (LangChain / LlamaIndex / Vercel AI middleware) → the
  // framework wrapper owns telemetry for this call. Pass through untouched:
  // no pre-flight, no suppress flag, no manual log.
  try {
    const _s = getCurrentSession();
    if (_s.inLangchain || _s.inLlamaIndex || _inAiSdkMiddlewareScope()) {
      return orig.apply(thisArg, args);
    }
  } catch {
    // fail-open: if the scope probe throws, continue with the guarded path
  }

  // One-shot log latch shared with the create-path bypass tap. Created BEFORE the
  // pre-flight and the factory run, and propagated via async context so any
  // internally-delegated `create({stream:true})` fired within the stream
  // construction window — sync or via promise continuations started inside it —
  // inherits it and dedupes to exactly one row. Per-call scoping (vs. the
  // long-lived session) means a later independent `create({stream:true})` in the
  // same session never inherits it. It also carries the single-flight pre-flight
  // channel (`checkPromise`/`checkedBody` — the delegated create awaits the
  // shared check instead of running its own) and the observation dedup set
  // (see _pushObservationOnce) for the degraded paths where two layer-local
  // checks still run.
  const latch: StreamLatch = { logged: false };

  // THE async pre-flight for this streamed call. failSafeAsync-wrapped and
  // firewall-gated: rejects ONLY with TokenPoliceBlockedError under a verified
  // enforce block; resolves under allow / dry_run / off / any swallowed error.
  // Resolve serving provider from the bound client's baseURL (e.g.
  // api.minimax.io → minimax) — do NOT hardcode "anthropic". The sample
  // support_agent_minimax_anthropic_node stream variant uses messages.stream().
  let checkPromise: Promise<unknown>;
  try {
    const body =
      args[0] && typeof args[0] === "object" ? (args[0] as Record<string, any>) : {};
    const serving = _resolveServingProvider("anthropic", thisArg);
    // Run inside the latch scope: ALS propagates through this async function's
    // awaits, so observation pushes deep inside _runAsyncCheck see the store and
    // dedupe on the degraded paths where the delegated create still runs its own
    // pre-flight.
    checkPromise = _anthropicStreamLatchStorage.run(latch, () =>
      Promise.resolve(
        _runAsyncCheck(body, serving.provider, null, true, serving.servingUnverified),
      ),
    );
    // Publish the pre-flight on the per-call latch (single-flight): the
    // internally-delegated `create({stream:true})` awaits this promise instead
    // of issuing a second /check, and reads `checkedBody` to sync an applied
    // reroute onto its own body copy. MUST stay inside this try, after the
    // successful kickoff — if the kickoff threw and the catch substituted
    // Promise.resolve(), the fields stay unset so the delegated layer falls
    // back to its own check (never zero checks before dispatch).
    latch.checkPromise = checkPromise;
    latch.checkedBody = body;
  } catch {
    checkPromise = Promise.resolve(); // pre-flight kickoff must never throw here
  }
  // Defensive: observe the pre-flight rejection at its source so it can never be
  // an unhandled rejection, even on paths that early-return before installing the
  // gate (e.g. _wrapAnthropicMessageStream returns a non-object mgr). This is a
  // no-op handler on a DERIVED promise — `checkPromise` itself is unchanged, so
  // the gate below still observes the rejection and a real enforce block still
  // propagates TokenPoliceBlockedError exactly as before.
  void checkPromise.catch(() => {});

  // Construct the real MessageStream (provider request may fire here — honest
  // limitation). A customer-side construction error is theirs → re-propagate.
  // Nested async-context scopes cover the construction window: the latch (log
  // dedup with any internally-delegated `create({stream:true})`) and the
  // OTel-suppression flag. The inner OTel anthropic instrumentor's `.stream`
  // patch runs WITHIN this window, so its duplicate span starts while the
  // suppression flag is active → telemetry `onStart` tags exactly that span
  // (per-call), and `onEnd` drops it. See runWithAnthropicStreamOtelSuppress.
  const mgr = _anthropicStreamLatchStorage.run(latch, () =>
    runWithAnthropicStreamOtelSuppress(() => orig.apply(thisArg, args)),
  );

  try {
    return _wrapAnthropicMessageStream(mgr, thisArg, args, checkPromise, latch);
  } catch {
    // fail-open: setup failure degrades to the raw (un-gated) stream — a token
    // MAY leak on a block via this degraded path, but the stream never breaks.
    // The pre-flight rejection is already handled below to avoid an unhandled
    // rejection; re-attach a swallow in case _wrapAnthropicMessageStream threw
    // before installing its own handler.
    try {
      checkPromise.catch(() => {});
    } catch {
      // ignore
    }
    return mgr;
  }
}

/**
 * Installs the in-place gates on a MessageStream and returns the SAME object.
 * Sets up: the never-rejecting decision gate, the one-shot log latch, the
 * `.on`/`.once` emitter buffer, and the `Symbol.asyncIterator` + `finalMessage`
 * await-gates. (The duplicate-OTel-span suppression is now handled per-call
 * by the caller's runWithAnthropicStreamOtelSuppress window, not here.)
 */
function _wrapAnthropicMessageStream(
  mgr: any,
  thisArg: any,
  args: any[],
  checkPromise: Promise<unknown>,
  latch: { logged: boolean },
): any {
  if (!mgr || typeof mgr !== "object") return mgr;

  const session = getCurrentSession();
  // Captured obs key for the drain-time callbacks below (logFinal /
  // iterator-error paths fire from the consumer's event/iteration context,
  // outside this call's obs scope; this function itself runs inside it).
  const _obsKey = _currentObsKey();

  // The Traceloop anthropic `.stream` duplicate OTel span is
  // suppressed so it does not double-log alongside our manual row. Suppression
  // is now PER-CALL: the caller (_anthropicStreamWrapper) constructs this stream
  // inside runWithAnthropicStreamOtelSuppress, so the instrumentor's span starts
  // within that async-context window and telemetry `onStart` tags it for
  // `onEnd` to drop. This replaced a session-wide one-shot boolean that, under
  // concurrent same-session `.stream()` calls, could let one call's span consume
  // another call's suppression (eat the wrong span → lose its row → double-count
  // the intended duplicate). Nothing to set here anymore.

  // Reserve the span order + name for the manual log (Python parity: __enter__
  // reserves next_span_order() and stashes composition). Fail-open defaults.
  let order = 0;
  let spanName: string | null = null;
  try {
    order = session.nextSpanOrder();
    spanName = consumePendingSpanName();
  } catch {
    // fail-open
  }
  const startTime = new Date();
  const reqStartMono = performance.now();

  // kwargs copy carrying the serving endpoint so _logManual's _extractBaseURL
  // pickup emits model_extras.api_base (mirror _tapAnthropicStreamBypass:3892).
  let kwargsArgs: any[];
  try {
    kwargsArgs = [{ ...(args[0] || {}), baseURL: _extractBaseURL(thisArg) }];
  } catch {
    kwargsArgs = [args[0] || {}];
  }

  // Best-effort prompt composition at the reserved order (Python parity).
  try {
    _captureCompositionAt("anthropic", kwargsArgs, undefined, order);
  } catch {
    // fail-open
  }

  // One-shot log latch shared with the create-path tap (created by the caller and
  // propagated to a delegated create via async context). Stamp it on the stream
  // object for this wrapper's own log point. The defineProperty write is
  // fail-open (frozen/sealed stream).
  try {
    Object.defineProperty(mgr, "__tpLogLatch", {
      configurable: true,
      writable: true,
      enumerable: false,
      value: latch,
    });
  } catch {
    // fail-open: a frozen/sealed stream can't hold the latch; this
    // wrapper's own log point then can't dedupe (possible dup row), never a throw.
  }

  // ── Never-rejecting decision gate ──────────────────────────────────
  // Convert the pre-flight into a gate the token-delivery surfaces await. It
  // resolves once the check settles; `blockErr` is set ONLY on a verified enforce
  // block. Every other check error was already swallowed by failSafeAsync.
  let blockErr: TokenPoliceBlockedError | null = null;
  const gate: Promise<void> = checkPromise.then(
    () => {},
    (e: unknown) => {
      if (e instanceof TokenPoliceBlockedError) blockErr = e;
      // else: swallow (defense-in-depth; failSafeAsync already swallowed it)
    },
  );

  const abortBestEffort = (): void => {
    // Abort hygiene: the SDK's `mgr.abort()` fires an internal
    // `_emit('abort', APIUserAbortError)` that becomes a DELIBERATE, process-
    // killing `Promise.reject` whenever no `'abort'`/`'error'` listener is
    // registered and no awaited-promise flag is set. Our `done()`/`finalText()`/
    // `finalMessage()` block overrides throw BEFORE calling the native method, so
    // a `done()`- or `finalText()`-only consumer would otherwise leave that abort
    // unlistened — which could stop the customer's process. Register no-op
    // `'abort'`/`'error'` listeners just-in-time (only on this block-abort path —
    // the allow path is untouched) so the rejection never fires. These are
    // ADDITIVE: the customer's own `error`/`abort` listeners still fire alongside
    // (both pass through the `.on`/`.once` gate).
    try {
      mgr.on?.("abort", () => {});
    } catch {
      // fail-open
    }
    try {
      mgr.on?.("error", () => {});
    } catch {
      // fail-open
    }
    try {
      mgr.controller?.abort?.();
    } catch {
      // fail-open
    }
    try {
      mgr.abort?.();
    } catch {
      // fail-open
    }
  };

  // ── Manual-log accumulator (iterator-drain path) ───────────────────
  const acc = {
    model: "",
    usage: {} as Record<string, unknown>,
    stopReason: null as unknown,
  };
  let ttftMono: number | null = null;

  // Emit the manual /log row exactly once (latched). Validity/zero-usage guard
  // runs BEFORE taking the latch so a no-op path never blocks the other (never 0).
  const logFinal = (finalMsg: any): void => {
    try {
      const u: any = (finalMsg && finalMsg.usage) || acc.usage || {};
      if (!(Number(u.input_tokens) > 0 || Number(u.output_tokens) > 0)) return;
      if (!_streamLatchTake(mgr)) return; // another path logged
      const synthetic = {
        model:
          (finalMsg && finalMsg.model) ||
          acc.model ||
          String((kwargsArgs[0] as any)?.model ?? "unknown"),
        stop_reason: (finalMsg && finalMsg.stop_reason) ?? acc.stopReason ?? null,
        usage: u,
        content: Array.isArray(finalMsg?.content) ? finalMsg.content : [],
      };
      if (synthetic.content.length > 0) {
        _captureCompositionAt("anthropic", kwargsArgs, synthetic, order);
      }
      const latency = _buildStreamLatency(reqStartMono, ttftMono, performance.now());
      // Re-enter the captured obs scope for _logManual's keyed drain
      // (logFinal fires from the consumer's event/iteration context).
      _reenterObsScope(_obsKey, () =>
        _logManual(
          "anthropic", kwargsArgs, synthetic, order, spanName, startTime, "chat",
          null, latency,
        ),
      );
    } catch {
      // fail-open: telemetry loss only
    }
  };

  // ── `.on` / `.once` emitter gate ───────────────────────────────
  // Token-bearing listeners are held in a wrapper-owned map and only invoked
  // after the check resolves ALLOWED (buffered events flushed in order, then live
  // events forwarded); on an enforce BLOCK the buffered token events are dropped
  // and the held callbacks never fire. `error`/`abort` pass through unswallowed.
  // The whole interception is fail-open: any setup failure
  // degrades to the original `.on` pass-through.
  try {
    const origOn = typeof mgr.on === "function" ? mgr.on.bind(mgr) : null;
    const origOnce = typeof mgr.once === "function" ? mgr.once.bind(mgr) : null;
    const origOff = typeof mgr.off === "function" ? mgr.off.bind(mgr) : null;
    if (origOn) {
      const held = new Map<string, Array<{ cb: (...a: any[]) => void; once: boolean }>>();
      const buffer: Array<{ evt: string; emitArgs: any[] }> = [];
      let released = false; // check allowed → flushing / live
      let dropped = false; // check blocked → drop token events

      const deliver = (evt: string, emitArgs: any[]): void => {
        const list = held.get(evt);
        if (!list || list.length === 0) return;
        for (const entry of [...list]) {
          if (entry.once) {
            const idx = list.indexOf(entry);
            if (idx !== -1) list.splice(idx, 1);
          }
          try {
            entry.cb(...emitArgs);
          } catch {
            // fail-open: a throwing customer callback must not break our capture
          }
        }
      };

      // Install our OWN capture listener SYNCHRONOUSLY per token event (Node
      // EventEmitter does not replay, so capture MUST precede the eager emitter).
      for (const evt of _ANTHROPIC_TOKEN_EVENTS) {
        try {
          origOn(evt, (...emitArgs: any[]) => {
            try {
              if (dropped) return; // blocked → swallow
              if (released) deliver(evt, emitArgs); // live forward
              else buffer.push({ evt, emitArgs }); // buffer until check resolves
            } catch {
              // fail-open
            }
          });
        } catch {
          // this event failed to capture — its listeners degrade to pass-through
        }
      }

      // Resolve the buffer when the gate settles.
      void gate.then(() => {
        try {
          if (blockErr) {
            dropped = true;
            buffer.length = 0; // drop buffered token events
            held.clear(); // never invoke held token callbacks
            abortBestEffort();
          } else {
            released = true;
            const pending = buffer.splice(0, buffer.length);
            for (const item of pending) deliver(item.evt, item.emitArgs);
          }
        } catch {
          // fail-open
        }
      });

      const gatedOn = function (this: any, evt: any, cb: any): any {
        try {
          if (evt !== "error" && evt !== "abort" && _ANTHROPIC_TOKEN_EVENTS.has(evt)) {
            let list = held.get(evt);
            if (!list) held.set(evt, (list = []));
            list.push({ cb, once: false });
            return this;
          }
        } catch {
          // fall through to pass-through
        }
        try {
          origOn(evt, cb);
        } catch {
          // fail-open
        }
        return this;
      };
      const gatedOnce = function (this: any, evt: any, cb: any): any {
        try {
          if (evt !== "error" && evt !== "abort" && _ANTHROPIC_TOKEN_EVENTS.has(evt)) {
            let list = held.get(evt);
            if (!list) held.set(evt, (list = []));
            list.push({ cb, once: true });
            return this;
          }
        } catch {
          // fall through to pass-through
        }
        try {
          (origOnce ?? origOn)(evt, cb);
        } catch {
          // fail-open
        }
        return this;
      };
      // Gated `off` (the ONLY unsubscribe surface the Anthropic MessageStream
      // exposes — no `removeListener`/`removeAllListeners`). A token-bearing
      // listener registered through `gatedOn`/`gatedOnce` lives in `held` and is
      // invoked from there by `deliver` — it is NEVER attached to the native
      // emitter — so removing it from `held` is what actually stops it firing.
      // We ALSO delegate to native `off`: pass-through events (`error`/`abort`)
      // are attached natively, and native `off` clears them. Mirroring native's
      // "removes at most one instance" semantics: we splice at most one match
      // from `held` (the sole home of token cbs) and let native remove at most
      // one from its own list. The whole override is fail-open — it never throws.
      const gatedOff = function (this: any, evt: any, cb: any): any {
        try {
          const list = held.get(evt);
          if (list) {
            const idx = list.findIndex((entry) => entry.cb === cb);
            if (idx !== -1) list.splice(idx, 1);
          }
        } catch {
          // fail-open: fall through to native delegation regardless
        }
        try {
          if (origOff) return origOff(evt, cb);
        } catch {
          // fail-open
        }
        return this;
      };

      Object.defineProperty(mgr, "on", {
        configurable: true,
        writable: true,
        value: gatedOn,
      });
      Object.defineProperty(mgr, "once", {
        configurable: true,
        writable: true,
        value: gatedOnce,
      });
      if (origOff) {
        Object.defineProperty(mgr, "off", {
          configurable: true,
          writable: true,
          value: gatedOff,
        });
      }
    }
  } catch {
    // fail-open: the whole emitter gate degrades to the SDK's
    // native `.on`/`.once` (untouched) — a token MAY leak on a block via this
    // degraded path, but the customer's stream is never broken.
  }

  // ── `finalMessage()` await-gate ────────────────────────────────────
  try {
    const origFinal =
      typeof mgr.finalMessage === "function" ? mgr.finalMessage.bind(mgr) : null;
    if (origFinal) {
      Object.defineProperty(mgr, "finalMessage", {
        configurable: true,
        writable: true,
        value: async function (): Promise<any> {
          await gate;
          if (blockErr) {
            abortBestEffort();
            throw blockErr; // ONLY propagation — verified enforce block
          }
          const msg = await origFinal();
          try {
            logFinal(msg);
          } catch {
            // fail-open: telemetry loss only
          }
          return msg;
        },
      });
    }
  } catch {
    // fail-open
  }

  // Best-effort read of the metering payload for the `finalText()`/`done()`
  // surfaces, which return text/void (not the Message). The SDK's own
  // `getFinalMessage()` returns exactly `receivedMessages.at(-1)`, so this is the
  // byte-faithful source. Guarded read → `null` on failure → `logFinal(null)` is
  // a guarded no-op (telemetry loss, never a throw).
  const _finalFromReceived = (): any => {
    try {
      const rm = mgr.receivedMessages;
      if (Array.isArray(rm) && rm.length > 0) return rm[rm.length - 1];
    } catch {
      // fail-open
    }
    return null;
  };

  // ── `finalText()` await-gate ───────────────────────────────────────
  // Mirrors the `finalMessage` gate. On a verified enforce block: abort + throw
  // blockErr (no text surfaced). On allow: return the native text and meter once
  // via the shared latch (a finalText-only consumer is still metered). The SDK's
  // `finalText()` internally does `await this.done()`, so the gated `done()`
  // override (below) takes the latch first and logs; this outer `logFinal` then
  // sees it taken and skips → exactly one row.
  try {
    const origFinalText =
      typeof mgr.finalText === "function" ? mgr.finalText.bind(mgr) : null;
    if (origFinalText) {
      Object.defineProperty(mgr, "finalText", {
        configurable: true,
        writable: true,
        value: async function (): Promise<any> {
          await gate;
          if (blockErr) {
            abortBestEffort();
            throw blockErr; // ONLY propagation — verified enforce block
          }
          const text = await origFinalText();
          try {
            logFinal(_finalFromReceived());
          } catch {
            // fail-open: telemetry loss only
          }
          return text;
        },
      });
    }
  } catch {
    // fail-open
  }

  // ── `done()` await-gate ────────────────────────────────────────────
  // On a verified enforce block: abort + throw blockErr (stream torn down, no
  // completion surfaced). On allow: await native completion and meter once
  // (a done-only consumer is still metered). `finalMessage()`/`finalText()`
  // internally route through `this.done()`, so this override is where the shared
  // latch is taken for those paths → exactly one row across all of them.
  try {
    const origDone = typeof mgr.done === "function" ? mgr.done.bind(mgr) : null;
    if (origDone) {
      Object.defineProperty(mgr, "done", {
        configurable: true,
        writable: true,
        value: async function (): Promise<void> {
          await gate;
          if (blockErr) {
            abortBestEffort();
            throw blockErr; // ONLY propagation — verified enforce block
          }
          await origDone();
          try {
            logFinal(_finalFromReceived());
          } catch {
            // fail-open: telemetry loss only
          }
        },
      });
    }
  } catch {
    // fail-open
  }

  // ── `emitted(event)` gate ──────────────────────────────────────────
  // `emitted(event)` resolves the FIRST time `event` fires (SDK:
  // `new Promise((res,rej)=>{ this.once('error',rej); this.once(event,res); })`).
  // Once a token event is gated, its `res` is held by the `.on`/`.once` gate and
  // DROPPED on a block (`held.clear()`), while `once('error',rej)` never fires on
  // a TP block — so a naive path would HANG the customer's `await`. We therefore
  // give `emitted()` its own override: sync-call the native `emitted` (so the
  // eager emitter's listeners are registered immediately), let control events
  // pass straight through, and gate-REJECT token events with blockErr on a block
  // (closing the hang). NOTE: this override and the `.on`/`.once` held-drop share
  // the same `mgr`-extensibility precondition — both install together or both
  // degrade to native together, so no realistic object presents an
  // on-gate-active-but-emitted-native hang.
  try {
    const origEmitted =
      typeof mgr.emitted === "function" ? mgr.emitted.bind(mgr) : null;
    if (origEmitted) {
      Object.defineProperty(mgr, "emitted", {
        configurable: true,
        writable: true,
        value: function (event: any): any {
          let native: any;
          try {
            native = origEmitted(event); // registers native/gated listeners now
          } catch {
            // fail-open: degrade to a resolved no-op rather than throw
            return Promise.resolve(undefined);
          }
          // T-2 (defense-in-depth): the block path discards `native`; ensure a
          // stray `'error'` rejection on it can never surface as unhandled.
          try {
            (native as Promise<any>)?.catch?.(() => {});
          } catch {
            // ignore
          }
          // Control events (`error`/`abort`/`connect`/`end`/…) pass through.
          if (
            event === "error" ||
            event === "abort" ||
            !_ANTHROPIC_TOKEN_EVENTS.has(event)
          ) {
            return native;
          }
          // Token events: gate on the decision. On block, reject with blockErr
          // (closes the held-map-drop HANG); on allow, resolve natively.
          return gate.then(() => {
            if (blockErr) {
              abortBestEffort();
              throw blockErr; // ONLY propagation — verified enforce block
            }
            return native;
          });
        },
      });
    }
  } catch {
    // fail-open
  }

  // ── `Symbol.asyncIterator` await-gate + drain-log ──────────────────
  try {
    const getOrig =
      typeof mgr[Symbol.asyncIterator] === "function"
        ? mgr[Symbol.asyncIterator].bind(mgr)
        : null;
    if (getOrig) {
      Object.defineProperty(mgr, Symbol.asyncIterator, {
        configurable: true,
        writable: true,
        value: function (): AsyncIterator<any> {
          const inner: AsyncIterator<any> = getOrig();
          let gated = false;
          return {
            async next(): Promise<IteratorResult<any>> {
              if (!gated) {
                gated = true;
                await gate;
                if (blockErr) {
                  abortBestEffort();
                  throw blockErr; // ONLY propagation — verified enforce block
                }
              }
              let r: IteratorResult<any>;
              try {
                r = await inner.next();
              } catch (err) {
                // F-17-A dedupe: when the delegated create layer (or its
                // stream tap) already emitted THIS request's failure row, the
                // vendor MessageStream re-delivers the same rejection here —
                // emitting again would add a second, degraded row
                // (model='unknown'/provider='': the first emit cleared the
                // `_attempted_*` stash). The source layer marks the latch
                // ONLY after a REAL dispatch, so skipping here can never lose
                // the only row; that's why the check lives at this consumer
                // layer rather than the mark. Rethrow untouched, WITHOUT
                // stamping a stale failed outcome onto the session.
                if (_streamLatchFailureTaken(mgr)) throw err;
                // Customer's OWN stream error — best-effort failure telemetry,
                // then re-propagate UNSWALLOWED (mirror:3970-3985).
                try {
                  (session as any)._call_outcome = buildCallOutcome(
                    err,
                    Math.round(Math.max(0, performance.now() - reqStartMono)),
                  );
                  // Re-enter the captured obs scope (consumer-context
                  // iteration) so the failure drain claims this call's own
                  // observations.
                  const _failureDispatched = _reenterObsScope(_obsKey, () =>
                    _emitCallFailureLog(getClient(), session),
                  );
                  // Mark our own dispatch too — repeated next() calls after a
                  // rejection must not re-emit the failure row.
                  if (_failureDispatched) {
                    _streamLatchMarkFailure(mgr?.__tpLogLatch);
                  }
                } catch {
                  // fail-safe
                }
                throw err;
              }
              try {
                if (!r.done) {
                  const ev: any = r.value;
                  if (ev?.type === "message_start" && ev.message) {
                    if (ev.message.model) acc.model = String(ev.message.model);
                    if (ev.message.usage && typeof ev.message.usage === "object") {
                      acc.usage = { ...acc.usage, ...ev.message.usage };
                    }
                    if (ttftMono === null) ttftMono = performance.now();
                  } else if (ev?.type === "message_delta") {
                    if (ev.usage && typeof ev.usage === "object") {
                      acc.usage = { ...acc.usage, ...ev.usage };
                    }
                    if (ev.delta?.stop_reason) acc.stopReason = ev.delta.stop_reason;
                  } else if (
                    ev?.type === "content_block_start" ||
                    ev?.type === "content_block_delta" ||
                    ev?.type === "text"
                  ) {
                    if (ttftMono === null) ttftMono = performance.now();
                  }
                } else {
                  // Drain-end must log from `receivedMessages`: the vendor pump
                  // has already stored the final message (with merged usage)
                  // before the consumer iterator sees done. Passing null here
                  // logged a row with empty response composition and
                  // latch-blocked the later rich `finalMessage()` log (F-23-2).
                  logFinal(_finalFromReceived());
                }
              } catch {
                // fail-open: a telemetry tap must never break customer iteration
              }
              return r;
            },
            return: inner.return
              ? (v?: any): Promise<IteratorResult<any>> => {
                  try {
                    // Break-out path: receivedMessages may still be empty
                    // mid-message → null → acc-only row (guarded).
                    logFinal(_finalFromReceived());
                  } catch {
                    // fail-open
                  }
                  return inner.return!(v as any);
                }
              : undefined,
            throw: inner.throw ? (e?: any) => inner.throw!(e) : undefined,
          } as AsyncIterator<any>;
        },
      });
    }
  } catch {
    // fail-open
  }

  return mgr;
}

// ═══════════════════════════════════════════════════════════════════
// LangChain wrappers — for @langchain/core BaseChatModel.{generate,stream}.
//
// LangChain is a framework: it calls the underlying provider SDK (openai,
// anthropic, ...) internally, and that SDK is also patched by the enforcer.
// These wrappers run the SINGLE pre-flight check, capture prompt/response
// composition from LangChain's own message objects, and hold the session's
// inLangchain guard so the nested provider wrapper passes through. Telemetry
// flows through the OpenLLMetry LangChain instrumentor span — see telemetry.ts.
// ═══════════════════════════════════════════════════════════════════

/** Capture prompt composition for a LangChain generate/stream call. */
function _captureLangchainPrompt(args: any[], thisArg?: any, order?: number): void {
  try {
    const session = getCurrentSession();
    // generate(messages, options, callbacks) — args[0] = messages
    // _streamIterator(input, options) — args[0] = input
    const payload = args[0];
    const comp = buildPromptComposition("langchain", { messages: payload });
    // Use the reserved order threaded from the wrapper. Falling back to peeking
    // spanCounter (the value the LLM span's onStart would consume) is the
    // pre-fix behavior for the un-threaded stream paths.
    const key = order ?? session.spanCounter;
    const compKey = `${session.traceId}:${key}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    if (comp.length > 0) {
      session._pendingCompositions[compKey].prompt = comp;
    }
    // Stash the chat model's name from the instance — the OpenLLMetry LangChain
    // JS instrumentor doesn't always emit gen_ai.request.model for non-OpenAI
    // providers (e.g. ChatGoogleGenerativeAI). The onEnd in telemetry.ts uses
    // this override when present.
    if (thisArg) {
      const modelName =
        thisArg.model ?? thisArg.modelName ?? thisArg.model_name;
      if (typeof modelName === "string" && modelName) {
        session._pendingCompositions[compKey].model = modelName;
      }
    }
  } catch {
    // fail-safe: composition is best-effort
  }
}

/** Capture response composition for a LangChain generate call (LLMResult). */
function _captureLangchainResponse(result: any, order?: number): void {
  try {
    const session = getCurrentSession();
    try {
      const comp = buildResponseComposition("langchain", result);
      if (comp.length > 0) {
        const key = order ?? Math.max(0, session.spanCounter - 1);
        const compKey = `${session.traceId}:${key}`;
        if (!session._pendingCompositions[compKey]) {
          session._pendingCompositions[compKey] = {};
        }
        session._pendingCompositions[compKey].response = comp;
      }
    } catch {
      // fail-safe: composition is best-effort
    }
    // Stash tool-call ids so OTel tool spans / toolSpan() can auto-correlate.
    // REPLACE on every response (including []) — matches Mode A + Python LC.
    // Isolated from composition so a composition throw never skips the stash.
    try {
      session.setPendingToolCalls(extractPendingToolCalls("langchain", result));
    } catch {
      /* fail-open */
    }
    // Generate-path usage stash. The Responses-API arm of @langchain/openai
    // reports usage under llmOutput.estimatedTokenUsage, which the traceloop
    // callback never reads (it only reads llmOutput.usage / .tokenUsage), so
    // the LLM span ends with zero usage attrs and telemetry drops the row.
    // message.usage_metadata is LangChain-standard and endpoint-agnostic —
    // stash it (tagged langchain_message) so the telemetry merge repairs the
    // zeros. GATED on the wire usage traceloop DOES read being absent: when
    // llmOutput.usage / .tokenUsage carry positive counts the span attrs are
    // authoritative and this stash must not perturb them (e.g. Anthropic's
    // cache-EXCLUSIVE llmOutput.usage vs LC's cache-INCLUSIVE usage_metadata
    // would flip the preferStash comparison on cached calls).
    try {
      const lo = result?.llmOutput;
      const hasWireUsage =
        Number(lo?.usage?.input_tokens) > 0 ||
        Number(lo?.usage?.output_tokens) > 0 ||
        Number(lo?.tokenUsage?.promptTokens) > 0 ||
        Number(lo?.tokenUsage?.completionTokens) > 0;
      if (!hasWireUsage && Array.isArray(result?.generations)) {
        // LLMResult.generations is Generation[][]: outer index = one provider
        // call (usage must SUM across them for batch generate), inner index =
        // candidate choices of that call — @langchain/openai stamps the SAME
        // full-call usage_metadata on every candidate, so read only the first
        // usage-bearing one per list to avoid n>1 over-count.
        let inTok = 0;
        let outTok = 0;
        let cacheRead = 0;
        let cacheCreation = 0;
        let found = false;
        for (let genList of result.generations) {
          if (!Array.isArray(genList)) genList = [genList];
          for (const gen of genList) {
            const um = gen?.message?.usage_metadata;
            if (!um || typeof um !== "object") continue;
            inTok += Number(um.input_tokens) || 0;
            outTok += Number(um.output_tokens) || 0;
            // Cache detail: LC-standard input_token_details.cache_read; the
            // Responses arm forwards the raw Response.usage instead, whose
            // detail lives at input_tokens_details.cached_tokens. Non-stream,
            // so reading the message directly is safe (no concat re-sum).
            cacheRead +=
              Number(um.input_token_details?.cache_read) ||
              Number(um.input_tokens_details?.cached_tokens) ||
              0;
            cacheCreation += Number(um.input_token_details?.cache_creation) || 0;
            found = true;
            break;
          }
        }
        if (found) {
          _stashLangchainUsageFromMessage(
            { usage_metadata: { input_tokens: inTok, output_tokens: outTok } },
            order,
            { cached_tokens: cacheRead, cache_creation_tokens: cacheCreation },
          );
        }
      }
    } catch {
      /* fail-open: usage stash is best-effort */
    }
    // G3-O1: generate-path reasoning capture — deliberately NOT gated on the
    // hasWireUsage guard above (ChatOpenAI always has llmOutput.tokenUsage,
    // so that block never runs) and a slot-level write ONLY: creating a
    // `.usage` stash here would flip telemetry's preferStash / stream-guard
    // decisions; in/out/cached must stay attr-derived.
    try {
      if (Array.isArray(result?.generations)) {
        let reasoning = 0;
        for (let genList of result.generations) {
          if (!Array.isArray(genList)) genList = [genList];
          for (const gen of genList) {
            const um = gen?.message?.usage_metadata;
            if (!um || typeof um !== "object") continue;
            // First usage-bearing candidate per outer list — candidates carry
            // the SAME full-call usage_metadata (see the stash loop above).
            reasoning += Number(um.output_token_details?.reasoning) || 0;
            break;
          }
        }
        if (reasoning > 0) {
          const key = order ?? Math.max(0, session.spanCounter - 1);
          const compKey = `${session.traceId}:${key}`;
          if (!session._pendingCompositions[compKey]) {
            session._pendingCompositions[compKey] = {};
          }
          const prev =
            Number(session._pendingCompositions[compKey].reasoning_tokens) || 0;
          if (reasoning > prev) {
            session._pendingCompositions[compKey].reasoning_tokens = reasoning;
          }
        }
      }
    } catch {
      /* fail-open: reasoning stash is best-effort */
    }
  } catch {
    // fail-safe
  }
}

/**
 * Stream-path cache aggregation overrides for LangChain usage stash.
 *
 * Anthropic (via @langchain/anthropic) often re-states absolute
 * cache_read / cache_creation on more than one stream chunk. LC's
 * AIMessageChunk.concat *sums* those fields, which double-counts.
 * Stream path therefore tracks max-across-chunks and passes the running
 * max here — never re-reads concat'd msg cache details.
 *
 * Production streamIterator always supplies this. When omitted (test-only /
 * defensive), cache fields stay 0 rather than trusting concat sums.
 */
type LangchainStreamCacheAgg = {
  cached_tokens: number;
  cache_creation_tokens: number;
};

/**
 * Stash concat'd LangChain stream usage under the reserved span order.
 *
 * MUST be updated after each accumulated chunk (not only in EOS finally):
 * BaseChatModel._streamIterator calls handleLLMEnd (→ span.end → onEnd
 * process.nextTick) *inside* the final `it.next()` *before* our generator
 * `finally` runs. Node drains nextTick before remaining promise microtasks,
 * so an EOS-only stash is too late and telemetry logs last-chunk under-bill.
 * Number copy is O(1) — unlike composition hashing, per-chunk is safe.
 *
 * input/output: from concat'd msg (sum of deltas — under-bill fix).
 * cache fields: from cacheAgg max-across-chunks on the stream path only.
 */
function _stashLangchainUsageFromMessage(
  msg: any,
  order?: number,
  cacheAgg?: LangchainStreamCacheAgg,
): void {
  try {
    if (!msg) return;
    const um = msg?.usage_metadata;
    if (!um || typeof um !== "object") return;
    const input_tokens = Number(um.input_tokens) || 0;
    const output_tokens = Number(um.output_tokens) || 0;
    // Never read cache from concat'd msg on the stream path — absolute
    // cache fields re-stated across chunks would sum to ~2N. Omit → 0.
    const cached_tokens =
      cacheAgg != null ? Number(cacheAgg.cached_tokens) || 0 : 0;
    const cache_creation_tokens =
      cacheAgg != null ? Number(cacheAgg.cache_creation_tokens) || 0 : 0;
    // G3-O1: reasoning from LC-standard output_token_details. MAX across
    // stash refreshes, never sum — concat() already SUMS reasoning across
    // chunks (a provider restating absolutes would double-sum), and max also
    // keeps a details-less refresh from clobbering a stashed value.
    // Slot-level field (not on `.usage`, which is REPLACED every refresh);
    // BEFORE the all-zero early return below so a refresh carrying only
    // reasoning still lands. Slot created only on a positive count.
    try {
      const r = Number(um.output_token_details?.reasoning) || 0;
      if (r > 0) {
        const session = getCurrentSession();
        const key = order ?? Math.max(0, session.spanCounter - 1);
        const compKey = `${session.traceId}:${key}`;
        if (!session._pendingCompositions[compKey]) {
          session._pendingCompositions[compKey] = {};
        }
        const prev =
          Number(session._pendingCompositions[compKey].reasoning_tokens) || 0;
        if (r > prev) {
          session._pendingCompositions[compKey].reasoning_tokens = r;
        }
      }
    } catch {
      // fail-open: reasoning stash is best-effort
    }
    if (
      !(
        input_tokens > 0 ||
        output_tokens > 0 ||
        cached_tokens > 0 ||
        cache_creation_tokens > 0
      )
    ) {
      return;
    }
    const session = getCurrentSession();
    const key = order ?? Math.max(0, session.spanCounter - 1);
    const compKey = `${session.traceId}:${key}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    // Tagged so telemetry skips OpenAI N1 add-cached-back.
    session._pendingCompositions[compKey].usage = {
      usage_source: "langchain_message",
      input_tokens,
      output_tokens,
      cached_tokens,
      cache_creation_tokens,
    };
  } catch {
    // fail-safe: usage stash is best-effort
  }
}

/**
 * Capture response composition (+ usage when present) from the fully-accumulated
 * streaming message (AIMessageChunk). Composition is built ONCE at end-of-stream
 * (from the streamIterator wrapper's `finally`) — rebuilding it per-chunk was
 * O(n²). Usage is also stashed here as a safety net for early-break paths;
 * the hot path refreshes usage after every chunk via
 * `_stashLangchainUsageFromMessage` so it is present before handleLLMEnd.
 *
 * `order` must be the same reserved span order used for prompt capture /
 * onStart; falling back to spanCounter-1 is only for un-threaded callers.
 *
 * `cacheAgg` MUST be the same max-across-chunks running totals from the
 * streamIterator loop (including on full drain). Re-reading concat'd
 * msg cache fields here would overwrite correct max with 2N and silently
 * no-op the cache double-count fix on the primary path.
 */
function _captureLangchainResponseFromMessage(
  msg: any,
  order?: number,
  cacheAgg?: LangchainStreamCacheAgg,
): void {
  try {
    if (!msg) return;
    const session = getCurrentSession();
    const key = order ?? Math.max(0, session.spanCounter - 1);
    const compKey = `${session.traceId}:${key}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }

    // Composition — independent of usage (empty content still bills tokens).
    try {
      const comp = buildResponseComposition("langchain", {
        generations: [[{ message: msg }]],
      });
      if (comp.length > 0) {
        session._pendingCompositions[compKey].response = comp;
      }
    } catch {
      // fail-safe: composition is best-effort
    }

    // Stash tool-call ids from the folded AIMessage (stream path). REPLACE on
    // every capture so a no-tool turn clears stale ids. Fail-open.
    try {
      session.setPendingToolCalls(extractPendingToolCalls("langchain", msg));
    } catch {
      /* fail-open */
    }

    // Usage safety net (primary path is per-chunk stash above).
    // Forward cacheAgg — never re-derive cache from concat'd msg.
    _stashLangchainUsageFromMessage(msg, order, cacheAgg);
  } catch {
    // fail-safe
  }
}

/**
 * Pre-flight context for the LangChain chat wrappers. The wrapped args
 * carry messages, not a request body, so model + provider come off the chat
 * model instance. The model is a matching-only HINT (never merged into a
 * body — a REROUTE never applies on this path; the enforce-mode refusal is
 * recorded as REROUTE_REJECTED `unappliable_call_shape`).
 *
 * Provider is omitted when `_deriveLangChainProvider` can't resolve it: the
 * literal "langchain" is not a provider slug, and sending it would make every
 * REROUTE directive look cross-provider (spurious
 * `cross_provider_unsupported` rejections).
 *
 * Never throws — a hostile instance degrades to the previous bare check.
 */
function _langchainChatCheckCtx(
  instance: any,
): { model?: string; provider?: string } {
  try {
    const raw = instance?.model ?? instance?.modelName;
    return {
      model: typeof raw === "string" && raw ? raw : undefined,
      provider: _deriveLangChainProvider(instance),
    };
  } catch {
    return {};
  }
}

function _setLangchainWrapper(
  obj: any,
  methodName: string,
  original: Function,
  kind: "async" | "streamIterator",
): void {
  if (kind === "async") {
    obj[methodName] = _withCallObsScope(async function (
      this: any,
      ...args: any[]
    ): Promise<any> {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const session: any = getCurrentSession();
      if (session.inLangchain) {
        return await original.apply(this, args);
      }
      const ctx = _langchainChatCheckCtx(this);
      await _runAsyncCheck(null, ctx.provider, null, true, false, ctx.model);
      // Stash attempt context so a failed call lands with a real
      // model/provider (this wrapper otherwise has no failure emission at
      // all — the LC ERROR span carries zero usage attrs and is dropped by
      // telemetry's zero-usage gate). Synthetic [{model}] args:
      // _stashAttemptContext reads args[0].model — same precedent as the
      // Vercel AI SDK path. No wire key exists at this layer.
      _stashAttemptContext(session, ctx.provider ?? "", [{ model: ctx.model }]);
      // Snapshot the attempt context: the _attempted_* slots are
      // session-global, so a concurrent same-session call may overwrite them
      // before this call fails. (Nested provider calls under LC short-circuit
      // on the inLangchain guard BEFORE their own stash, so only genuine
      // concurrency can — the snapshot handles it.)
      const _attemptSnap = {
        model: session._attempted_model,
        provider: session._attempted_provider,
        operation: session._attempted_operation,
        shape: session._attempted_shape,
        wireKey: session._attempted_wire_key,
      };
      // Reserve the span order for this LangChain call up front so the
      // instrumentor's onStart opens its span with exactly this order and both
      // the prompt and response stashes key by the same value — concurrent
      // same-session calls can't cross-attribute (see
      // tests/concurrentAttribution.test.ts).
      const order = session.nextSpanOrder();
      _captureLangchainPrompt(args, this, order);
      session.enterLangchain();
      const _callStart = Date.now();
      let result: any;
      try {
        result = await runWithReservedSpanOrder(
          { order, consumed: false },
          () => original.apply(this, args),
        );
      } catch (err) {
        // Provider failure — emit the llm failure row, then rethrow the
        // ORIGINAL error by identity so the app sees exactly the provider's
        // error. Every added statement is fail-open (golden rule); this
        // branch runs entirely inside _withCallObsScope (ALS survives await),
        // so no obs-key re-entry is needed — same as the openai
        // request-failure catch.
        try {
          session._call_outcome = buildCallOutcome(err, Date.now() - _callStart);
          // Marker for symmetry/insurance with the streamIterator branch: the
          // LC ERROR span (handleLLMError) would be gate-dropped anyway with
          // zero usage attrs, but the marker guarantees exactly one row.
          ((session._failedStreamCompKeys ??= new Set()) as Set<string>).add(
            `${session.traceId}:${order}`,
          );
          // Conditional-restore protocol (see the stream tap): put THIS
          // call's attempt context back for the emit; hand a newer call's
          // context back afterwards if we borrowed the slots.
          const cur = {
            model: session._attempted_model,
            provider: session._attempted_provider,
            operation: session._attempted_operation,
            shape: session._attempted_shape,
            wireKey: session._attempted_wire_key,
          };
          const overwritten =
            cur.model !== _attemptSnap.model ||
            cur.provider !== _attemptSnap.provider ||
            cur.operation !== _attemptSnap.operation ||
            cur.shape !== _attemptSnap.shape ||
            cur.wireKey !== _attemptSnap.wireKey;
          session._attempted_model = _attemptSnap.model;
          session._attempted_provider = _attemptSnap.provider;
          session._attempted_operation = _attemptSnap.operation;
          session._attempted_shape = _attemptSnap.shape;
          session._attempted_wire_key = _attemptSnap.wireKey;
          _emitCallFailureLog(getClient(), session);
          if (overwritten) {
            session._attempted_model = cur.model;
            session._attempted_provider = cur.provider;
            session._attempted_operation = cur.operation;
            session._attempted_shape = cur.shape;
            session._attempted_wire_key = cur.wireKey;
          }
        } catch {
          // fail-open: failure logging must never affect error propagation
        }
        throw err;
      } finally {
        session.exitLangchain();
      }
      _captureLangchainResponse(result, order);
      return result;
    });
    return;
  }

  // kind === "streamIterator"
  // BaseChatModel.prototype._streamIterator is an async-generator function:
  // calling it returns the async iterator synchronously. Replace it with an
  // async generator wrapper so we can await the pre-flight check before
  // yielding the first chunk, and hold the inLangchain guard for the entire
  // iteration. This is the path agents / RunnableSequence use internally.
  //
  // Span-order reservation: ALS does not survive yield-to-customer, so each
  // inner pull that can open the LC LLM span must run under
  // runWithReservedSpanOrder with a shared reservation (first onStart consumes
  // `order`; nested spans allocate fresh). EOS stashes composition + usage
  // under the same `order` so telemetry onEnd can prefer concat'd
  // usage_metadata over last-chunk Traceloop llmOutput under-billing.
  obj[methodName] = async function* (this: any, ...args: any[]): AsyncGenerator<any> {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const session: any = getCurrentSession();
    if (session.inLangchain) {
      yield* (original.apply(this, args) as AsyncIterable<any>);
      return;
    }
    // Monotonic anchor for the failure outcome's elapsed time (see the catch
    // below) — immune to clock steps.
    const _streamStartMono = performance.now();
    // Async-generator body: a single ALS scope cannot span customer-driven
    // yields (each resumption runs in the CONSUMER's context), so instead of
    // the usual whole-body _withCallObsScope this wrapper mints one key up
    // front and re-enters it around (a) the pre-flight — tagging its pushed
    // observations — and (b) each reserved-order pull below — the LC LLM
    // span's onStart fires inside a pull and stamps the key for the deferred
    // drain. Reuse-if-exists keeps SDK-internal delegation on the caller's key.
    const _obsKey =
      _currentObsKey() ??
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (typeof (state as any).newObsKey === "function" ? (state as any).newObsKey() : null);
    const ctx = _langchainChatCheckCtx(this);
    await _reenterObsScope(_obsKey, () =>
      _runAsyncCheck(null, ctx.provider, null, true, false, ctx.model),
    );
    // Stash attempt context so a failed stream lands with a real
    // model/provider (this wrapper otherwise has no failure emission at all).
    // Synthetic [{model}] args: _stashAttemptContext reads args[0].model —
    // same precedent as the Vercel AI SDK path. No wire key at this layer.
    _stashAttemptContext(session, ctx.provider ?? "", [{ model: ctx.model }]);
    // Snapshot the attempt context: the _attempted_* slots are session-global,
    // so a concurrent same-session call may overwrite them before this stream
    // fails. (Nested provider calls under LC short-circuit on the inLangchain
    // guard BEFORE their own stash, so only genuine concurrency can — the
    // snapshot handles it.)
    const _attemptSnap = {
      model: session._attempted_model,
      provider: session._attempted_provider,
      operation: session._attempted_operation,
      shape: session._attempted_shape,
      wireKey: session._attempted_wire_key,
    };
    // Reserve BEFORE any pull that can open the OTel LC span.
    const order = session.nextSpanOrder();
    const reservation = { order, consumed: false };
    _captureLangchainPrompt(args, this, order);
    session.enterLangchain();
    // Accumulate streamed chunks into a single AIMessageChunk so we can
    // emit a real response composition (with tool_calls / content) + usage
    // for the current LLM span. The JS LangChain instrumentor's span attributes
    // (gen_ai.output.messages) report tool-calling assistant turns as
    // `parts:[{type:"text", content:""}], finish_reason:"tool_call"` — they
    // don't carry the tool name/args, so the telemetry-side fallback can
    // only emit a placeholder. The accumulated composition must be in place
    // before the LangChain span's onEnd schedules its deferred (nextTick) log.
    //
    // Cache fields: LC concat *sums* absolute cache_read/cache_creation when
    // Anthropic re-states them on multiple chunks → ~2× over-bill. Track
    // max-across-chunks here and pass via cacheAgg on every stash (incl.
    // finally). input/output still come from concat (sum of deltas).
    let acc: any = null;
    let maxCacheRead = 0;
    let maxCacheCreation = 0;
    let it: AsyncIterator<any> | undefined;
    try {
      // Explicit iterator so each pull runs under ALS reservation. Bare
      // `for await ... of original` would not re-enter the reservation on
      // each customer-driven next().
      const iterable = original.apply(this, args) as AsyncIterable<any>;
      it = iterable[Symbol.asyncIterator]();
      for (;;) {
        // Obs-key re-entry per pull (see the mint above): the LC LLM span
        // starts inside a pull, so onStart must see this call's key.
        const step = await _reenterObsScope(_obsKey, () =>
          runWithReservedSpanOrder(reservation, () => it!.next()),
        );
        if (step.done) break;
        const chunk = step.value;
        // Max cache from the *incoming* chunk before concat — independent of
        // concat success so a concat throw cannot skip the update (§5.3).
        try {
          const d = chunk?.usage_metadata?.input_token_details;
          if (d && typeof d === "object") {
            const cr = Number(d.cache_read) || 0;
            const cc = Number(d.cache_creation) || 0;
            if (cr > maxCacheRead) maxCacheRead = cr;
            if (cc > maxCacheCreation) maxCacheCreation = cc;
          }
        } catch {
          // fail-open: cache aggregation must never break the stream
        }
        try {
          // Incrementally merge chunks — preserves LangChain's chunk-merge
          // semantics (usage_metadata / tool_call_chunks / response_metadata)
          // which we must not hand-roll. This per-chunk step is cheap (V8 rope
          // strings); the expensive full-composition rebuild is done ONCE below.
          if (acc == null) acc = chunk;
          else if (chunk != null && typeof acc.concat === "function") {
            acc = acc.concat(chunk);
          }
          // Refresh usage stash AFTER each concat so totals exist before
          // handleLLMEnd (runs on the next pull after the last yield) ends
          // the OTel span and schedules onEnd's nextTick log. Cache from
          // max*, never from concat'd msg cache details.
          if (acc != null) {
            _stashLangchainUsageFromMessage(acc, order, {
              cached_tokens: maxCacheRead,
              cache_creation_tokens: maxCacheCreation,
            });
          }
        } catch {
          // fail-safe: accumulation is best-effort
        }
        // OUTSIDE reservation — customer must not inherit ALS.
        yield chunk;
      }
    } catch (err) {
      // Provider failure mid-stream — emit the llm failure row, then rethrow
      // the ORIGINAL error by identity so the app sees exactly the provider's
      // error. Emission is entirely fail-open (golden rule) and synchronous.
      // No once-guard needed: a single generator body runs this catch at most
      // once. (An early customer break/return skips the catch — finally only,
      // no failure row; a consumer-injected gen.throw(e) resumes at the yield
      // and IS caught — same accepted semantics as the manual-stream
      // wrappers.)
      try {
        // Mark the compKey failed FIRST and SYNCHRONOUSLY: LC's
        // handleLLMError already ended the span and scheduled the deferred
        // onEnd log (nextTick) BEFORE this catch sees the rejection —
        // microtask-before-nextTick ordering means the marker is observed.
        // This is what kills the mislabeled-success row when streamed chunks
        // carried usage_metadata before the failure (the langchain_message
        // stash would otherwise pass telemetry's zero-usage gate).
        ((session._failedStreamCompKeys ??= new Set()) as Set<string>).add(
          `${session.traceId}:${order}`,
        );
        const elapsedMs = Math.round(
          Math.max(0, performance.now() - _streamStartMono),
        );
        session._call_outcome = buildCallOutcome(err, elapsedMs);
        // Conditional-restore protocol (see the stream tap): put THIS call's
        // attempt context back for the emit; hand a newer call's context back
        // afterwards if we borrowed the slots.
        const cur = {
          model: session._attempted_model,
          provider: session._attempted_provider,
          operation: session._attempted_operation,
          shape: session._attempted_shape,
          wireKey: session._attempted_wire_key,
        };
        const overwritten =
          cur.model !== _attemptSnap.model ||
          cur.provider !== _attemptSnap.provider ||
          cur.operation !== _attemptSnap.operation ||
          cur.shape !== _attemptSnap.shape ||
          cur.wireKey !== _attemptSnap.wireKey;
        session._attempted_model = _attemptSnap.model;
        session._attempted_provider = _attemptSnap.provider;
        session._attempted_operation = _attemptSnap.operation;
        session._attempted_shape = _attemptSnap.shape;
        session._attempted_wire_key = _attemptSnap.wireKey;
        // Emit inside this call's obs scope — the generator body is resumed
        // from the CONSUMER's async context, so re-enter the key this branch
        // minted up front (the same one every pull re-enters).
        _reenterObsScope(_obsKey, () => _emitCallFailureLog(getClient(), session));
        if (overwritten) {
          session._attempted_model = cur.model;
          session._attempted_provider = cur.provider;
          session._attempted_operation = cur.operation;
          session._attempted_shape = cur.shape;
          session._attempted_wire_key = cur.wireKey;
        }
      } catch {
        // fail-open: failure logging must never affect error propagation
      }
      throw err;
    } finally {
      // Propagate early customer break/return/throw to the underlying provider
      // iterator so it can release resources — the manual `.next()` loop
      // doesn't auto-close the way `for await … of` did. Idempotent on a
      // fully-drained iterator; fail-open on any throw.
      try {
        await it?.return?.();
      } catch {
        // fail-open: closing the source stream must never throw out
      }
      // Build + stash response composition + usage exactly ONCE from the fully
      // accumulated message. The `finally` runs once per stream on every exit
      // path (full drain, early break, mid-stream throw). Same reserved `order`
      // as prompt capture / onStart. MUST pass the same max* cacheAgg — re-
      // reading concat cache here overwrites correct max with 2N on full drain.
      // Guarded so capture errors never skip guard release.
      try {
        _captureLangchainResponseFromMessage(acc, order, {
          cached_tokens: maxCacheRead,
          cache_creation_tokens: maxCacheCreation,
        });
      } catch {
        // fail-safe: capture is best-effort, must never block guard release
      }
      session.exitLangchain();
    }
  };
}


/**
 * ═══════════════════════════════════════════════════════════════════
 * LlamaIndex wrappers — for @llamaindex/openai, @llamaindex/anthropic,
 * @llamaindex/google. {OpenAI,Anthropic,Gemini}.{chat,complete}.
 *
 * LlamaIndex is a framework: its provider LLM classes call the underlying
 * provider SDK. Telemetry CANNOT reliably flow through OpenLLMetry's inner-
 * provider spans because each `@llamaindex/<provider>` package bundles its
 * own copy of the underlying SDK in nested `node_modules` — OpenLLMetry
 * only patches the customer's top-level install. So these wrappers extract
 * usage manually from `result.raw` (OpenAI/Anthropic/Gemini response
 * shape) and log directly. The inLlamaIndex guard still fires so that if
 * the customer DOES have a top-level provider SDK that LlamaIndex happens
 * to use (e.g. via dedupe), the inner enforcer wrapper short-circuits.
 *
 * `chat()` returns either Promise<ChatResponse> (non-stream) or
 * Promise<AsyncIterable<ChatResponseChunk>> (stream:true). The wrapper
 * detects streaming at runtime by checking the resolved value.
 * ═══════════════════════════════════════════════════════════════════
 */

/** Map a LlamaIndex provider class instance to a normalized provider name. */
function _llamaIndexProvider(thisArg: any): string {
  const cls = String(thisArg?.constructor?.name ?? "").toLowerCase();
  if (cls.includes("anthropic")) return "anthropic";
  if (cls.includes("gemini") || cls.includes("google")) return "google";
  // @llamaindex/openai's OpenAIResponses is a SIBLING of OpenAI (its own
  // chat/streamChat against /v1/responses) with a different usage shape —
  // map it to the Responses pseudo-provider (serving identity stays openai).
  if (cls.includes("responses")) return "openai_responses";
  return "openai"; // OpenAI default — also covers OpenAI-compatible subclasses
}

/**
 * True when `obj` is a non-null object carrying at least one positive numeric
 * token count at the top level. Gates verbatim-usage forwarding: an empty /
 * all-zero usage object is not worth forwarding (it would map to zeros
 * server-side), so the caller falls back to the positional counts instead.
 */
function _liUsageHasPositiveCount(obj: unknown): boolean {
  try {
    if (!obj || typeof obj !== "object") return false;
    for (const v of Object.values(obj as Record<string, unknown>)) {
      if (typeof v === "number" && Number.isFinite(v) && v > 0) return true;
    }
    return false;
  } catch {
    return false;
  }
}

/**
 * Wrap a provider's verbatim LlamaIndex usage object as a {shape, raw} block the
 * server-side usage-mapper prices directly, or undefined to fall back to the
 * positional counts. Forwarding verbatim is what lets the mapper apply each
 * provider's own cache semantics: Gemini usageMetadata is cache-INCLUSIVE (the
 * mapper subtracts cachedContentTokenCount itself), anthropic input_tokens is
 * cache-EXCLUSIVE, OpenAI prompt_tokens is cache-inclusive. Fail-safe: any
 * surprise shape degrades to undefined, never throws.
 *
 * The forwarded raw is a JSON snapshot, not a reference: the log POST
 * stringifies in the background, so a value that only fails at stringify time
 * (cycles, BigInt, hostile getters) would otherwise drop the whole row there —
 * snapshotting here surfaces that failure NOW and falls back to the positional
 * counts instead. It also freezes the block against later mutation of the
 * source object (e.g. the anthropic stash accumulator).
 */
function _buildLlamaIndexVerbatimUsage(
  shape: string,
  rawUsage: unknown,
): { shape: string; raw: unknown } | undefined {
  try {
    if (!rawUsage || typeof rawUsage !== "object") return undefined;
    if (!_liUsageHasPositiveCount(rawUsage)) return undefined;
    return { shape, raw: JSON.parse(JSON.stringify(rawUsage)) };
  } catch {
    return undefined;
  }
}

/**
 * Extract token usage from a LlamaIndex ChatResponse / streaming chunk.
 * Pulls from `result.raw` (the underlying provider's response object).
 *
 * `verbatimUsage`, when set, is the provider's raw usage object tagged with its
 * server-side shape — the log path forwards it so the mapper prices cache the
 * provider's own way. The positional counts remain the fallback used only when
 * no verbatim block is available. Each branch is guarded so a hostile/missing
 * raw shape degrades to the zero fallback rather than throwing.
 */
function _extractLlamaIndexUsage(
  innerProvider: string,
  result: any,
  thisArg: any,
): {
  model: string;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  verbatimUsage?: { shape: string; raw: unknown };
} {
  let raw: any = {};
  try {
    raw = result?.raw ?? {};
  } catch {
    raw = {};
  }
  const fallbackModel = String(thisArg?.model ?? thisArg?.modelName ?? "unknown");

  if (innerProvider === "anthropic") {
    try {
      const u = raw.usage ?? {};
      const cachedTokens = Number(u.cache_read_input_tokens ?? 0);
      const inputTokens = Number(u.input_tokens ?? 0);
      const outputTokens = Number(u.output_tokens ?? 0);
      const model = String(raw.model ?? fallbackModel);
      // anthropic input_tokens is cache-EXCLUSIVE and mapAnthropicMessages knows
      // that — forward the usage verbatim rather than the positional counts.
      const verbatimUsage = _buildLlamaIndexVerbatimUsage("anthropic_messages", u);
      return { model, inputTokens, outputTokens, cachedTokens, verbatimUsage };
    } catch {
      return { model: fallbackModel, inputTokens: 0, outputTokens: 0, cachedTokens: 0 };
    }
  }

  if (innerProvider === "google") {
    try {
      const um = raw.usageMetadata ?? {};
      const promptTokens = Number(um.promptTokenCount ?? 0);
      const cachedTokens = Number(um.cachedContentTokenCount ?? 0);
      const toolUseTokens = Number(um.toolUsePromptTokenCount ?? 0);
      const inputTokens = Math.max(0, promptTokens - cachedTokens) + toolUseTokens;
      const outputTokens =
        Number(um.candidatesTokenCount ?? 0) + Number(um.thoughtsTokenCount ?? 0);
      const model = String(raw.modelVersion ?? fallbackModel);
      // usageMetadata is cache-INCLUSIVE; mapGoogleGenAI subtracts
      // cachedContentTokenCount itself, so forward it verbatim (camelCase,
      // promptTokensDetails modality arrays and thoughtsTokenCount are all
      // mapper-known). The pre-netted positional inputTokens would otherwise be
      // netted a second time by the mapper — the under-bill this fixes.
      const verbatimUsage = _buildLlamaIndexVerbatimUsage("google_genai", um);
      return { model, inputTokens, outputTokens, cachedTokens, verbatimUsage };
    } catch {
      return { model: fallbackModel, inputTokens: 0, outputTokens: 0, cachedTokens: 0 };
    }
  }

  if (innerProvider === "openai_responses") {
    // OpenAI Responses API via OpenAIResponses. Non-stream: `raw` is the
    // Response object with `usage` directly on it. Stream: `raw` is the
    // terminal `response.completed` event — the Response (and its usage)
    // nest under `.response`. input_tokens already INCLUDES cached tokens,
    // so subtract; output_tokens already includes reasoning_tokens
    // (mirrors the direct-SDK openai_responses mapping in _extractUsage).
    try {
      const resp = raw?.response?.usage ? raw.response : raw;
      const u = resp.usage ?? {};
      const inputT = Number(u.input_tokens ?? 0);
      const cachedTokens = Number(u.input_tokens_details?.cached_tokens ?? 0);
      const inputTokens = Math.max(0, inputT - cachedTokens);
      const outputTokens = Number(u.output_tokens ?? 0);
      const model = String(resp.model ?? fallbackModel);
      // input_tokens is cache-INCLUSIVE and the openai_responses mapper
      // subtracts cached_tokens itself — forward the usage verbatim.
      const verbatimUsage = _buildLlamaIndexVerbatimUsage("openai_responses", u);
      return { model, inputTokens, outputTokens, cachedTokens, verbatimUsage };
    } catch {
      return { model: fallbackModel, inputTokens: 0, outputTokens: 0, cachedTokens: 0 };
    }
  }

  // OpenAI default — OpenAI-shaped usage. cached prompt tokens are folded
  // into prompt_tokens, so subtract to avoid double-counting. `completion_tokens`
  // already includes `completion_tokens_details.reasoning_tokens`, so they must
  // not be added again.
  try {
    const u = raw.usage ?? {};
    const promptTokens = Number(u.prompt_tokens ?? 0);
    const cachedTokens = Number(u.prompt_tokens_details?.cached_tokens ?? 0);
    const inputTokens = Math.max(0, promptTokens - cachedTokens);
    const outputTokens = Number(u.completion_tokens ?? 0);
    const model = String(raw.model ?? fallbackModel);
    // prompt_tokens is cache-INCLUSIVE (same shape the native OpenAI path ships)
    // and mapOpenAIChat subtracts cached_tokens itself — forward verbatim.
    const verbatimUsage = _buildLlamaIndexVerbatimUsage("openai_compatible_chat", u);
    return { model, inputTokens, outputTokens, cachedTokens, verbatimUsage };
  } catch {
    return { model: fallbackModel, inputTokens: 0, outputTokens: 0, cachedTokens: 0 };
  }
}

/**
 * Pass-through tap on an inner @anthropic-ai/sdk stream consumed by
 * @llamaindex/anthropic. Merges message_start/message_delta usage (the same
 * merge _tapAnthropicStreamBypass does) into session._pendingLlamaIndexUsage —
 * it does NOT log; _logLlamaIndex consumes the stash as a zero-usage fallback.
 * Non-destructive: the customer's (LlamaIndex's) iteration sees every original
 * event untouched, and any tap failure returns the stream as-is.
 */
function _tapLlamaIndexAnthropicUsage(stream: any, session: any): any {
  try {
    if (!stream || typeof stream[Symbol.asyncIterator] !== "function") return stream;
    const acc = { model: "", usage: {} as Record<string, unknown> };
    const publish = (): void => {
      try {
        if (Object.keys(acc.usage).length > 0) {
          session._pendingLlamaIndexUsage = { model: acc.model, usage: acc.usage };
        }
      } catch {
        // fail-open
      }
    };
    const getOrig = stream[Symbol.asyncIterator].bind(stream);
    Object.defineProperty(stream, Symbol.asyncIterator, {
      configurable: true,
      writable: true,
      value: function (): AsyncIterator<any> {
        const inner: AsyncIterator<any> = getOrig();
        return {
          async next(): Promise<IteratorResult<any>> {
            const r = await inner.next();
            try {
              if (!r.done) {
                const ev: any = r.value;
                if (ev?.type === "message_start" && ev.message) {
                  if (ev.message.model) acc.model = String(ev.message.model);
                  if (ev.message.usage && typeof ev.message.usage === "object") {
                    acc.usage = { ...acc.usage, ...ev.message.usage };
                  }
                  publish();
                } else if (ev?.type === "message_delta" && ev.usage && typeof ev.usage === "object") {
                  acc.usage = { ...acc.usage, ...ev.usage };
                  // message_delta is the terminal usage event — publish so the
                  // stash is complete before LlamaIndex's stream ends.
                  publish();
                }
              }
            } catch {
              // fail-open: a telemetry tap must never break iteration
            }
            return r;
          },
          return: inner.return ? (v?: any) => inner.return!(v as any) : undefined,
          throw: inner.throw ? (e?: any) => inner.throw!(e) : undefined,
        } as AsyncIterator<any>;
      },
    });
  } catch {
    // fail-open: return the stream untouched
  }
  return stream;
}

function _captureLlamaIndexPromptAt(args: any[], thisArg: any, order: number): void {
  try {
    const session = getCurrentSession();
    const params = (args && args[0]) || {};
    const messages = (params as any).messages ?? (params as any).prompt;
    if (messages == null) return;
    const comp = buildPromptComposition("llamaindex", { messages });
    const compKey = `${session.traceId}:${order}`;
    if (!session._pendingCompositions[compKey]) {
      session._pendingCompositions[compKey] = {};
    }
    if (comp.length > 0) {
      session._pendingCompositions[compKey].prompt = comp;
    }
    const modelName = thisArg?.model ?? thisArg?.modelName ?? thisArg?.model_name;
    if (typeof modelName === "string" && modelName) {
      session._pendingCompositions[compKey].model = modelName;
    }
  } catch {
    // fail-safe
  }
}

function _captureLlamaIndexResponseAt(result: any, order: number): void {
  try {
    const session = getCurrentSession();
    // Composition and the tool-id stash are independent (layout mirrors
    // _captureLangchainResponse): the old early-return on an empty composition
    // would have skipped the stash below, and a composition throw must not
    // skip it either.
    try {
      const comp = buildResponseComposition("llamaindex", result);
      if (comp.length > 0) {
        const compKey = `${session.traceId}:${order}`;
        if (!session._pendingCompositions[compKey]) {
          session._pendingCompositions[compKey] = {};
        }
        session._pendingCompositions[compKey].response = comp;
      }
    } catch {
      // fail-safe: composition is best-effort
    }
    // Stash tool-call ids so wrapLlamaIndexTools / toolSpan() can
    // auto-correlate them by name. REPLACE on every response (including [])
    // so a no-tool turn clears stale ids from a prior agent-loop iteration.
    try {
      session.setPendingToolCalls(extractPendingToolCalls("llamaindex", result));
    } catch {
      /* fail-open */
    }
  } catch {
    // fail-safe
  }
}

/**
 * Build the full log payload for a completed LlamaIndex call and dispatch
 * it via tp.log() (background fire-and-forget).
 */
const _logLlamaIndex = failSafeSync(function (
  thisArg: any,
  result: any,
  order: number,
  spanName: string | null,
  startTime: Date,
  latency: TPLatency | null = null,
  // The emitting call's obs key. It keys BOTH this row's `_tp_routing` stamp
  // AND the keyed observation drain below. The streaming caller runs this from
  // a generator `finally` in the CONSUMER's async context, where the ALS read
  // below yields null (or, worse, a sibling's key) — so it passes the key
  // captured at wrapper entry instead. The call site must NOT use
  // `_reenterObsScope` around this function: a re-entered scope would let a
  // keyed drain/claim steal a sibling's records. Defaults to the live key,
  // which is correct for the non-streaming caller.
  obsKey?: string | null,
): void {
  const tp = getClient();
  if (!tp) return;

  const session = getCurrentSession();
  const innerProvider = _llamaIndexProvider(thisArg);
  const _li = _extractLlamaIndexUsage(innerProvider, result, thisArg);
  let { model, inputTokens, outputTokens, cachedTokens } = _li;
  // Verbatim provider usage forwarded to the mapper when present; falls back to
  // the positional counts synthesized client-side when undefined.
  let verbatimUsage = _li.verbatimUsage;

  // Anthropic-streaming fallback: @llamaindex/anthropic never surfaces usage
  // on its chunks, so the extraction above reads 0/0. The inner-stream tap
  // (_tapLlamaIndexAnthropicUsage) stashed the real message_start/message_delta
  // counts — consume them. Gated: only fills in when extraction yielded zero.
  try {
    const stash = (session as any)._pendingLlamaIndexUsage;
    if (
      stash?.usage &&
      innerProvider === "anthropic" &&
      inputTokens === 0 &&
      outputTokens === 0
    ) {
      const su: any = stash.usage;
      inputTokens = Number(su.input_tokens ?? 0);
      outputTokens = Number(su.output_tokens ?? 0);
      cachedTokens = Number(su.cache_read_input_tokens ?? 0);
      if (stash.model && (!model || model === "unknown")) model = String(stash.model);
      // The stash merged real message_start/message_delta usage (input_tokens,
      // output_tokens, cache_read_input_tokens); anthropic input_tokens is
      // cache-EXCLUSIVE so forward it verbatim for correct cache pricing.
      verbatimUsage = _buildLlamaIndexVerbatimUsage("anthropic_messages", su);
    }
    (session as any)._pendingLlamaIndexUsage = null; // consume once per call
  } catch {
    // fail-safe
  }

  const spanObj: Record<string, unknown> = {
    ...manualSpanIds(session),
    span_kind: "llm",
    span_name: spanName ?? model,
    span_order: order,
    start_time: startTime.toISOString(),
    end_time: new Date().toISOString(),
  };

  const metadata: Record<string, unknown> = { workflow_name: session.workflowName };
  if (session.sessionId) metadata.session_id = session.sessionId;
  // B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
  // session metadata WITHOUT it, then re-add it only when this row belongs
  // to the call that was actually rerouted (exact obs-key match).
  if (session.metadata) {
    copySessionMetadata(metadata, session.metadata);
  }
  // Normalize before any drain: `undefined` would hit drainObservations'
  // drain-all sentinel default and empty the ENTIRE global queue.
  const _rowObsKey: string | null = obsKey !== undefined ? obsKey : _currentObsKey();
  stampRoutingMarker(metadata, session, _rowObsKey);

  let promptComp: unknown[] = [];
  let responseComp: unknown[] = [];
  const compKey = `${session.traceId}:${order}`;
  const stash = session._pendingCompositions[compKey];
  if (stash) {
    promptComp = stash.prompt ?? [];
    responseComp = stash.response ?? [];
    delete session._pendingCompositions[compKey];
  }

  // Success duration: streamed calls carry the monotonic total from the
  // wrapper's latency block; non-streaming falls back to wall-clock.
  let durationMs = 0;
  try {
    durationMs = latency?.total_ms ?? Math.max(0, Date.now() - startTime.getTime());
  } catch {
    durationMs = 0;
  }

  // Keyed observation drain (would_block / would_reroute / reroute_rejected)
  // so success rows ship what the pre-flight minted. Observations only: a
  // body-less seam never owns a keyed local_decision, and a claim here would
  // only resolve through the untagged fallback and steal a sibling's decision.
  let observations: unknown[] = [];
  try {
    observations = state.drainObservations(_rowObsKey);
  } catch {
    observations = [];
  }

  tp.log(
    session.userId,
    session.paidPlan,
    session.workflowName,
    session.sessionId,
    model,
    innerProvider,
    inputTokens,
    outputTokens,
    cachedTokens,
    metadata,
    spanObj,
    promptComp,
    responseComp,
    {
      ...(verbatimUsage ? { usage: verbatimUsage } : {}),
      ...(latency ? { latency } : {}),
      ...(durationMs > 0
        ? { call_outcome: { status: "success", duration_ms: durationMs } }
        : {}),
      ...(observations.length > 0 ? { observations } : {}),
      planSource: session.planSource,
    },
  );
});

/**
 * Pass-through async iterator whose every pull runs inside the LlamaIndex
 * OTel-suppression window (see runWithLlamaIndexOtelSuppress).
 *
 * Why per-pull and not the wrapper's own window: @llamaindex/anthropic's
 * `streamChat` is a LAZY async generator — the inner
 * `anthropic.messages.create({stream:true})` only executes at FIRST PULL,
 * which happens in the CONSUMER's async context (the wrapper's `await
 * original.apply()` window has already unwound). So the Traceloop provider
 * instrumentor's duplicate span for that HTTP call starts here, during a
 * `next()` — wrapping each pull in the window tags it for the onEnd drop
 * (F-23-1 double-billing).
 *
 * The shim must never alter stream semantics: rejections propagate by
 * identity through the delegated `next()`; `return()` is delegated so a
 * consumer early break still closes the source (and the guarded stream's
 * `finally` still emits its partial-success row); a consumer-injected
 * `throw()` still resumes at the source's yield. Fail-open: if constructing
 * the shim throws, the raw source is iterated instead — the duplicate span
 * then simply survives (the pre-fix over-count), never a broken stream.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _llamaIndexSuppressedIter(src: AsyncIterable<any>): AsyncIterable<any> {
  try {
    const it = (src as any)[Symbol.asyncIterator]();
    const out: any = {
      [Symbol.asyncIterator]() {
        return out;
      },
      next: (...a: any[]) =>
        runWithLlamaIndexOtelSuppress(() => it.next(...a)),
    };
    if (typeof it.return === "function") {
      out.return = (...a: any[]) =>
        runWithLlamaIndexOtelSuppress(() => it.return(...a));
    }
    if (typeof it.throw === "function") {
      out.throw = (...a: any[]) =>
        runWithLlamaIndexOtelSuppress(() => it.throw(...a));
    }
    return out;
  } catch {
    return src; // fail-open: duplicate span survives, stream untouched
  }
}

/**
 * Wrap a LlamaIndex stream so the inLlamaIndex guard stays active during
 * consumption, accumulate text deltas / tool-call options across chunks,
 * and emit a single manual log when the stream ends. The last raw chunk
 * with `.usage` (or `.usageMetadata` for Gemini) provides token counts.
 *
 * Why accumulate: LlamaIndex JS streams emit incremental chunks where the
 * final usage-only chunk has no `.message` — so passing it directly to
 * composition would fall through to the Tier-3 `complete_response` hash.
 * We synthesize a full ChatResponse from accumulated state instead.
 */
async function* _guardedLlamaIndexStream(
  stream: AsyncIterable<any>,
  session: any,
  thisArg: any,
  order: number,
  spanName: string | null,
  startTime: Date,
  reqStartMono?: number,
  obsKey: string | null = null,
  // Attempt-slot snapshot from the wrapper (see _setLlamaIndexWrapper) —
  // restored around the failure emit below via the conditional-restore
  // protocol. Optional so the guard degrades to a snapshot-less emit.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  attemptSnap?: Record<string, any>,
): AsyncIterable<any> {
  let textAcc = "";
  let lastOptions: any = undefined;
  let lastChunkWithUsage: any = null;
  let lastChunk: any = null;
  // This wrapper sees every streamed chunk, so it owns the latency markers
  // (is_streaming/ttft) for LlamaIndex calls — the inner provider wrapper is
  // short-circuited by the inLlamaIndex guard and can't provide them.
  let ttftMono: number | null = null;
  // Gates the finally's synthetic success emission: without it a provider
  // rejection at first pull would land as a success row, and a mid-stream
  // failure after a usage-bearing chunk (Gemini carries usage on every chunk)
  // as a PRICED success row.
  let failed = false;
  try {
    // Pulls routed through the suppression shim so a lazy adapter's
    // first-pull HTTP (and its instrumentor span) starts inside the
    // per-call OTel-suppression window — see _llamaIndexSuppressedIter.
    for await (const chunk of _llamaIndexSuppressedIter(stream)) {
      lastChunk = chunk;
      if (ttftMono === null && (typeof chunk?.delta === "string" || chunk?.options)) {
        ttftMono = performance.now();
      }
      if (typeof chunk?.delta === "string") textAcc += chunk.delta;
      if (chunk?.options) lastOptions = chunk.options;
      // Gemini puts usage on `raw.usageMetadata`; OpenAI on `raw.usage`.
      // OpenAIResponses chunks carry the raw stream event: usage arrives once,
      // on the terminal `response.completed` event at `raw.response.usage`
      // (earlier events carry `response.usage: null`, so the truthy test
      // skips them). Anthropic via @llamaindex/anthropic forwards only
      // content_block_delta events and never surfaces usage — the inner-stream
      // tap (_tapLlamaIndexAnthropicUsage) covers that combo via the session
      // stash.
      if (chunk?.raw?.usage || chunk?.raw?.usageMetadata || chunk?.raw?.response?.usage) {
        lastChunkWithUsage = chunk;
      }
      yield chunk;
    }
  } catch (err) {
    // Provider failure at/after first pull — for LAZY adapters
    // (@llamaindex/openai: streamChat is an async generator) `await
    // original.apply()` resolved BEFORE any HTTP, so the rejection first
    // surfaces here (401/400/429, mid-stream abort) and this catch turns it
    // into a failure row. EAGER adapters (@llamaindex/google 0.4.x: plain
    // async method that awaits before returning an iterator) and all
    // non-streaming calls reject at the wrapper's own await instead — its
    // eager catch emits those, and the two catches are mutually exclusive
    // (if the eager catch fired, no iterable was ever returned), so no
    // double emission. Emission is entirely fail-open (golden rule)
    // and the ORIGINAL error is rethrown by identity. A consumer early
    // break/return skips this catch (finally only — partial success row kept);
    // a consumer-injected gen.throw(e) resumes at the yield and IS caught —
    // same accepted semantics as the LangChain streamIterator catch.
    //
    // No _failedStreamCompKeys marker here: no deferred telemetry span can key
    // by this wrapper's `order` — the inner provider wrapper short-circuits on
    // inLlamaIndex and this path never reserves its order for an instrumentor
    // span, so the manual row is the only emitter and a marker entry would
    // just leak unconsumed.
    failed = true;
    try {
      const elapsedMs =
        reqStartMono != null
          ? Math.round(Math.max(0, performance.now() - reqStartMono))
          : 0;
      session._call_outcome = buildCallOutcome(err, elapsedMs);
      if (attemptSnap) {
        // Conditional-restore protocol (see the stream tap): put THIS call's
        // attempt context back for the emit; hand a newer call's context back
        // afterwards if we borrowed the slots.
        const cur = {
          model: session._attempted_model,
          provider: session._attempted_provider,
          operation: session._attempted_operation,
          shape: session._attempted_shape,
          wireKey: session._attempted_wire_key,
        };
        const overwritten =
          cur.model !== attemptSnap.model ||
          cur.provider !== attemptSnap.provider ||
          cur.operation !== attemptSnap.operation ||
          cur.shape !== attemptSnap.shape ||
          cur.wireKey !== attemptSnap.wireKey;
        session._attempted_model = attemptSnap.model;
        session._attempted_provider = attemptSnap.provider;
        session._attempted_operation = attemptSnap.operation;
        session._attempted_shape = attemptSnap.shape;
        session._attempted_wire_key = attemptSnap.wireKey;
        // Emit inside the wrapper's captured obs scope — the generator body is
        // resumed from the CONSUMER's async context, where ALS doesn't survive.
        _reenterObsScope(obsKey, () => _emitCallFailureLog(getClient(), session));
        if (overwritten) {
          session._attempted_model = cur.model;
          session._attempted_provider = cur.provider;
          session._attempted_operation = cur.operation;
          session._attempted_shape = cur.shape;
          session._attempted_wire_key = cur.wireKey;
        }
      } else {
        _reenterObsScope(obsKey, () => _emitCallFailureLog(getClient(), session));
      }
    } catch {
      // fail-open: failure logging must never affect error propagation
    }
    throw err;
  } finally {
    // Guard release stays unconditional and exactly-once — the catch above
    // must NOT call exitLlamaIndex.
    session.exitLlamaIndex();
    if (!failed) {
      try {
        // Build a synthetic ChatResponse from accumulated state so the
        // composition parser sees a full assistant message (text + tool_call).
        const syntheticMessage: Record<string, unknown> = {
          role: "assistant",
          content: textAcc,
        };
        if (lastOptions) syntheticMessage.options = lastOptions;
        const synthetic: Record<string, unknown> = {
          raw: lastChunkWithUsage?.raw ?? lastChunk?.raw ?? {},
          message: syntheticMessage,
        };
        const latency =
          reqStartMono != null
            ? _buildStreamLatency(reqStartMono, ttftMono, performance.now())
            : null;
        _captureLlamaIndexResponseAt(synthetic, order);
        // `obsKey` (captured at wrapper entry) — this `finally` runs in the
        // consumer's async context, so the live ALS key is not this call's.
        // Passed so both the routing stamp and the keyed observation drain
        // inside _logLlamaIndex use this call's own key. Threading beats a
        // scope re-entry here: _reenterObsScope with a NULL key runs the
        // callee bare (runWithObsKey), so an in-callee live read would see
        // the consumer's ambient key and claim a sibling's records.
        _logLlamaIndex(
          thisArg,
          synthetic,
          order,
          spanName,
          startTime,
          latency,
          obsKey,
        );
      } catch {
        // fail-safe
      }
    }
  }
}

function _setLlamaIndexWrapper(
  obj: any,
  methodName: string,
  original: Function,
): void {
  obj[methodName] = _withCallObsScope(async function (
    this: any,
    ...args: any[]
  ): Promise<any> {
    // LlamaIndex runs tools off the OTel path (the app's manual loop calls
    // `tool.call(input)` directly). The chat params carry the `tools` array —
    // wrap each tool's `.call` so executions become tool rows.
    wrapLlamaIndexTools(args[0]?.tools);
    const session = getCurrentSession();
    if (session.inLlamaIndex) {
      return await original.apply(this, args);
    }
    // The wrapped args carry messages, not a request body — take model +
    // provider off the LLM instance. The model is a matching-only HINT (never
    // merged into a body, so a REROUTE never applies here; the enforce-mode
    // refusal is recorded as REROUTE_REJECTED `unappliable_call_shape`); the
    // provider mirrors exactly what `_logLlamaIndex` attributes the row to,
    // so a rule matches
    // the same way pre- and post-call. Never throws.
    let liModel: string | undefined;
    let liProvider: string | undefined;
    try {
      const rawModel = (this as any)?.model ?? (this as any)?.modelName;
      if (typeof rawModel === "string" && rawModel) liModel = rawModel;
      liProvider = _llamaIndexProvider(this);
    } catch {
      liModel = undefined;
      liProvider = undefined;
    }
    await _runAsyncCheck(null, liProvider, null, true, false, liModel);
    // Stash attempt context so a failed streamed call lands with a real
    // model/provider (the manual row is this path's ONLY emitter — the inner
    // provider wrapper short-circuits on inLlamaIndex). Synthetic [{model}]
    // args: _stashAttemptContext reads args[0].model — same precedent as the
    // LangChain streamIterator path. No wire key at this layer. liProvider
    // "openai_responses" doubles as the shape key: _emitCallFailureLog
    // resolves it via _resolveUsageShape, so a failed OpenAIResponses row
    // carries the same usage_shape as its successful siblings.
    _stashAttemptContext(session, liProvider ?? "", [{ model: liModel }]);
    // Snapshot the attempt context: the _attempted_* slots are session-global,
    // so a concurrent same-session call may overwrite them before this stream
    // fails. (Nested provider calls under LlamaIndex short-circuit on the
    // inLlamaIndex guard BEFORE their own stash, so only genuine concurrency
    // can — the snapshot handles it.)
    const _attemptSnap = {
      model: (session as any)._attempted_model,
      provider: (session as any)._attempted_provider,
      operation: (session as any)._attempted_operation,
      shape: (session as any)._attempted_shape,
      wireKey: (session as any)._attempted_wire_key,
    };
    // Obs key for the streamed failure emit: the guarded stream's pulls run in
    // the CONSUMER's async context (ALS does not survive yield-to-customer),
    // so capture this wrapper's own scope key here and re-enter it around the
    // emit. Reuse-if-exists keeps SDK-internal delegation on the caller's key.
    const _obsKey =
      _currentObsKey() ??
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      (typeof (state as any).newObsKey === "function" ? (state as any).newObsKey() : null);
    // Reserve this call's span order + name up front so prompt/response
    // composition and the logged span all share one consistent key.
    const order = session.nextSpanOrder();
    const spanName = consumePendingSpanName();
    const startTime = new Date();
    const reqStartMono = performance.now();
    _captureLlamaIndexPromptAt(args, this, order);
    // Clear any stale inner-stream usage stash from a previous call so the
    // zero-usage fallback in _logLlamaIndex can never read another call's
    // counts.
    try {
      (session as any)._pendingLlamaIndexUsage = null;
    } catch {
      // fail-safe
    }
    session.enterLlamaIndex();
    let result: any;
    try {
      // OTel-suppression window (F-23-1): this wrapper is the SOLE billable
      // emitter for the call (_logLlamaIndex / _emitCallFailureLog); when a
      // Traceloop provider instrumentor ALSO wraps the underlying SDK (e.g.
      // instrumentModules: { anthropic, llamaIndex }), its span for the same
      // physical call would double-log as a second full-cost llm row. Any
      // instrumentor span that STARTS inside this window is tagged in
      // telemetry onStart and dropped in onEnd. ALS propagates through awaits
      // initiated inside run(), so this covers non-streaming calls of all
      // three providers, eager-HTTP adapters, and any span the instrumentor
      // opens within the awaited chain. LAZY streaming adapters
      // (@llamaindex/anthropic streamChat) fire their HTTP at first PULL, in
      // the consumer's context — _guardedLlamaIndexStream's per-pull shim
      // covers those. The nested pass-through above is deliberately left
      // without a window of its own: it usually runs inside an outer call's
      // window, but not always — inLlamaIndex is held across stream
      // CONSUMPTION (released in the guarded stream's finally), where no
      // window is active, so a concurrent same-session LlamaIndex call can
      // take it bare. Acceptable because the failure direction is fail-open:
      // an untagged duplicate merely survives (the pre-fix over-count) —
      // never a lost manual row.
      result = await runWithLlamaIndexOtelSuppress(() =>
        original.apply(this, args),
      );
    } catch (e) {
      // Eager provider rejection — @llamaindex/anthropic awaits HTTP inside
      // chat(), and @llamaindex/google 0.4.x's streamChat is a plain async
      // method (awaits sendMessageStream before returning an iterator), so
      // this catch — not _guardedLlamaIndexStream's — sees their failures,
      // along with every non-streaming rejection. The inner provider wrapper
      // is a pass-through while inLlamaIndex is set, so without an emit here
      // an eager failure lands ZERO llm rows. Emit the failure row, then
      // rethrow the ORIGINAL error by identity. Every added statement is
      // fail-open (golden rule); this catch still runs inside
      // _withCallObsScope (ALS survives await), so no obs-key re-entry is
      // needed — same as the LangChain generate catch. No
      // _failedStreamCompKeys marker: no deferred telemetry span can key by
      // this wrapper's `order` (see the stream catch), so an entry would just
      // leak unconsumed.
      try {
        const elapsedMs = Math.round(Math.max(0, performance.now() - reqStartMono));
        (session as any)._call_outcome = buildCallOutcome(e, elapsedMs);
        // Conditional-restore protocol (see the stream tap): put THIS call's
        // attempt context back for the emit; hand a newer call's context back
        // afterwards if we borrowed the slots.
        const cur = {
          model: (session as any)._attempted_model,
          provider: (session as any)._attempted_provider,
          operation: (session as any)._attempted_operation,
          shape: (session as any)._attempted_shape,
          wireKey: (session as any)._attempted_wire_key,
        };
        const overwritten =
          cur.model !== _attemptSnap.model ||
          cur.provider !== _attemptSnap.provider ||
          cur.operation !== _attemptSnap.operation ||
          cur.shape !== _attemptSnap.shape ||
          cur.wireKey !== _attemptSnap.wireKey;
        (session as any)._attempted_model = _attemptSnap.model;
        (session as any)._attempted_provider = _attemptSnap.provider;
        (session as any)._attempted_operation = _attemptSnap.operation;
        (session as any)._attempted_shape = _attemptSnap.shape;
        (session as any)._attempted_wire_key = _attemptSnap.wireKey;
        _emitCallFailureLog(getClient(), session);
        if (overwritten) {
          (session as any)._attempted_model = cur.model;
          (session as any)._attempted_provider = cur.provider;
          (session as any)._attempted_operation = cur.operation;
          (session as any)._attempted_shape = cur.shape;
          (session as any)._attempted_wire_key = cur.wireKey;
        }
      } catch {
        // fail-open: failure logging must never affect error propagation
      } finally {
        // Guard release unconditional and exactly-once — a throw inside the
        // emit block above must never skip it.
        session.exitLlamaIndex();
      }
      throw e;
    }
    // Streaming → wrap to keep the guard for the duration of consumption,
    // then log when the stream ends.
    if (result != null && typeof result[Symbol.asyncIterator] === "function") {
      return _guardedLlamaIndexStream(
        result,
        session,
        this,
        order,
        spanName,
        startTime,
        reqStartMono,
        _obsKey,
        _attemptSnap,
      );
    }
    session.exitLlamaIndex();
    _captureLlamaIndexResponseAt(result, order);
    _logLlamaIndex(this, result, order, spanName, startTime);
    return result;
  });
}

/**
 * Patches the chat/complete methods on a LlamaIndex provider LLM class.
 * Each provider lives in its own subpackage (@llamaindex/openai,
 * @llamaindex/anthropic, @llamaindex/google) — call this helper with each
 * module reference (CJS via require, ESM via user-provided import).
 */
function _instrumentLlamaIndexProvider(mod: any, classNames: string[]): void {
  try {
    if (!mod) return;
    for (const className of classNames) {
      const Cls =
        mod?.[className] ??
        mod?.default?.[className] ??
        (typeof mod === "function" && (mod as any).name === className ? mod : null);
      if (typeof Cls !== "function" || !Cls.prototype) continue;
      const proto = Cls.prototype;
      for (const method of ["chat", "complete"] as const) {
        const original = proto[method];
        if (typeof original !== "function") continue;
        // Idempotency: each prototype method patched at most once.
        if ((original as any).__tp_li_wrapped) continue;
        const key = `llamaindex:${className}:${method}`;
        _originals.set(key, original);
        _setLlamaIndexWrapper(proto, method, original);
        // Restore this provider prototype directly (the class lives in a
        // subpackage, not at any root path uninstrument() could re-resolve).
        _restoreThunks.push(() => {
          proto[method] = original;
        });
        try {
          (proto[method] as any).__tp_li_wrapped = true;
        } catch {
          /* prototype method may be non-configurable — ignore */
        }
      }
    }
  } catch {
    // fail-safe: SDK not present or its shape changed — skip
  }
}

/**
 * Public API to manually apply the pre-flight check to a specific module/class/method.
 * Useful for protecting custom internal SDKs or unlisted providers.
 *
 * @example
 * ```typescript
 * import * as tp from 'token-police';
 *
 * tp.protect('my-custom-llm', ['CustomClient', 'prototype'], 'generate', true);
 * ```
 */
/**
 * Public API to manually apply the pre-flight check to a specific
 * module/class/method. Useful for protecting custom internal SDKs or unlisted
 * providers (e.g. raw fetch calls to LLM provider REST endpoints).
 *
 * `options.manual` routes the wrapper through the manual-telemetry path —
 * the same path used internally for SDKs without an OpenLLMetry instrumentor
 * (e.g. @google/genai). The wrapper extracts token usage from the response
 * itself and logs synchronously, instead of relying on an OTel span.
 *
 * `options.provider` overrides the substring-based provider detection
 * (`_detectProvider`) which yields "" for arbitrary module paths. Composition
 * parsing keys off this string — pass e.g. "openai", "openai_responses",
 * "anthropic", "google", "cohere", "together", "cerebras", "huggingface",
 * "mistral", "litellm".
 *
 * `options.streaming` is a static hint that the wrapped method returns an
 * async iterable. The wrapper also auto-detects via Symbol.asyncIterator at
 * runtime, so this is only required when the iterable is produced lazily.
 *
 * `isAsync` — pass `true` for methods that return a Promise (the common case).
 * IMPORTANT: with `isAsync: false` the wrapped method is **telemetry-only** — it
 * captures usage/composition and logs, but runs NO pre-flight `/check` and
 * therefore performs NO enforcement (blocking/rerouting). A synchronous method
 * cannot `await` the pre-flight, so it can never be blocked. (This is a
 * Node-only asymmetry: the Python SDK's sync path *does* enforce.) If you need
 * enforcement, expose an async method and register it with `isAsync: true`.
 */
export function protect(
  moduleName: string,
  objectPath: string[],
  methodName: string,
  isAsync: boolean,
  options?: {
    manual?: boolean;
    provider?: string;
    streaming?: boolean;
    /**
     * When set, the wrapper patches this object/class directly instead of
     * doing `require(moduleName)`. Needed for in-app classes that don't live
     * in a separate npm package — pass the class constructor (or any object
     * the walker can traverse via objectPath). `moduleName` is then just an
     * identifier for diagnostics.
     */
    module?: any;
  },
): void {
  _wrapMethod(
    {
      moduleName,
      objectPath,
      method: methodName,
      isAsync,
      manualTelemetry: options?.manual ? true : undefined,
      provider: options?.provider,
      streaming: options?.streaming,
    },
    options?.module,
  );
}

/**
 * Instruments the native OpenRouter SDK (@openrouter/sdk).
 *
 * The SDK is ESM-only and does NOT export its `Chat` class from the package
 * root — `OpenRouter.prototype` only exposes a lazy `get chat()`. So we cannot
 * resolve `Chat.prototype` via a static object path. Instead we construct a
 * throw-away `OpenRouter` instance (the Speakeasy constructor only stores
 * options — no network, no validation) to reach the `Chat` class, then patch
 * `Chat.prototype.send` with the manual-telemetry wrapper (there is no
 * OpenLLMetry instrumentor for this SDK).
 */
function _instrumentOpenRouter(orModule: any): void {
  try {
    const OpenRouterClass =
      orModule?.OpenRouter ?? orModule?.default ?? orModule;
    if (typeof OpenRouterClass !== "function") return;

    const probe = new OpenRouterClass({ apiKey: "tp_probe" });
    const chatInstance = probe?.chat;
    if (!chatInstance) return;
    const ChatClass = Object.getPrototypeOf(chatInstance)?.constructor;
    if (typeof ChatClass !== "function") return;

    _wrapMethod(
      {
        moduleName: "@openrouter/sdk",
        objectPath: ["prototype"],
        method: "send",
        isAsync: true,
        manualTelemetry: true,
      },
      ChatClass,
    );
  } catch {
    // fail-safe: SDK not present or its shape changed — skip instrumentation
  }
}

/**
 * Instruments the Mistral SDK (@mistralai/mistralai — `Mistral` class).
 *
 * `Mistral.chat` is an instance property holding a `Chat` instance; the `Chat`
 * class itself is NOT root-exported from the package, so we cannot reach
 * `Chat.prototype` via a static object path. Construct a throw-away `Mistral`
 * instance (the Speakeasy constructor only stores options — no network, no
 * validation) to access the `Chat` class, then patch `Chat.prototype.complete`
 * and `Chat.prototype.stream` via the manual-telemetry path (there is no
 * OpenLLMetry-JS instrumentor for `@mistralai/mistralai`).
 *
 * Mistral's stream wraps each chunk as `CompletionEvent = { data: CompletionChunk }`;
 * the `mistral` branches in _chunkHasUsage / _accumulateStreamChunk /
 * _streamAccumulatorToResponse / _extractUsage transparently unwrap `.data` so
 * the standard _wrapManualStream handles the iteration.
 */
function _instrumentMistral(mistralModule: any): void {
  try {
    const MistralClass =
      mistralModule?.Mistral ?? mistralModule?.default ?? mistralModule;
    if (typeof MistralClass !== "function") return;

    const probe = new MistralClass({ apiKey: "tp_probe" });
    const chatInstance = probe?.chat;
    if (!chatInstance) return;
    const ChatClass = Object.getPrototypeOf(chatInstance)?.constructor;
    if (typeof ChatClass !== "function" || !ChatClass.prototype) return;

    for (const method of ["complete", "stream"] as const) {
      const original = ChatClass.prototype[method];
      if (typeof original !== "function") continue;
      const key = `@mistralai/mistralai:prototype:${method}`;
      if (_originals.has(key)) continue;

      _mistralRestore.push({
        proto: ChatClass.prototype,
        method,
        original,
      });

      _wrapMethod(
        {
          moduleName: "@mistralai/mistralai",
          objectPath: ["prototype"],
          method,
          isAsync: true,
          manualTelemetry: true,
          streaming: method === "stream",
        },
        ChatClass,
      );
    }

    // Embeddings — Mistral.embeddings is a separate instance property holding
    // an `Embeddings` client. Probe again to reach its prototype.
    try {
      const embeddingsInstance = (probe as any)?.embeddings;
      if (embeddingsInstance) {
        const EmbeddingsClass = Object.getPrototypeOf(embeddingsInstance)?.constructor;
        if (typeof EmbeddingsClass === "function" && EmbeddingsClass.prototype) {
          const embedOriginal = EmbeddingsClass.prototype.create;
          if (typeof embedOriginal === "function") {
            const key = `@mistralai/mistralai:embeddings:create`;
            if (!_originals.has(key)) {
              _mistralRestore.push({
                proto: EmbeddingsClass.prototype,
                method: "create",
                original: embedOriginal,
              });
              _wrapMethod(
                {
                  moduleName: "@mistralai/mistralai",
                  objectPath: ["prototype"],
                  method: "create",
                  isAsync: true,
                  manualTelemetry: true,
                  operation: "embedding",
                  shape: "mistral_embed",
                },
                EmbeddingsClass,
              );
            }
          }
        }
      }
    } catch {
      // fail-safe: embeddings surface may not exist on older Mistral SDKs
    }

    // OCR — Mistral.ocr is a separate instance property holding an `Ocr`
    // client (method `process`, plus `processAsync` on newer SDKs). Reached
    // via the same throw-away probe. Wrapped through the modality path so the
    // pre-flight /check carries an `{kind:"ocr"}` intent and the /log row uses
    // the `mistral_ocr` shape. The objectPath is wrapped so _wrapMethod's dedup
    // key stays distinct from Chat's `prototype:*` key.
    try {
      const ocrInstance = (probe as any)?.ocr;
      if (ocrInstance) {
        const OcrClass = Object.getPrototypeOf(ocrInstance)?.constructor;
        if (typeof OcrClass === "function" && OcrClass.prototype) {
          for (const method of ["process", "processAsync"] as const) {
            const original = OcrClass.prototype[method];
            if (typeof original !== "function") continue;
            const key = `@mistralai/mistralai:Ocr:${method}`;
            if (_originals.has(key)) continue;
            _mistralRestore.push({ proto: OcrClass.prototype, method, original });
            _wrapMethod(
              {
                moduleName: "@mistralai/mistralai",
                objectPath: ["Ocr", "prototype"],
                method,
                isAsync: true,
                manualTelemetry: true,
                modality: "ocr",
                shape: "mistral_ocr",
              },
              { Ocr: OcrClass },
            );
          }
        }
      }
    } catch {
      // fail-safe: OCR surface may not exist on older Mistral SDKs
    }

    // Audio transcriptions — Mistral.audio.transcriptions is an instance
    // property holding a `Transcriptions` client (method `complete`, plus
    // `completeAsync` on newer SDKs). `stream` is out of scope (Chat already
    // owns the `stream` shape). Wrapped through the modality path → `audio_stt`
    // intent + `mistral_audio_stt` shape. The nested objectPath keeps the
    // dedup key distinct from Chat's `prototype:complete`.
    try {
      const transcriptionsInstance = (probe as any)?.audio?.transcriptions;
      if (transcriptionsInstance) {
        const TranscriptionsClass =
          Object.getPrototypeOf(transcriptionsInstance)?.constructor;
        if (typeof TranscriptionsClass === "function" && TranscriptionsClass.prototype) {
          for (const method of ["complete", "completeAsync"] as const) {
            const original = TranscriptionsClass.prototype[method];
            if (typeof original !== "function") continue;
            const key = `@mistralai/mistralai:Transcriptions:${method}`;
            if (_originals.has(key)) continue;
            _mistralRestore.push({
              proto: TranscriptionsClass.prototype,
              method,
              original,
            });
            _wrapMethod(
              {
                moduleName: "@mistralai/mistralai",
                objectPath: ["Transcriptions", "prototype"],
                method,
                isAsync: true,
                manualTelemetry: true,
                modality: "audio_stt",
                shape: "mistral_audio_stt",
              },
              { Transcriptions: TranscriptionsClass },
            );
          }
        }
      }
    } catch {
      // fail-safe: transcriptions surface may not exist on older Mistral SDKs
    }
  } catch {
    // fail-safe: SDK not present or its shape changed — skip instrumentation
  }
}

/**
 * Instruments the Cohere v2 SDK (cohere-ai — CohereClientV2).
 *
 * In cohere-ai v8 `CohereClientV2.chat` / `.chatStream` are instance-bound
 * arrow fields set in the constructor — not prototype methods — so they
 * cannot be patched via a static object path. They delegate to an internal
 * `V2Client` (not exported from the package root) whose `chat` / `chatStream`
 * ARE real prototype methods. We construct a throw-away `CohereClientV2` (the
 * Fern constructor only stores options — no network) to reach the `V2Client`
 * class, then patch `V2Client.prototype.chat` / `.chatStream` with the
 * manual-telemetry wrapper. @traceloop/instrumentation-cohere only supports
 * the legacy v1 CohereClient, so v2 token usage is read from response.usage.
 *
 * This must run before the app constructs its CohereClientV2 — the instance
 * fields bind from V2Client.prototype at construction time.
 */
function _instrumentCohere(cohereModule: any): void {
  try {
    const CohereClientV2 =
      cohereModule?.CohereClientV2 ?? cohereModule?.default?.CohereClientV2;
    if (typeof CohereClientV2 !== "function") return;

    const probe = new CohereClientV2({ token: "tp_probe" });
    const v2client = (probe as any)?.clientV2;
    if (!v2client) return;
    const V2ClientClass = Object.getPrototypeOf(v2client)?.constructor;
    if (typeof V2ClientClass !== "function") return;

    for (const method of ["chat", "chatStream"] as const) {
      _wrapMethod(
        {
          moduleName: "cohere-ai",
          objectPath: ["prototype"],
          method,
          isAsync: true,
          manualTelemetry: true,
          streaming: method === "chatStream",
        },
        V2ClientClass,
      );
    }

    // Embedding methods on V2Client.prototype — instance-bound arrow fields
    // on CohereClientV2 delegate here, same as chat. Cover both the sync and
    // async surface so customer code using `client.embed(...)` /
    // `client.embedAsync(...)` is wrapped.
    for (const method of ["embed", "embedAsync"] as const) {
      const original = V2ClientClass.prototype?.[method];
      if (typeof original !== "function") continue;
      _wrapMethod(
        {
          moduleName: "cohere-ai",
          objectPath: ["prototype"],
          method,
          isAsync: true,
          manualTelemetry: true,
          operation: "embedding",
          shape: "cohere_embed",
        },
        V2ClientClass,
      );
    }
  } catch {
    // fail-safe: SDK not present or its shape changed — skip instrumentation
  }
}

/**
 * Instruments the HuggingFace SDK (@huggingface/inference — InferenceClient).
 *
 * `InferenceClient.chatCompletion` / `.chatCompletionStream` are NOT prototype
 * methods — the constructor copies each task function into a per-instance,
 * non-configurable, non-writable field. The instance fields therefore cannot
 * be patched, and there is no delegated class to reach (unlike Cohere's
 * V2Client). The constructor reads each task function BY VALUE from the `tasks`
 * index module at construction time; that index re-exports via getters pointing
 * at the source submodules `tasks/nlp/chatCompletion.js` /
 * `chatCompletionStream.js`, whose exports ARE plain writable value properties.
 * So we patch those source-submodule exports via Node's require cache: every
 * `InferenceClient` constructed afterwards picks up the wrapper. This must run
 * before the app constructs its client.
 *
 * There is no OpenLLMetry instrumentor for @huggingface/inference, so token
 * usage is extracted via the manual-telemetry path (response.usage) — the SDK
 * is OpenAI-compatible (snake_case usage), exactly like Cerebras.
 */
function _instrumentHuggingFace(hfModule: any): void {
  try {
    // Ensure the CJS module graph (and its task submodules) are in the CJS
    // module cache. App-first so the graph warmed is the APP's copy, not the
    // SDK's own devDependency; createRequire instances share Node's single
    // global cache, so the lookup below reads exactly what this warmed.
    if (!resolveProviderModule("@huggingface/inference") && !hfModule) {
      // Unresolvable from every anchor (pure-ESM context) — we cannot reach
      // the submodule exports, and the module reference itself is unusable
      // for the require-cache strategy, so bail out.
      return;
    }

    const cache = _requireCache();
    const findExports = (suffix: string): any => {
      const key = Object.keys(cache).find(
        (k) =>
          k.includes("@huggingface/inference") &&
          k.replace(/\\/g, "/").endsWith(suffix),
      );
      return key ? cache[key]?.exports : null;
    };

    const targets: Array<{
      suffix: string;
      method: string;
      streaming: boolean;
      operation?: string;
      shape?: string;
      modality?: ModalityKey;
    }> = [
      {
        suffix: "/tasks/nlp/chatCompletion.js",
        method: "chatCompletion",
        streaming: false,
      },
      {
        suffix: "/tasks/nlp/chatCompletionStream.js",
        method: "chatCompletionStream",
        streaming: true,
      },
      // featureExtraction is the embeddings surface. Same require-cache
      // trick: the submodule export is a plain writable function property.
      {
        suffix: "/tasks/nlp/featureExtraction.js",
        method: "featureExtraction",
        streaming: false,
        operation: "embedding",
        shape: "huggingface_embed",
      },
      // Non-text modalities. Same require-cache trick against the cv/audio task
      // submodules. These route through the modality path (intent → /check,
      // usage logged via _logModality with the explicit usage shape) rather
      // than the plain _logManual chat/embedding path.
      {
        suffix: "/tasks/cv/textToImage.js",
        method: "textToImage",
        streaming: false,
        modality: "image_gen",
        shape: "huggingface_image",
      },
      {
        suffix: "/tasks/audio/textToSpeech.js",
        method: "textToSpeech",
        streaming: false,
        modality: "audio_tts",
        shape: "huggingface_audio_tts",
      },
      {
        suffix: "/tasks/audio/automaticSpeechRecognition.js",
        method: "automaticSpeechRecognition",
        streaming: false,
        modality: "audio_stt",
        shape: "huggingface_audio_stt",
      },
    ];

    const provider = "huggingface";

    for (const t of targets) {
      const submodExports = findExports(t.suffix);
      if (!submodExports) continue;
      const original = submodExports[t.method];
      if (typeof original !== "function") continue;

      const key = `@huggingface/inference:${t.suffix}:${t.method}`;
      if (_originals.has(key)) continue; // already instrumented
      _originals.set(key, original);
      _hfRestore.push({ exportsObj: submodExports, method: t.method, original });

      // chatCompletion is `async (...)`; chatCompletionStream is
      // `async function* (...)`. A single async wrapper covers both: for the
      // streaming case `original.apply` returns an async generator, which the
      // `isStreaming` check below detects and hands to _wrapManualStream.
      submodExports[t.method] = _withCallObsScope(async function (
        this: any,
        ...args: any[]
      ): Promise<any> {
        // Inside a LangChain-instrumented call → pass through.
        if (getCurrentSession().inLangchain) {
          return await original.apply(this, args);
        }
        // 1. Pre-flight check (may throw TokenPoliceBlockedError — intended).
        // Every target forwards the request body + provider; args[0] is the
        // merged request object and carries a literal `model` key, so
        // a same-provider REROUTE genuinely swaps the model the wire call
        // reads — the same mechanism as the openai/anthropic paths, and
        // correct here because the original call re-reads this object.
        // Modality targets add a modality intent; featureExtraction (the
        // embeddings surface) adds `{ kind: "embedding" }` to mirror the
        // log side's operation="embedding" so a modality-scoped rule matches
        // pre- and post-call alike.
        {
          const reqBody =
            args[0] && typeof args[0] === "object"
              ? (args[0] as Record<string, any>)
              : null;
          const intent = t.modality
            ? _buildIntent(provider, t.modality, args)
            : t.operation === "embedding"
              ? { kind: "embedding" }
              : undefined;
          await _runAsyncCheck(reqBody, provider, intent);
        }

        // 2. Reserve this call's span order + name up front.
        let order = 0;
        let spanName: string | null = null;
        try {
          order = getCurrentSession().nextSpanOrder();
          spanName = consumePendingSpanName();
        } catch {
          // fail-safe
        }
        const startTime = new Date();

        // 3. Capture prompt composition. args[0] is the merged request object
        // ({ endpointUrl, accessToken, model, messages, tools, ... }).
        _captureCompositionAt(provider, args, undefined, order, undefined, t.operation);

        // Stash attempt context so failures land with real model/provider
        // (streaming failures surface inside _wrapManualStream's iteration,
        // which reads this stash via _emitCallFailureLog). modality
        // fallback so failed image/tts rows do not mis-default to "chat".
        // `t.shape` is the same override the success path forwards, so a
        // failed row carries its successful siblings' usage_shape.
        _stashAttemptContext(
          getCurrentSession(),
          provider,
          args,
          _resolveFailOperation(t.operation, t.modality),
          t.shape,
        );

        // 4. Call original. A thrown provider error (e.g.
        // InferenceClientProviderApiError on a 4xx/5xx) must still emit a
        // failed llm row — without this catch the failure is
        // never recorded at all (no row). Mirrors the standard wrapper.
        const _callStart = Date.now();
        // Monotonic anchor for stream latency (is_streaming/ttft).
        const _callStartMono = performance.now();
        let result: any;
        try {
          result = await original.apply(this, args);
        } catch (err) {
          const elapsedMs = Date.now() - _callStart;
          const session = getCurrentSession();
          try {
            (session as any)._call_outcome = buildCallOutcome(err, elapsedMs);
          } catch {
            // fail-safe
          }
          _emitCallFailureLog(getClient(), session);
          throw err;
        }

        // 5a. Streaming → wrap the async iterable, log on completion.
        const isStreaming =
          t.streaming === true ||
          (result != null &&
            typeof result[Symbol.asyncIterator] === "function");
        if (isStreaming) {
          return _wrapManualStream(
            result,
            provider,
            args,
            order,
            spanName,
            startTime,
            _callStartMono,
          );
        }

        // 5b. Non-streaming → capture response composition + log immediately.
        // Modality (image/audio) targets log via _logModality with the
        // explicit usage shape + extracted items/duration; embedding
        // and chat targets keep the _logManual path. For both modality and
        // embedding responses the shape is passed to composition so binary
        // bodies / vectors are never read as text.
        if (t.modality && t.shape) {
          _captureCompositionAt(provider, args, result, order, t.shape);
          const elapsedSeconds = Math.max(0, (Date.now() - _callStart) / 1000);
          _logModality(
            provider,
            t.modality,
            t.shape,
            args,
            result,
            order,
            spanName,
            startTime,
            elapsedSeconds,
          );
          return result;
        }
        const respShape = t.operation === "embedding" ? t.shape : undefined;
        _captureCompositionAt(provider, args, result, order, respShape, t.operation);
        _logManual(provider, args, result, order, spanName, startTime,
                   t.operation || "chat", t.shape ?? null);
        return result;
      });
    }
  } catch {
    // fail-safe: SDK not present or its shape changed — skip instrumentation
  }
}

/**
 * Patches `BaseChatModel.prototype.{generate,stream}` on the given module.
 *
 * @langchain/core ships dual builds (CJS + ESM) and consumers like
 * @langchain/openai (ESM) import the ESM build, while a generic require()
 * from this SDK loads the CJS build — they are DIFFERENT objects. To cover
 * both, call this helper with each module reference (CJS via require, ESM
 * via a user-provided import, ESM via async dynamic import as a fallback).
 * The wrapper function itself stamps the prototype method with a marker so
 * the second-pass patch is a no-op.
 */
// ═══════════════════════════════════════════════════════════════════
// Vercel AI SDK (`ai` package) — generic LanguageModel V2/V3 wrappers.
//
// Every AI SDK model — first-party (@ai-sdk/openai, @ai-sdk/anthropic, ...),
// the AI Gateway (plain "vendor/model" strings), and custom / community /
// OpenAI-compatible providers — implements the same LanguageModel spec:
// instances expose `doGenerate(options)` / `doStream(options)` plus
// `.provider`, `.modelId` and `.specificationVersion`. The concrete classes
// are not root-exported, so we construct throw-away probe models (purely
// local — constructing a model performs no network I/O) and patch the
// resolved prototypes with manual-telemetry wrappers — there is no
// OpenLLMetry-JS instrumentor for the AI SDK.
//
// Internally these calls parse under the "ai_sdk" provider key (shared with
// the legacy "xai" key — see _isAiSdkParse); the REPORTED provider is derived
// per call from `model.provider` (e.g. "minimax.messages" → "minimax",
// "gateway" → "vercel-gateway").
// ═══════════════════════════════════════════════════════════════════

/**
 * True when `provider` is an internal parsing key for Vercel AI SDK
 * LanguageModel V2/V3 call shapes. "xai" predates the generic path (it was
 * the first AI SDK provider instrumented) and shares all of its parsers.
 */
function _isAiSdkParse(provider: string): boolean {
  return provider === "ai_sdk" || provider === "xai";
}

/**
 * Async-scoped guard held by `tokenPoliceAiSdkMiddleware` around its inner
 * call. A model can be BOTH wrapped with the middleware AND prototype-patched
 * by the enforcer; the prototype wrapper checks this scope and passes straight
 * through (one check, one log). AsyncLocalStorage (not a session flag) because
 * outside a tp.session()/workflow() scope getCurrentSession() returns a fresh
 * instance per call, so an instance flag can't bridge the two wrappers.
 */
const _aiSdkMiddlewareScope = new AsyncLocalStorage<boolean>();

function _inAiSdkMiddlewareScope(): boolean {
  try {
    return _aiSdkMiddlewareScope.getStore() === true;
  } catch {
    return false;
  }
}

interface AiSdkProviderSpec {
  /** require()/import() target. */
  pkg: string;
  /** Exported provider factory / singleton names, tried in order. */
  factoryNames: string[];
  /** Options passed to `create*` factories. Defaults to {apiKey:"tp_probe"}. */
  factoryOptions?: Record<string, any>;
  /** Local-only model id used to construct the probe (no network). */
  probeModelId: string;
}

/**
 * Known AI SDK provider packages, auto-discovered at init. Each entry needs a
 * probe recipe because provider factories differ in exported names and
 * required construction options. Community providers that bundle their own
 * nested @ai-sdk/* copies (which require() from here cannot reach) are
 * covered by `instrumentModules.aiSdkProviders` / the exported middleware.
 */
const _AI_SDK_REGISTRY: AiSdkProviderSpec[] = [
  { pkg: "@ai-sdk/xai", factoryNames: ["createXai", "xai"], probeModelId: "grok-4" },
  { pkg: "@ai-sdk/openai", factoryNames: ["createOpenAI", "openai"], probeModelId: "gpt-4o" },
  { pkg: "@ai-sdk/anthropic", factoryNames: ["createAnthropic", "anthropic"], probeModelId: "claude-sonnet-4-5" },
  { pkg: "@ai-sdk/google", factoryNames: ["createGoogleGenerativeAI", "google"], probeModelId: "gemini-2.5-flash" },
  { pkg: "@ai-sdk/mistral", factoryNames: ["createMistral", "mistral"], probeModelId: "mistral-large-latest" },
  { pkg: "@ai-sdk/groq", factoryNames: ["createGroq", "groq"], probeModelId: "llama-3.3-70b-versatile" },
  // AI Gateway — plain "vendor/model" model strings in generateText/streamText
  // resolve through this provider. The `ai` package re-exports it, so probing
  // `ai` directly also covers apps that never install @ai-sdk/gateway at the
  // top level (it may be nested under ai/node_modules out of require()'s
  // reach from here).
  { pkg: "@ai-sdk/gateway", factoryNames: ["createGatewayProvider", "createGateway", "gateway"], probeModelId: "openai/gpt-4o" },
  { pkg: "ai", factoryNames: ["createGateway", "gateway"], probeModelId: "openai/gpt-4o" },
  {
    pkg: "@ai-sdk/openai-compatible",
    factoryNames: ["createOpenAICompatible"],
    factoryOptions: { name: "tp-probe", apiKey: "tp_probe", baseURL: "https://tp-probe.invalid" },
    probeModelId: "tp-probe",
  },
];

/**
 * `model.provider` head → TokenPolice provider slug. Heads not listed pass
 * through verbatim (e.g. "minimax.messages" → "minimax", "openai.chat" →
 * "openai") — AI SDK provider heads already match the server's
 * provider-identity vocabulary.
 */
const _AI_SDK_PROVIDER_MAP: Record<string, string> = {
  gateway: "vercel-gateway",
  "amazon-bedrock": "bedrock",
  vertex: "vertex-ai",
  "google-vertex": "vertex-ai",
};

function _mapAiSdkProvider(rawProvider: string): string {
  try {
    const head = String(rawProvider || "").split(".")[0].trim().toLowerCase();
    if (!head) return "ai_sdk";
    return _AI_SDK_PROVIDER_MAP[head] ?? head;
  } catch {
    return "ai_sdk";
  }
}

interface AiSdkCallMeta {
  modelId: string;
  /** Mapped TokenPolice provider slug — what gets reported to the server. */
  provider: string;
  rawProvider: string;
  specVersion: string;
  /** Sanitized serving endpoint (host metadata only) or "". */
  baseURL: string;
}

/**
 * Extracts call metadata from a LanguageModel instance. The model id and
 * provider are bound at provider-construction time, not present on the spec
 * call options. `config.baseURL` is provider-internal (not spec) — read
 * opportunistically and sanitized to host metadata. Fail-open throughout.
 */
function _aiSdkCallMeta(modelLike: any): AiSdkCallMeta {
  let modelId = "unknown";
  let rawProvider = "";
  let specVersion = "";
  let baseURL = "";
  try {
    modelId = String(modelLike?.modelId ?? modelLike?.model ?? "unknown");
    rawProvider = String(modelLike?.provider ?? "");
    specVersion = String(modelLike?.specificationVersion ?? "");
    baseURL = _sanitizeBaseURLString(modelLike?.config?.baseURL);
  } catch {
    // fail-safe — meta is advisory
  }
  return {
    modelId,
    provider: _mapAiSdkProvider(rawProvider),
    rawProvider,
    specVersion,
    baseURL,
  };
}

/**
 * Builds the kwargs copy handed to the usage/composition extractors and
 * _logManual. Mirrors the original `__tpXaiModel` stash precedent; adds the
 * full meta under `__tpAiSdk` (reported provider) and surfaces `baseURL` so
 * _logManual's existing _extractBaseURL pickup emits model_extras.api_base.
 */
function _aiSdkKwargs(args0: any, meta: AiSdkCallMeta): any {
  try {
    return {
      ...(args0 && typeof args0 === "object" ? args0 : {}),
      __tpXaiModel: meta.modelId,
      __tpAiSdk: meta,
      ...(meta.baseURL ? { baseURL: meta.baseURL } : {}),
    };
  } catch {
    return { __tpXaiModel: meta?.modelId ?? "unknown", __tpAiSdk: meta };
  }
}

/**
 * Shared doGenerate execution path — used by both the prototype patch and the
 * exported `tokenPoliceAiSdkMiddleware`. `callOriginal` performs the
 * underlying call; `modelLike` is the LanguageModel instance (or the
 * middleware's `model` param); `args0` is the spec call-options object.
 *
 * Fail-open: every TokenPolice step here is fail-open; only
 * TokenPoliceBlockedError (thrown by _runAsyncCheck under enforce=true) may
 * propagate. Provider errors from callOriginal belong to the customer and are
 * re-thrown after capturing a call-failure row.
 */
const _aiSdkRunGenerate = _withCallObsScope(async function (
  callOriginal: () => Promise<any>,
  modelLike: any,
  args0: any,
): Promise<any> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const session: any = getCurrentSession();
  if (session.inLangchain || session.inLlamaIndex || _inAiSdkMiddlewareScope()) {
    return await callOriginal();
  }
  const meta = _aiSdkCallMeta(modelLike);
  // The success row logs through _logManual("ai_sdk", …) → `vercel_ai`
  // whatever the concrete provider slug stashed here, so pin that shape on the
  // failure row too instead of letting it resolve off `meta.provider`.
  _stashAttemptContext(
    session, meta.provider, [{ model: meta.modelId }], "chat", "vercel_ai",
  );
  // Body-less framework convention (as LangChain / LlamaIndex / Bedrock
  // Converse): the model is bound to the LanguageModel instance, not to a
  // per-call body, so a REROUTE swap could never reach the wire from here.
  // Pass a null body plus `meta.modelId` as the matching-only modelHint —
  // rules still match for enforcement/audit, and a live ENFORCE directive
  // resolves as REROUTE_REJECTED (`unappliable_call_shape`) instead of
  // silently doing nothing.
  await _runAsyncCheck(null, meta.provider, undefined, true, false, meta.modelId);
  let order = 0;
  let spanName: string | null = null;
  try {
    order = session.nextSpanOrder();
    spanName = consumePendingSpanName();
  } catch {
    // fail-safe
  }
  const startTime = new Date();
  const callStartMono = performance.now();
  const kwargsArgs = [_aiSdkKwargs(args0, meta)];
  _captureCompositionAt("ai_sdk", kwargsArgs, undefined, order);
  let result: any;
  try {
    result = await callOriginal();
  } catch (err) {
    const elapsedMs = Math.round(Math.max(0, performance.now() - callStartMono));
    try {
      session._call_outcome = buildCallOutcome(err, elapsedMs);
    } catch {
      // fail-safe
    }
    _emitCallFailureLog(getClient(), session);
    throw err;
  }
  _captureCompositionAt("ai_sdk", kwargsArgs, result, order);
  _logManual("ai_sdk", kwargsArgs, result, order, spanName, startTime);
  return result;
});

/**
 * Shared doStream execution path. `doStream` returns
 * `{ stream: ReadableStream<StreamPart>, ... }`; the standard
 * _wrapManualStream would inspect the outer envelope (which has no
 * Symbol.asyncIterator) and short-circuit, so we wrap the inner `stream`
 * field directly and return a new envelope with the wrapped stream.
 */
const _aiSdkRunStream = _withCallObsScope(async function (
  callOriginal: () => Promise<any>,
  modelLike: any,
  args0: any,
): Promise<any> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const session: any = getCurrentSession();
  if (session.inLangchain || session.inLlamaIndex || _inAiSdkMiddlewareScope()) {
    return await callOriginal();
  }
  const meta = _aiSdkCallMeta(modelLike);
  // Pin `vercel_ai` — the shape _logManual("ai_sdk", …) logs on success.
  _stashAttemptContext(
    session, meta.provider, [{ model: meta.modelId }], "chat", "vercel_ai",
  );
  // Null body + modelHint — see _aiSdkRunGenerate.
  await _runAsyncCheck(null, meta.provider, undefined, true, false, meta.modelId);
  let order = 0;
  let spanName: string | null = null;
  try {
    order = session.nextSpanOrder();
    spanName = consumePendingSpanName();
  } catch {
    // fail-safe
  }
  const startTime = new Date();
  const reqStartMono = performance.now();
  const kwargsArgs = [_aiSdkKwargs(args0, meta)];
  _captureCompositionAt("ai_sdk", kwargsArgs, undefined, order);
  let result: any;
  try {
    result = await callOriginal();
  } catch (err) {
    const elapsedMs = Math.round(Math.max(0, performance.now() - reqStartMono));
    try {
      session._call_outcome = buildCallOutcome(err, elapsedMs);
    } catch {
      // fail-safe
    }
    _emitCallFailureLog(getClient(), session);
    throw err;
  }
  if (
    !result ||
    typeof result !== "object" ||
    !result.stream ||
    typeof result.stream[Symbol.asyncIterator] !== "function"
  ) {
    return result;
  }
  const innerStream = result.stream;
  // The AI SDK expects `stream` to be a Web ReadableStream — its
  // `runToolsTransformation` constructs a new ReadableStream and calls
  // `.pipeThrough(transformer)` on the source, which only works on proper
  // ReadableStreams. Build one that pulls from the original and runs the
  // accumulate-and-log logic on completion.
  //
  // Fail-open: the `new ReadableStream(...)` construction and the envelope
  // return are wrapped so that if the Web ReadableStream API is unavailable
  // (non-Node runtime, polyfill gap, prototype pollution) we hand back the
  // *raw* result untouched — telemetry is lost on that call but the
  // customer's stream still works. Without this guard a construction failure
  // throws into customer code.
  try {
    // Read the provider source one chunk per pull() so the Web Streams
    // backpressure protocol governs upstream consumption: the runtime only
    // calls pull() while controller.desiredSize > 0, so a slow/paused or
    // early-abandoning consumer never forces the whole response to be buffered.
    // Per-stream finalize state lives here (outer closure) so it survives across
    // pull() calls — it must fire the usage/composition log exactly once across
    // all three terminal paths (close, error, cancel), never per-pull.
    const iterator: AsyncIterator<any> = innerStream[Symbol.asyncIterator]();
    // Captured obs key for the drain-time callbacks below: pull()/finalize
    // run in the CONSUMER's async context (outside this call's obs scope), so
    // the key is captured here (still in scope) and re-entered around the
    // log calls via runWithObsKey.
    const _obsKey = _currentObsKey();
    let lastUsageChunk: any;
    let ttftMono: number | null = null;
    const acc = _newStreamAccumulator("ai_sdk");
    let settled = false; // once-guard: finalize/failure log fires at most once
    let failed = false; // mirrors the reference generator's `streamFailed` flag
    // — suppresses the usage log when the stream ended on a provider error (a
    // call-failure row is emitted instead).
    // T6: true once a `tool-call` part has been stashed incrementally in pull()
    // below. When set, finalizeUsage() must NOT re-stash from the full
    // accumulator (that would resurrect ids already consumed by the tools the
    // AI SDK executed mid-stream).
    let incrementalToolStash = false;
    const finalizeUsage = (): void => {
      // Emit the usage/composition log exactly once on normal completion.
      // Suppressed on the failure path (the failure row already covers it).
      if (settled) return;
      settled = true;
      if (failed) return;
      try {
        if (lastUsageChunk) {
          const latency = _buildStreamLatency(reqStartMono, ttftMono, performance.now());
          const responseForComp =
            _streamAccumulatorToResponse("ai_sdk", acc) ?? lastUsageChunk;
          _captureCompositionAt(
            "ai_sdk",
            kwargsArgs,
            responseForComp,
            order,
            undefined,
            undefined,
            incrementalToolStash, // T6: don't resurrect mid-stream-consumed ids
          );
          // Re-enter the captured obs scope so _logManual's keyed drain
          // claims this call's own observations (finalize runs in the
          // consumer's context, outside the original scope).
          _reenterObsScope(_obsKey, () =>
            _logManual(
              "ai_sdk",
              kwargsArgs,
              lastUsageChunk,
              order,
              spanName,
              startTime,
              "chat",
              null,
              latency,
            ),
          );
        }
      } catch {
        /* fail-open */
      }
    };
    const wrappedStream = new ReadableStream({
      // Relies on the default queuing strategy (highWaterMark = 1). Do not
      // add a { highWaterMark: N } / large strategy here — that would let the
      // runtime call pull() ahead of consumer demand and reintroduce the
      // unbounded-buffering defect this wrapper exists to prevent.
      async pull(controller) {
        let step: IteratorResult<any>;
        try {
          step = await iterator.next();
        } catch (err) {
          // Mid-stream provider error: surface it to the consumer via
          // controller.error (its read() rejects) — the ReadableStream surface
          // difference from the reference generator, which does `throw err`.
          // Emit exactly one call-failure row (buildCallOutcome +
          // _emitCallFailureLog, mirroring the reference generator's catch) and
          // set `failed` to suppress the usage log even if a usage chunk was
          // already seen. No error escapes into customer code.
          failed = true;
          const elapsedMs = Math.round(Math.max(0, performance.now() - reqStartMono));
          // Reuse the outer `session` (already carries the stashed attempt
          // context) exactly like the callOriginal catch above — a fresh
          // getCurrentSession() at pull() time may be outside the original ALS
          // scope and miss the attempted model/provider.
          try {
            try {
              session._call_outcome = buildCallOutcome(err, elapsedMs);
            } catch {
              /* fail-safe */
            }
            // Re-enter the captured obs scope (pull() runs in the consumer's
            // context) so the failure drain claims this call's observations.
            _reenterObsScope(_obsKey, () =>
              _emitCallFailureLog(getClient(), session),
            );
          } catch {
            /* fail-open */
          }
          settled = true; // failure row emitted; the usage log is suppressed
          try {
            controller.error(err);
          } catch {
            /* already closed */
          }
          return;
        }
        if (step.done) {
          finalizeUsage();
          try {
            controller.close();
          } catch {
            /* already closed */
          }
          return;
        }
        const chunk = step.value;
        // Tap is best-effort: a malformed chunk must cost telemetry, never
        // abort the customer's iteration.
        try {
          if (_chunkHasUsage("ai_sdk", chunk)) lastUsageChunk = chunk;
          _accumulateStreamChunk("ai_sdk", acc, chunk);
          // Mark TTFT at the first chunk carrying renderable content.
          if (ttftMono === null && _streamAccHasContent("ai_sdk", acc)) {
            ttftMono = performance.now();
          }
        } catch {
          // fail-open
        }
        // T6: The Vercel AI SDK invokes a tool's execute() the moment it reads a
        // consolidated `tool-call` part — which happens as we enqueue this chunk,
        // BEFORE the stream's close where finalizeUsage() would otherwise stash
        // the ids. Stash this call's (id, name) NOW, so popPendingToolCallId()
        // during that execute() resolves. APPEND (not replace) so two calls with
        // the same tool name in one step keep FIFO pop order. Separate best-effort
        // try — a throw here costs the correlation, never the customer's stream.
        try {
          if (chunk?.type === "tool-call") {
            const cid = chunk.toolCallId ?? chunk.id ?? "";
            if (cid) {
              session.appendPendingToolCall(String(cid), String(chunk.toolName ?? ""));
              incrementalToolStash = true;
            }
          }
        } catch {
          // fail-open
        }
        controller.enqueue(chunk);
      },
      async cancel() {
        // Customer abandoned the stream early. Release the provider source so
        // it can free its connection, then finalize once. Every step is
        // swallowed: a source whose return() throws must not reject the
        // customer's reader.cancel() promise (fail-open).
        try {
          await iterator.return?.();
        } catch {
          /* fail-open: releasing the source must never throw out */
        }
        finalizeUsage();
      },
    });
    return { ...result, stream: wrappedStream };
  } catch {
    /* fail-open: ReadableStream construction failed → raw stream */
    return result;
  }
});

/**
 * AI SDK model-class kinds. Beyond LanguageModel, the spec defines separate
 * model interfaces per modality, each reached via its own provider accessor:
 * `embeddingModel()` (doEmbed), `imageModel()` / `speechModel()` /
 * `transcriptionModel()` / `videoModel()` (each with its own doGenerate).
 * "auto" = a doGenerate-only instance whose modality couldn't be determined
 * at patch time (speech vs transcription share an identical surface) — the
 * wrapper resolves it per call from the call options.
 */
type AiSdkModelKind =
  | "language"
  | "embedding"
  | "image"
  | "speech"
  | "transcription"
  | "video"
  | "auto";

const _AI_SDK_KIND_TO_MODALITY: Partial<Record<AiSdkModelKind, ModalityKey>> = {
  image: "image_gen",
  speech: "audio_tts",
  transcription: "audio_stt",
  video: "video_gen",
};

const _AI_SDK_MODALITY_SHAPES: Partial<Record<ModalityKey, string>> = {
  image_gen: "vercel_ai_image",
  audio_tts: "vercel_ai_speech",
  audio_stt: "vercel_ai_transcribe",
  video_gen: "vercel_ai_video",
};

/**
 * Classifies a model instance by its spec surface. LanguageModels are the
 * only kind with doStream; ImageModel/VideoModel carry maxImagesPerCall /
 * maxVideosPerCall (class fields — present even when assigned undefined).
 * Returns null for things that aren't AI SDK models at all.
 */
function _detectAiSdkModelKind(m: any): AiSdkModelKind | null {
  try {
    if (!m || typeof m !== "object") return null;
    if (typeof m.doEmbed === "function") return "embedding";
    if (typeof m.doGenerate !== "function") return null;
    if (typeof m.doStream === "function") return "language";
    if ("maxImagesPerCall" in m) return "image";
    if ("maxVideosPerCall" in m) return "video";
    return "auto";
  } catch {
    return null;
  }
}

/**
 * Resolves the modality of an "auto" doGenerate call from its options:
 * transcription options carry `audio`, speech options carry `text`, image /
 * video options carry `prompt` (image is by far the more common, and a video
 * model would normally have been classified via maxVideosPerCall already).
 */
function _aiSdkAutoModality(options: any): ModalityKey {
  try {
    if (options && typeof options === "object") {
      if (options.audio != null) return "audio_stt";
      if (typeof options.text === "string") return "audio_tts";
    }
  } catch {
    /* fail-safe */
  }
  return "image_gen";
}

/**
 * Shared doEmbed execution path (EmbeddingModelV2/V3 — `embed`/`embedMany`).
 * doEmbed({ values }) → { embeddings, usage?: { tokens } }. Same fail-open
 * contract as _aiSdkRunGenerate.
 */
const _aiSdkRunEmbed = _withCallObsScope(async function (
  callOriginal: () => Promise<any>,
  modelLike: any,
  args0: any,
): Promise<any> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const session: any = getCurrentSession();
  if (session.inLangchain || session.inLlamaIndex || _inAiSdkMiddlewareScope()) {
    return await callOriginal();
  }
  const meta = _aiSdkCallMeta(modelLike);
  // Pin `vercel_ai_embed` — the shape the success path logs below.
  _stashAttemptContext(
    session, meta.provider, [{ model: meta.modelId }], "embedding", "vercel_ai_embed",
  );
  const kwargs = _aiSdkKwargs(args0, meta);
  try {
    // _logManual's embedding branch recovers the model from `req.model` —
    // the spec call options carry no model field, so surface it here.
    (kwargs as any).model = meta.modelId;
  } catch {
    /* fail-safe */
  }
  const kwargsArgs = [kwargs];
  let intent: Record<string, unknown> = { kind: "embedding" };
  try {
    if (Array.isArray(args0?.values)) intent = { kind: "embedding", count: args0.values.length };
  } catch {
    /* fail-safe */
  }
  // Null body + modelHint — see _aiSdkRunGenerate.
  await _runAsyncCheck(null, meta.provider, intent, true, false, meta.modelId);
  let order = 0;
  let spanName: string | null = null;
  try {
    order = session.nextSpanOrder();
    spanName = consumePendingSpanName();
  } catch {
    // fail-safe
  }
  const startTime = new Date();
  const callStartMono = performance.now();
  _captureCompositionAt("ai_sdk", kwargsArgs, undefined, order, undefined, "embedding");
  let result: any;
  try {
    result = await callOriginal();
  } catch (err) {
    const elapsedMs = Math.round(Math.max(0, performance.now() - callStartMono));
    try {
      session._call_outcome = buildCallOutcome(err, elapsedMs);
    } catch {
      // fail-safe
    }
    _emitCallFailureLog(getClient(), session);
    throw err;
  }
  // Embedding responses are vectors — pass the shape so composition returns [].
  _captureCompositionAt("ai_sdk", kwargsArgs, result, order, "vercel_ai_embed", "embedding");
  _logManual("ai_sdk", kwargsArgs, result, order, spanName, startTime, "embedding");
  return result;
});

/**
 * Shared doGenerate execution path for the non-language modality models
 * (image / speech / transcription / video). Logs through _logModality with
 * the registered `ai_sdk:{modality}` handler; pre-flight check forwards the
 * handler's intent so modality-scoped rules match.
 */
const _aiSdkRunModality = _withCallObsScope(async function (
  modality: ModalityKey,
  callOriginal: () => Promise<any>,
  modelLike: any,
  args0: any,
): Promise<any> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const session: any = getCurrentSession();
  if (session.inLangchain || session.inLlamaIndex || _inAiSdkMiddlewareScope()) {
    return await callOriginal();
  }
  const meta = _aiSdkCallMeta(modelLike);
  // Same shape the _logModality call below forwards on success.
  _stashAttemptContext(
    session,
    meta.provider,
    [{ model: meta.modelId }],
    modality,
    _AI_SDK_MODALITY_SHAPES[modality] ?? "unknown",
  );
  const kwargsArgs = [_aiSdkKwargs(args0, meta)];
  const intent = _buildIntent("ai_sdk", modality, kwargsArgs);
  // Null body + modelHint — see _aiSdkRunGenerate.
  await _runAsyncCheck(null, meta.provider, intent, true, false, meta.modelId);
  let order = 0;
  let spanName: string | null = null;
  try {
    order = session.nextSpanOrder();
    spanName = consumePendingSpanName();
  } catch {
    // fail-safe
  }
  const startTime = new Date();
  const callStartMono = performance.now();
  let result: any;
  try {
    result = await callOriginal();
  } catch (err) {
    const elapsedMs = Math.round(Math.max(0, performance.now() - callStartMono));
    try {
      session._call_outcome = buildCallOutcome(err, elapsedMs);
    } catch {
      // fail-safe
    }
    _emitCallFailureLog(getClient(), session);
    throw err;
  }
  const elapsedSeconds = Math.max(0, (performance.now() - callStartMono) / 1000);
  _logModality(
    "ai_sdk",
    modality,
    _AI_SDK_MODALITY_SHAPES[modality] ?? "unknown",
    kwargsArgs,
    result,
    order,
    spanName,
    startTime,
    elapsedSeconds,
  );
  return result;
});

/**
 * Patches the model methods on an AI SDK model prototype, kind-aware:
 * language → doGenerate + doStream (chat wrappers); embedding → doEmbed;
 * image/speech/transcription/video/auto → doGenerate (modality wrapper).
 * Idempotent across entry points and module copies: the same class can be
 * reached via the factory call, `.languageModel()`, `.chat()` and a bound
 * singleton, so we dedupe by prototype identity (WeakSet) plus a marker on
 * the wrapper.
 */
function _patchAiSdkPrototype(proto: any, kind: AiSdkModelKind = "language"): void {
  try {
    if (!proto || typeof proto !== "object") return;
    if (_patchedAiSdkProtos.has(proto)) return;

    const makeWrapper = (
      method: "doGenerate" | "doStream" | "doEmbed",
      original: Function,
    ): Function => {
      if (method === "doEmbed") {
        return function (this: any, ...args: any[]): Promise<any> {
          return _aiSdkRunEmbed(() => original.apply(this, args), this, args?.[0]);
        };
      }
      if (method === "doStream") {
        return function (this: any, ...args: any[]): Promise<any> {
          return _aiSdkRunStream(() => original.apply(this, args), this, args?.[0]);
        };
      }
      // doGenerate — language chat or a modality model.
      if (kind === "language") {
        return function (this: any, ...args: any[]): Promise<any> {
          return _aiSdkRunGenerate(() => original.apply(this, args), this, args?.[0]);
        };
      }
      const fixedModality = _AI_SDK_KIND_TO_MODALITY[kind] ?? null;
      return function (this: any, ...args: any[]): Promise<any> {
        const modality = fixedModality ?? _aiSdkAutoModality(args?.[0]);
        return _aiSdkRunModality(modality, () => original.apply(this, args), this, args?.[0]);
      };
    };

    const methods: Array<"doGenerate" | "doStream" | "doEmbed"> =
      kind === "embedding"
        ? ["doEmbed"]
        : kind === "language"
          ? ["doGenerate", "doStream"]
          : ["doGenerate"];

    let patchedAny = false;
    for (const method of methods) {
      const original = proto[method];
      if (typeof original !== "function") continue;
      if ((original as any).__tp_aisdk_wrapped) continue;
      _aiSdkRestore.push({ proto, method, original });
      const wrapper = makeWrapper(method, original);
      try {
        (wrapper as any).__tp_aisdk_wrapped = true;
      } catch {
        /* marker is best-effort */
      }
      proto[method] = wrapper;
      patchedAny = true;
    }
    if (patchedAny) _patchedAiSdkProtos.add(proto);
  } catch {
    // fail-safe — a failed patch must never break init
  }
}

interface AiSdkProbe {
  model: any;
  kind: AiSdkModelKind;
}

/**
 * Collects probe model instances from a provider object/function, across all
 * modality entry points. Language models are probed via `provider(id)`,
 * `.languageModel(id)`, `.chat(id)`, `.responses(id)` (providers expose
 * different — and, across versions, different-default — model classes per
 * entry point); the other modalities via their dedicated ProviderV2/V3
 * accessors. Every construction is local-only (no network).
 */
function _collectAiSdkProbeModels(provider: any, probeModelId: string): AiSdkProbe[] {
  const probes: AiSdkProbe[] = [];
  const tryAdd = (kind: AiSdkModelKind | null, make: () => any): void => {
    try {
      const m = make();
      if (!m || typeof m !== "object") return;
      const resolved = kind ?? _detectAiSdkModelKind(m);
      if (!resolved) return;
      // The instance must actually carry the method its kind implies.
      const needed = resolved === "embedding" ? m.doEmbed : m.doGenerate;
      if (typeof needed !== "function") return;
      probes.push({ model: m, kind: resolved });
    } catch {
      /* fail-safe — entry point not supported by this provider */
    }
  };
  if (!provider) return probes;
  // The bare call returns the provider's DEFAULT model — usually a language
  // model, but an image-only provider returns an image model: detect.
  if (typeof provider === "function") tryAdd(null, () => provider(probeModelId));
  tryAdd("language", () => provider.languageModel?.(probeModelId));
  tryAdd("language", () => provider.chat?.(probeModelId));
  tryAdd("language", () => provider.responses?.(probeModelId));
  tryAdd("embedding", () => provider.embeddingModel?.(probeModelId));
  tryAdd("embedding", () => provider.textEmbeddingModel?.(probeModelId));
  tryAdd("image", () => provider.imageModel?.(probeModelId));
  tryAdd("speech", () => provider.speechModel?.(probeModelId));
  tryAdd("transcription", () => provider.transcriptionModel?.(probeModelId));
  tryAdd("video", () => provider.videoModel?.(probeModelId));
  return probes;
}

/**
 * `candidate` may be a `create*` factory (takes options, returns a provider)
 * or a bound provider singleton (callable with a model id). Try the factory
 * interpretation first — it doesn't depend on provider env vars being set at
 * instrumentation time — then the singleton interpretation. A probe model
 * constructed with bogus credentials/ids is harmless: only its prototype is
 * used, and construction performs no network I/O.
 */
function _instrumentAiSdkFactory(
  candidate: any,
  options: Record<string, any>,
  probeModelId: string,
): void {
  try {
    let probes: AiSdkProbe[] = [];
    if (typeof candidate === "function") {
      try {
        const provider = candidate(options);
        probes = _collectAiSdkProbeModels(provider, probeModelId);
      } catch {
        /* fall through to the singleton interpretation */
      }
    }
    if (probes.length === 0) {
      probes = _collectAiSdkProbeModels(candidate, probeModelId);
    }
    for (const p of probes) {
      _patchAiSdkPrototype(Object.getPrototypeOf(p.model), p.kind);
    }
  } catch {
    /* fail-safe */
  }
}

/**
 * Resolves provider factories from a package namespace + registry spec,
 * constructs probe models, and patches every distinct prototype found.
 */
function _instrumentAiSdkPackage(mod: any, spec: AiSdkProviderSpec): void {
  try {
    if (!mod) return;
    const options = spec.factoryOptions ?? { apiKey: "tp_probe" };
    for (const name of spec.factoryNames) {
      const candidate = mod?.[name] ?? mod?.default?.[name];
      if (!candidate) continue;
      _instrumentAiSdkFactory(candidate, options, spec.probeModelId);
    }
  } catch {
    /* fail-safe */
  }
}

/**
 * Instruments one user-supplied `instrumentModules.aiSdkProviders` entry.
 * Accepts (a) a LanguageModel instance, (b) a provider factory/singleton
 * (callable or exposing `.languageModel()`/`.chat()`), or (c) a package
 * namespace whose `create*` exports are provider factories. This is the
 * supported path for community providers (e.g. `vercel-minimax-ai-provider`)
 * that bundle their own nested @ai-sdk/* copies, which the registry's
 * require() from this package cannot reach.
 */
export function _instrumentAiSdkEntry(entry: any): void {
  try {
    if (!entry) return;
    // (a) Model instance (any kind — language, embedding, image, speech,
    // transcription, video) — patch its own prototype.
    if (typeof entry === "object") {
      const kind = _detectAiSdkModelKind(entry);
      if (kind) {
        _patchAiSdkPrototype(Object.getPrototypeOf(entry), kind);
        return;
      }
    }
    // (b) Provider factory or bound provider singleton.
    if (
      typeof entry === "function" ||
      typeof entry?.languageModel === "function" ||
      typeof entry?.chat === "function" ||
      typeof entry?.embeddingModel === "function" ||
      typeof entry?.imageModel === "function"
    ) {
      _instrumentAiSdkFactory(entry, { apiKey: "tp_probe" }, "tp-probe");
      return;
    }
    // (c) Package namespace — try every create* export (and default).
    if (typeof entry === "object") {
      const seen = new Set<any>();
      for (const ns of [entry, entry.default]) {
        if (!ns || typeof ns !== "object" || seen.has(ns)) continue;
        seen.add(ns);
        for (const key of Object.keys(ns)) {
          if (!/^create[A-Z]/.test(key)) continue;
          const candidate = (ns as any)[key];
          if (typeof candidate !== "function") continue;
          _instrumentAiSdkFactory(candidate, { apiKey: "tp_probe" }, "tp-probe");
        }
      }
    }
  } catch {
    /* fail-safe */
  }
}

/**
 * TokenPolice middleware for the Vercel AI SDK's `wrapLanguageModel`.
 *
 * The zero-config path patches LanguageModel prototypes at init(); use this
 * middleware when a model's prototype is out of reach — e.g. a community
 * provider bundling its own @ai-sdk/* copy in a bundled/edge runtime where
 * `instrumentModules.aiSdkProviders` isn't an option:
 *
 * ```ts
 * import { wrapLanguageModel } from "ai";
 * import { tokenPoliceAiSdkMiddleware } from "token-police";
 *
 * const model = wrapLanguageModel({
 * model: minimax("MiniMax-M2"),
 * middleware: tokenPoliceAiSdkMiddleware(),
 * });
 * ```
 *
 * Structural object — no type dependency on the `ai` package; compatible with
 * the LanguageModel V2 and V3 middleware contracts. While the inner call
 * runs, an async-scoped guard suppresses the prototype wrapper so a model
 * that is BOTH wrapped and prototype-patched logs exactly once.
 */
export function tokenPoliceAiSdkMiddleware(): {
  wrapGenerate: (ctx: any) => Promise<any>;
  wrapStream: (ctx: any) => Promise<any>;
} {
  const callViaGuard = (inner: () => Promise<any>): Promise<any> => {
    try {
      return _aiSdkMiddlewareScope.run(true, () => inner());
    } catch {
      /* fail-safe — run unguarded (worst case: a double log, never a crash) */
      return inner();
    }
  };
  return {
    wrapGenerate: async ({ doGenerate, params, model }: any = {}): Promise<any> =>
      _aiSdkRunGenerate(() => callViaGuard(doGenerate), model, params),
    wrapStream: async ({ doStream, params, model }: any = {}): Promise<any> =>
      _aiSdkRunStream(() => callViaGuard(doStream), model, params),
  };
}

function _instrumentLangChainChatModels(mod: any): void {
  try {
    const BCM = mod?.BaseChatModel;
    if (typeof BCM !== "function" || !BCM.prototype) return;
    const proto = BCM.prototype;
    // `generate` covers `invoke` / `batch` / `generatePrompt` (which all
    // delegate to it). `_streamIterator` covers `stream()` AND, importantly,
    // agent / RunnableSequence internal streaming — modern LangChain agents
    // call `_streamIterator` directly when their root entry is `invoke()`.
    const targets: Array<[string, "async" | "streamIterator"]> = [
      ["generate", "async"],
      ["_streamIterator", "streamIterator"],
    ];
    for (const [method, kind] of targets) {
      const original = proto[method];
      if (typeof original !== "function") continue;
      // Idempotency: each prototype is patched at most once.
      if ((original as any).__tp_lc_wrapped) continue;
      const key = `@langchain/core:${kind}:${method}`;
      // _originals key includes the prototype identity (via a Map of WeakRefs
      // would be cleanest; for now we just track the most-recently-patched).
      _originals.set(`${key}@${(BCM as any).name ?? "BaseChatModel"}`, original);
      _setLangchainWrapper(proto, method, original, kind);
      // Restore BaseChatModel.prototype directly — `@langchain/core` doesn't
      // expose this method at a root path uninstrument() could re-resolve.
      _restoreThunks.push(() => {
        proto[method] = original;
      });
      try {
        (proto[method] as any).__tp_lc_wrapped = true;
      } catch {
        /* prototype method may be non-configurable — ignore */
      }
    }
  } catch {
    // fail-safe
  }
}

/**
 * Module-name → canonical price-store provider key. Used by the base-class
 * patch path which doesn't know the concrete provider at patch time — at
 * call time we derive it from `this.lc_namespace` or `this.constructor.name`
 * so the framework wrapper can populate `model_extras.original_provider`,
 * letting pricing be resolved against the underlying provider's price table
 * instead of failing to price under the generic "langchain" framework name.
 *
 * Keep in sync with _LC_EMBEDDING_PROVIDERS below.
 */
const _LC_NAMESPACE_TO_PROVIDER: Record<string, string> = {
  // LangChain attaches lc_namespace on every serializable component; for
  // embeddings the second segment is the provider package name. Also consulted
  // by the chat pre-flight path, so chat-only vendors (e.g. anthropic —
  // lc_namespace ["langchain", "chat_models", "anthropic"]) belong here too.
  openai: "openai",
  anthropic: "anthropic",
  cohere: "cohere",
  mistralai: "mistral",
  google_genai: "gemini",
  huggingface: "huggingface",
  voyageai: "voyage",
  together: "together",
  aws: "bedrock",
  groq: "groq",
  xai: "xai",
  deepseek: "deepseek",
};

function _deriveLangChainProvider(instance: any): string | undefined {
  try {
    const ns = instance?.lc_namespace;
    if (Array.isArray(ns)) {
      // lc_namespace e.g. ["langchain", "embeddings", "openai"]
      for (let i = ns.length - 1; i >= 0; i--) {
        const seg = String(ns[i] ?? "").toLowerCase();
        if (_LC_NAMESPACE_TO_PROVIDER[seg]) return _LC_NAMESPACE_TO_PROVIDER[seg];
      }
    }
    // Fallback: walk the constructor name (e.g. OpenAIEmbeddings).
    const ctorName: string = instance?.constructor?.name ?? "";
    const lowered = ctorName.toLowerCase();
    for (const [seg, canonical] of Object.entries(_LC_NAMESPACE_TO_PROVIDER)) {
      if (lowered.startsWith(seg)) return canonical;
    }
  } catch {
    /* fail-safe */
  }
  return undefined;
}

/**
 * Build a framework-style embedding wrapper. Runs a single pre-flight check,
 * holds the framework guard so inner provider wrappers stay inert, then
 * manually logs the call with operation="embedding". Used by both
 * _instrumentLangChainEmbeddings and _instrumentLlamaIndexEmbeddings — they
 * differ only in which guard they set + which kwargs the inner method reads.
 *
 * `originalProvider` is the canonical price-store provider key (e.g. "openai")
 * threaded from the concrete-subclass patch site. When undefined (base-class
 * patch path), the wrapper derives it at call time from `this.lc_namespace`
 * via _deriveLangChainProvider.
 */
function _makeFrameworkEmbeddingWrapper(
  original: Function,
  framework: "langchain" | "llamaindex",
  methodName: string,
  originalProvider?: string,
): Function {
  return _withCallObsScope(async function (this: any, ...args: any[]): Promise<any> {
    const session = getCurrentSession();
    // Inside any framework guard → pass through; the outer framework already
    // ran the single pre-flight check.
    if (session.inLangchain || session.inLlamaIndex) {
      return await original.apply(this, args);
    }
    // The pre-flight ran with no model and the framework slug as provider,
    // so model/provider rules never matched and group-bys bucketed to
    // "unknown". Resolve both off `this` FIRST. The model is a matching-only
    // HINT (never merged into a body — a REROUTE never applies here; the
    // enforce-mode refusal is recorded as REROUTE_REJECTED
    // `unappliable_call_shape`) and deliberately excludes the `methodName`
    // fallback the log side uses: a
    // method name is not a model. Provider falls back to the framework slug
    // (status quo) when the underlying one can't be resolved. Never throws.
    let checkModel: string | undefined;
    let checkProvider: string = framework;
    try {
      const rawModel = this && (this.model ?? this.modelName);
      if (typeof rawModel === "string" && rawModel) checkModel = rawModel;
      const resolved =
        originalProvider ??
        (framework === "langchain" ? _deriveLangChainProvider(this) : undefined);
      if (resolved) checkProvider = resolved;
    } catch {
      checkModel = undefined;
      checkProvider = framework;
    }
    await _runAsyncCheck(
      null, checkProvider, { kind: "embedding" }, true, false, checkModel,
    );

    const startTime = new Date();
    const spanName = consumePendingSpanName();
    // Pull a model hint from `this` before the call so a failure row still
    // carries the attempted model (args are text/texts, not {model}).
    const modelHint =
      (this && (this.model || this.modelName)) || methodName;
    // Stash operation=embedding + model so a provider raise still
    // emits a failed embedding row (not silent zero rows / operation=chat).
    _stashAttemptContext(session, framework, [{ model: modelHint }], "embedding");
    let order = 0;
    try {
      order = session.nextSpanOrder();
    } catch {
      // fail-safe
    }

    // Set the framework guard so inner provider wrappers (e.g.
    // openai.embeddings.create) short-circuit.
    if (framework === "langchain") session.enterLangchain();
    else session.enterLlamaIndex();

    const callStartMono = performance.now();
    let result: any;
    try {
      result = await original.apply(this, args);
    } catch (err) {
      // Success log sits after the call; without this path a provider
      // failure emits zero embedding rows. Re-throw the original error
      // unchanged (golden rule); emit itself is try/catch so a logging
      // failure cannot mask it.
      const elapsedMs = Math.round(Math.max(0, performance.now() - callStartMono));
      try {
        (session as any)._call_outcome = buildCallOutcome(err, elapsedMs);
      } catch {
        // fail-safe
      }
      _emitCallFailureLog(getClient(), session);
      throw err;
    } finally {
      if (framework === "langchain") session.exitLangchain();
      else (session as any).exitLlamaIndex?.();
    }

    // Manual log — operation="embedding". No upstream usage object survived
    // the guard (we suppressed the inner call), so approximate input tokens
    // from the composition entry lengths.
    try {
      const tp = getClient();
      if (tp) {
        // Composition: LangChain takes (texts: string[]) for embedDocuments
        // and (text: string) for embedQuery. LlamaIndex takes (text: string)
        // or (texts: string[]) for batch.
        const compKwargs: Record<string, any> = {};
        if (args.length >= 1) compKwargs.input = args[0];
        let promptComp: unknown[] = [];
        try {
          promptComp = buildPromptComposition(framework, compKwargs, "embedding");
        } catch {
          promptComp = [];
        }
        let approxTokens = 0;
        for (const e of promptComp as any[]) {
          approxTokens += Math.max(1, Math.floor((e?.length ?? 0) / 4));
        }

        const spanObj = {
          ...manualSpanIds(session),
          span_kind: "llm" as const,
          span_name: spanName ?? modelHint,
          span_order: order,
          start_time: startTime.toISOString(),
          end_time: new Date().toISOString(),
        };

        const metadata: Record<string, unknown> = {
          workflow_name: session.workflowName,
          framework,
        };
        if (session.sessionId) metadata.session_id = session.sessionId;
        // B4: `_tp_routing` is PER-CALL provenance, never session-wide. Copy the
        // session metadata WITHOUT it, then re-add it only when this row belongs
        // to the call that was actually rerouted (exact obs-key match).
        if (session.metadata) {
          copySessionMetadata(metadata, session.metadata);
        }
        // In-scope ALS read (this async fn runs inside _withCallObsScope, and
        // ALS survives the awaits above) — resolved once, shared by the stamp
        // and the keyed drain below. Never undefined (drain-all sentinel).
        const _rowObsKey: string | null = _currentObsKey();
        stampRoutingMarker(metadata, session, _rowObsKey);

        // Resolve underlying provider: concrete-class patches pass it via
        // closure — that covers every LlamaIndex embedding class (each lives in
        // its own @llamaindex/* subpackage, so the slug is known at patch time)
        // and the concrete LangChain subclasses. Only LangChain's base-class
        // patch path has no closure value; it falls back to runtime derivation
        // from lc_namespace / constructor name.
        const resolvedOriginal: string | undefined =
          originalProvider ??
          (framework === "langchain" ? _deriveLangChainProvider(this) : undefined);

        // Keyed observation drain so embedding success rows ship what the
        // pre-flight minted. Observations only: a body-less seam never owns a
        // keyed local_decision (a claim here could only steal a sibling's).
        let observations: unknown[] = [];
        try {
          observations = state.drainObservations(_rowObsKey);
        } catch {
          observations = [];
        }

        tp.log(
          session.userId,
          session.paidPlan,
          session.workflowName,
          session.sessionId,
          modelHint,
          framework,
          approxTokens,
          0, 0,
          metadata,
          spanObj,
          promptComp,
          [],
          {
            usage: {
              shape: "openai_embeddings",
              raw: { approx_input_tokens: approxTokens, approximated: true },
            },
            operation: "embedding",
            model_extras: resolvedOriginal
              ? { framework, original_provider: resolvedOriginal }
              : { framework },
            ...(observations.length > 0 ? { observations } : {}),
            planSource: session.planSource,
          },
        );
      }
    } catch {
      // fail-open
    }
    return result;
  });
}

/**
 * Instruments the Voyage AI SDK (voyageai — VoyageAIClient).
 *
 * Voyage is the Anthropic-recommended embedding provider; every Claude RAG
 * customer is a Voyage customer. The Node SDK exports `VoyageAIClient` whose
 * `embed` and `multimodalEmbed` are on the prototype. Patch both via the
 * manual-telemetry path with operation="embedding" + voyage_embed shape.
 *
 * No OpenLLMetry instrumentor exists for Voyage — manual logging only.
 * voyageai is declared as an optional peer dependency in
 * package.json so customers who don't use Voyage don't pay the install cost.
 */
function _instrumentVoyage(voyageModule: any): void {
  try {
    const VoyageAIClient =
      voyageModule?.VoyageAIClient ?? voyageModule?.default?.VoyageAIClient ?? voyageModule?.default;
    if (typeof VoyageAIClient !== "function" || !VoyageAIClient.prototype) {
      return;
    }
    const proto = VoyageAIClient.prototype;
    for (const method of ["embed", "multimodalEmbed"]) {
      if (typeof proto[method] !== "function") continue;
      // _wrapMethod owns the wrap, the dedup guard (its own
      // `voyageai:prototype:${method}` key), and the restore thunk.
      // No manual `_originals` entry here: the old
      // `voyageai:VoyageAIClient:${method}` key made uninstrument()
      // write the original as a bogus STATIC method onto the constructor while
      // leaving the prototype wrapper live.
      _wrapMethod(
        {
          moduleName: "voyageai",
          objectPath: ["prototype"],
          method,
          isAsync: true,
          manualTelemetry: true,
          operation: "embedding",
          shape: "voyage_embed",
          provider: "voyage",
        },
        VoyageAIClient,
      );
    }
  } catch {
    // fail-safe: SDK not present or its shape changed — skip
  }
}

/**
 * Patch @langchain/core/embeddings.Embeddings.{embedDocuments, embedQuery}
 * with a framework wrapper. Mirrors _instrumentLangChainChatModels —
 * patches the CJS + ESM copies independently because each provides its
 * own module record.
 */
function _patchLangChainEmbeddingClass(
  Cls: any,
  originKey: string,
  originalProvider?: string,
): void {
  if (typeof Cls !== "function" || !Cls.prototype) return;
  const proto = Cls.prototype;
  for (const method of ["embedDocuments", "embedQuery"]) {
    // Only walk Cls's own prototype keys when patching a subclass — `in proto`
    // would include inherited base methods which we DO want for the base
    // class but NOT for subclasses (the base patch would otherwise double-
    // wrap when called on a subclass instance with no override).
    const isOwn = Object.prototype.hasOwnProperty.call(proto, method);
    if (!isOwn) continue;
    const original = proto[method];
    if (typeof original !== "function") continue;
    if ((original as any).__tp_lc_emb_wrapped) continue;
    _originals.set(`${originKey}:${method}`, original);
    proto[method] = _makeFrameworkEmbeddingWrapper(
      original,
      "langchain",
      method,
      originalProvider,
    );
    // Restore this embedding class prototype directly (provider classes live in
    // their own subpackages, not at any root path uninstrument() could reach).
    _restoreThunks.push(() => {
      proto[method] = original;
    });
    try {
      (proto[method] as any).__tp_lc_emb_wrapped = true;
    } catch {
      /* non-configurable — ignore */
    }
  }
}

function _instrumentLangChainEmbeddings(mod: any): void {
  try {
    // 1. Patch the base class. Any subclass that DOESN'T override the
    // four methods inherits the wrapper for free.
    const BaseCls = mod?.Embeddings ?? mod?.default?.Embeddings;
    if (typeof BaseCls === "function" && BaseCls.prototype) {
      const proto = BaseCls.prototype;
      for (const method of ["embedDocuments", "embedQuery"]) {
        const original = proto[method];
        if (typeof original !== "function") continue;
        if ((original as any).__tp_lc_emb_wrapped) continue;
        _originals.set(`@langchain/core:embeddings:${method}`, original);
        proto[method] = _makeFrameworkEmbeddingWrapper(original, "langchain", method);
        // Restore the base Embeddings.prototype directly — not reachable from a
        // root path uninstrument() could re-resolve.
        _restoreThunks.push(() => {
          proto[method] = original;
        });
        try {
          (proto[method] as any).__tp_lc_emb_wrapped = true;
        } catch {
          /* non-configurable — ignore */
        }
      }
    }

    // 2. Patch concrete provider subclasses directly. JavaScript's prototype
    // chain finds subclass overrides FIRST, so when a provider redefines
    // `embedDocuments` on its own class body the base-class patch is
    // shadowed. Mirrors the Python MRO fix in _instrument_langchain_embeddings.
    // Third tuple element is the canonical provider pricing key (kept in
    // sync with the server's provider price catalog).
    const lcProviders: Array<[string, string, string]> = [
      ["@langchain/openai",        "OpenAIEmbeddings",             "openai"],
      ["@langchain/cohere",        "CohereEmbeddings",             "cohere"],
      ["@langchain/mistralai",     "MistralAIEmbeddings",          "mistral"],
      ["@langchain/google-genai",  "GoogleGenerativeAIEmbeddings", "gemini"],
      ["@langchain/aws",           "BedrockEmbeddings",            "bedrock"],
      // @langchain/community has lots of providers — covering only the
      // most-used Voyage entry here; community classes are best-effort.
      ["@langchain/community/embeddings/voyageai", "VoyageEmbeddings", "voyage"],
    ];
    for (const [pkg, className, canonicalProvider] of lcProviders) {
      try {
        const pkgMod = resolveProviderModule(pkg);
        if (!pkgMod) continue; // not installed — skip
        const Cls = pkgMod?.[className] ?? pkgMod?.default?.[className];
        _patchLangChainEmbeddingClass(Cls, `${pkg}:${className}`, canonicalProvider);
      } catch {
        /* not installed — skip */
      }
    }
  } catch {
    // fail-safe
  }
}

/**
 * LlamaIndex embedding subpackage → [class names, canonical provider slug].
 *
 * The slug is the price-store provider key threaded into
 * `model_extras.original_provider`. Values match
 * _LC_NAMESPACE_TO_PROVIDER for the same vendor, so both frameworks attribute
 * one vendor to one slug; the server canonicalizes the aliases. A wrong slug
 * silently breaks server-side price resolution — do not invent slugs.
 *
 * Single source of truth for both registration paths below (sync `require` and
 * the async `import()` hook), which must never drift apart.
 */
const _LI_EMBEDDING_PACKAGES: Array<[string, string[], string]> = [
  ["@llamaindex/openai", ["OpenAIEmbedding"], "openai"],
  ["@llamaindex/cohere", ["CohereEmbedding"], "cohere"],
  ["@llamaindex/mistral", ["MistralAIEmbedding"], "mistral"],
  ["@llamaindex/google", ["GeminiEmbedding"], "gemini"],
  ["@llamaindex/huggingface", ["HuggingFaceEmbedding"], "huggingface"],
];

/**
 * Patch LlamaIndex BaseEmbedding subclasses (per provider package) with the
 * framework wrapper. Mirrors _instrumentLlamaIndexProvider for chat: each
 * provider's embedding class lives in its own subpackage.
 *
 * `originalProvider` is the canonical price-store provider key for that
 * subpackage. Because every LlamaIndex embedding class is patched
 * concretely (never via a base class), it is always known here — so the wrapper
 * never has to sniff it at call time the way the LangChain base-class path does.
 */
function _instrumentLlamaIndexEmbeddings(
  mod: any,
  classNames: string[],
  originalProvider?: string,
): void {
  try {
    if (!mod) return;
    for (const className of classNames) {
      const Cls =
        mod?.[className] ??
        mod?.default?.[className] ??
        (typeof mod === "function" && (mod as any).name === className ? mod : null);
      if (typeof Cls !== "function" || !Cls.prototype) continue;
      const proto = Cls.prototype;
      for (const method of [
        "getTextEmbedding",
        "getQueryEmbedding",
        "getTextEmbeddings",
      ]) {
        const original = proto[method];
        if (typeof original !== "function") continue;
        if ((original as any).__tp_li_emb_wrapped) continue;
        _originals.set(`@llamaindex:embeddings:${className}:${method}`, original);
        proto[method] = _makeFrameworkEmbeddingWrapper(
          original,
          "llamaindex",
          method,
          originalProvider,
        );
        // Restore this embedding class prototype directly (per-provider
        // subpackage class, no root path for uninstrument() to re-resolve).
        _restoreThunks.push(() => {
          proto[method] = original;
        });
        try {
          (proto[method] as any).__tp_li_emb_wrapped = true;
        } catch {
          /* non-configurable — ignore */
        }
      }
    }
  } catch {
    // fail-safe
  }
}

/**
 * Applies the pre-flight enforcement hook to all known, installed SDKs.
 * Call this once at application startup (done automatically by init() when enforce=true).
 */
export function autoInstrument(instrumentModules?: Record<string, any>): void {
  if (_isInstrumented) return;

  // Map instrumentModules keys to module names
  // instrumentModules gives us the class/constructor directly (e.g., OpenAI class),
  // but _resolvePath expects the module-level object (e.g., { OpenAI: OpenAI }).
  // We wrap accordingly.
  const moduleMap: Record<string, any> = {};
  let openRouterModule: any;
  let cohereModule: any;
  let huggingFaceModule: any;
  let voyageModule: any;
  let mistralModule: any;
  let langChainChatModelsModule: any;
  let xaiModule: any;
  const aiSdkProviderEntries: any[] = [];
  let llamaIndexModules: { openai?: any; anthropic?: any; google?: any } = {};
  if (instrumentModules) {
    for (const [key, mod] of Object.entries(instrumentModules)) {
      try {
      const normalizedKey = key.toLowerCase();
      // Class-keyed entries: normalize a module namespace to the class first
      // (_pickClassExport) — resource statics live on the class, not the root.
      if (normalizedKey === "openai") {
        const cls = _pickClassExport(mod, "OpenAI");
        moduleMap["openai"] = { OpenAI: cls, default: cls };
      }
      else if (normalizedKey === "anthropic") {
        const cls = _pickClassExport(mod, "Anthropic");
        const entry: any = { Anthropic: cls, default: cls };
        // Carry the ROOT `APIPromise` export across the normalization. It is
        // the SOLE input to `anthropicStreamBypass` in _wrapMethod: that probe
        // reads `mod?.APIPromise ?? mod?.default?.APIPromise ??
        // mod?.Anthropic?.APIPromise` off THIS object and, when absent, routes
        // every streamed create() around the Traceloop layer. Rebuilding the
        // entry from the class alone drops the export (a class never carries
        // APIPromise on its static chain), which would flip the bypass ON for
        // every namespace-import app and silently abandon the working Traceloop
        // streaming path the probe's own comment promises to keep
        // "byte-for-byte". Read from the ORIGINAL user value, guarded — for
        // class-form input there is nothing to recover and the bypass correctly
        // stays on. Do NOT "simplify" this away.
        try {
          const apiPromise =
            mod?.APIPromise ?? mod?.default?.APIPromise ?? mod?.Anthropic?.APIPromise;
          if (typeof apiPromise === "function") entry.APIPromise = apiPromise;
        } catch {
          /* hostile getter — leave absent; bypass stays on (fail-safe) */
        }
        moduleMap["@anthropic-ai/sdk"] = entry;
      }
      // cohere-ai needs special handling (CohereClientV2 chat/chatStream are
      // instance-bound, not on the prototype) — see _instrumentCohere.
      else if (normalizedKey === "cohere") cohereModule = mod;
      // @cerebras/cerebras_cloud_sdk exports the `Cerebras` class — wrap it
      // (normalized from a namespace if needed) so _resolvePath can reach
      // Cerebras.Chat.Completions.prototype.
      else if (normalizedKey === "cerebras") {
        const cls = _pickClassExport(mod, "Cerebras");
        moduleMap["@cerebras/cerebras_cloud_sdk"] = { Cerebras: cls, default: cls };
      }
      // together-ai default-exports the `Together` class — wrap it (normalized
      // from a namespace if needed) so _resolvePath can reach
      // Together.Chat.Completions.prototype.
      else if (normalizedKey === "together" || normalizedKey === "togetherai") {
        const cls = _pickClassExport(mod, "Together");
        moduleMap["together-ai"] = { Together: cls, default: cls };
      }
      // groq-sdk default-exports the `Groq` class — wrap it (normalized from a
      // namespace if needed) so _resolvePath can reach
      // Groq.Chat.Completions.prototype.
      else if (normalizedKey === "groq") {
        const cls = _pickClassExport(mod, "Groq");
        moduleMap["groq-sdk"] = { Groq: cls, default: cls };
      }
      // @google/genai exports the `Models` class directly; pass the module
      // namespace through so _resolvePath can reach Models.prototype.
      else if (normalizedKey === "googlegenai" || normalizedKey === "google") moduleMap["@google/genai"] = mod;
      // @aws-sdk/client-bedrock-runtime — pass the namespace import through so
      // _resolvePath can reach BedrockRuntimeClient.prototype.
      else if (normalizedKey === "bedrock") moduleMap["@aws-sdk/client-bedrock-runtime"] = mod;
      // @openrouter/sdk needs special handling (Chat class not exported).
      else if (normalizedKey === "openrouter") openRouterModule = mod;
      // @huggingface/inference needs special handling — chatCompletion /
      // chatCompletionStream are non-configurable per-instance fields, so the
      // enforcer patches the package's source submodules via the require cache.
      else if (normalizedKey === "huggingface") huggingFaceModule = mod;
      // voyageai needs special handling (VoyageAIClient.embed /
      // .multimodalEmbed go through the manual-telemetry path) — see
      // _instrumentVoyage. Pass the module namespace through here.
      else if (normalizedKey === "voyageai" || normalizedKey === "voyage") voyageModule = mod;
      // @mistralai/mistralai's `Chat` class is not root-exported — see
      // _instrumentMistral. Pass the module namespace through here.
      else if (normalizedKey === "mistral" || normalizedKey === "mistralai") mistralModule = mod;
      // @ai-sdk/xai — legacy key, routed through the generic Vercel AI SDK
      // path. Accepts either the package namespace or the `createXai`
      // factory directly.
      else if (normalizedKey === "xai" || normalizedKey === "aisdkxai") xaiModule = mod;
      // Vercel AI SDK providers (generic). Accepts an array of provider
      // factories / singletons / model instances / package namespaces — see
      // _instrumentAiSdkEntry. This is the supported path for community
      // providers that bundle their own nested @ai-sdk/* copies.
      else if (normalizedKey === "aisdkproviders") {
        if (Array.isArray(mod)) aiSdkProviderEntries.push(...mod);
        else if (mod) aiSdkProviderEntries.push(mod);
      }
      // @langchain/core needs special handling — CJS + ESM are distinct
      // module records, so we patch each one we can resolve. Accept either
      // the chat_models module directly, or an object with sub-modules.
      else if (normalizedKey === "langchain") {
        if (mod && typeof mod === "object" && mod.chatModelsModule) {
          langChainChatModelsModule = mod.chatModelsModule;
        } else if (mod && typeof mod === "object" && mod.BaseChatModel) {
          langChainChatModelsModule = mod;
        }
      }
      // LlamaIndex provider modules. Each provider lives in its own subpackage
      // (@llamaindex/openai, @llamaindex/anthropic, @llamaindex/google) — accept
      // them under a single `llamaIndex: { openaiModule, anthropicModule,
      // geminiModule }` envelope so callers don't have to call init() multiple
      // times.
      else if (normalizedKey === "llamaindex") {
        if (mod && typeof mod === "object") {
          llamaIndexModules.openai = mod.openaiModule ?? mod.openAIModule ?? mod.openai;
          llamaIndexModules.anthropic = mod.anthropicModule ?? mod.anthropic;
          llamaIndexModules.google = mod.geminiModule ?? mod.googleModule ?? mod.google;
        }
      }
      } catch (err) {
        // PREP-PHASE ISOLATION: a malformed / partially-installed entry whose
        // property getter throws (e.g. mod.chatModelsModule / mod.BaseChatModel
        // / mod.openaiModule) must NOT abort the whole prep loop. Log-and-skip
        // this one entry (gated on logErrors) and keep building moduleMap for
        // the rest, so one bad SDK doesn't skip instrumenting all the others.
        _warnSetupFailure(`instrumentModules[${JSON.stringify(key)}]`, err);
        continue;
      }
    }
  }

  for (const target of _TARGET_METHODS) {
    // PER-TARGET ISOLATION: a non-require throw in _wrapMethod's post-require
    // wrap body (resolve / unwrap / setattr) must not abort the loop — skip the
    // bad target and keep instrumenting the rest. Never throws into init().
    try {
      _wrapMethod(target, moduleMap[target.moduleName]);
    } catch (err) {
      _warnSetupFailure(`${target.moduleName}.${target.objectPath.join(".")}.${target.method}`, err);
    }
  }

  // Anthropic Message Batches results retrieval (A5) — one llm row per
  // succeeded entry with tier='batch'; batch spend is otherwise invisible.
  // Use the user-supplied module when present, else auto-discover.
  try {
    let anthropicEntry: any = moduleMap["@anthropic-ai/sdk"];
    if (!anthropicEntry) {
      // App-first, never throws; undefined when not installed — skip below.
      anthropicEntry = resolveProviderModule("@anthropic-ai/sdk");
    }
    if (anthropicEntry) _instrumentAnthropicBatches(anthropicEntry);
  } catch {
    // fail-safe
  }

  // Anthropic `Messages.stream()` (the `.on('text')` / `for await` /
  // `finalMessage()` context-manager helper) is a DISTINCT API from
  // `create({stream:true})` and is now instrumented via a manual-telemetry path
  // (pre-flight `/check` gated at every token-delivery surface + one `/log` from
  // MessageStream.finalMessage(), with the Traceloop `.stream` OTel span
  // suppressed to avoid a double-log) — parity with the Python SDK's
  // _instrument_anthropic_stream. Per-target isolated: a broken anthropic module
  // never throws into init().
  try {
    let anthropicStreamEntry: any = moduleMap["@anthropic-ai/sdk"];
    if (!anthropicStreamEntry) {
      // App-first, never throws; undefined when not installed — skip below.
      anthropicStreamEntry = resolveProviderModule("@anthropic-ai/sdk");
    }
    if (anthropicStreamEntry) _instrumentAnthropicStream(anthropicStreamEntry);
  } catch {
    // fail-safe
  }

  // Native OpenRouter SDK — instrumented separately (see _instrumentOpenRouter).
  if (openRouterModule) {
    _instrumentOpenRouter(openRouterModule);
  }

  // Cohere v2 SDK — instrumented separately (see _instrumentCohere). Fall back
  // to app-first resolution so auto-discovery (no instrumentModules) works too.
  if (!cohereModule) {
    cohereModule = resolveProviderModule("cohere-ai");
  }
  if (cohereModule) {
    _instrumentCohere(cohereModule);
  }

  // HuggingFace SDK — instrumented separately (see _instrumentHuggingFace).
  // Fall back to app-first resolution so auto-discovery (no instrumentModules)
  // works too.
  if (!huggingFaceModule) {
    huggingFaceModule = resolveProviderModule("@huggingface/inference");
  }
  if (huggingFaceModule) {
    _instrumentHuggingFace(huggingFaceModule);
  }

  // Mistral SDK — instrumented separately (see _instrumentMistral). Fall back
  // to app-first resolution so auto-discovery (no instrumentModules) works too.
  if (!mistralModule) {
    mistralModule = resolveProviderModule("@mistralai/mistralai");
  }
  if (mistralModule) {
    _instrumentMistral(mistralModule);
  }

  // Vercel AI SDK — generic LanguageModel V2/V3 instrumentation. User-supplied
  // entries run first so first-call coverage is guaranteed for the exact
  // module instances the app uses; then every registry package we can resolve
  // is probed (require for CJS, async dynamic import for ESM-only installs —
  // the latter may settle after the app's first call, same caveat as the
  // LangChain/LlamaIndex fallbacks below).
  if (xaiModule) {
    _instrumentAiSdkEntry(xaiModule);
  }
  for (const entry of aiSdkProviderEntries) {
    _instrumentAiSdkEntry(entry);
  }
  for (const spec of _AI_SDK_REGISTRY) {
    try {
      const mod = resolveProviderModule(spec.pkg);
      if (mod) _instrumentAiSdkPackage(mod, spec);
    } catch {
      // fail-safe — never abort the loop
    }
  }
  (async () => {
    for (const spec of _AI_SDK_REGISTRY) {
      try {
        // @ts-ignore — optional peer dep; not part of the SDK's bundled types.
        const mod = await import(spec.pkg);
        _instrumentAiSdkPackage(mod, spec);
      } catch {
        /* fail-safe */
      }
    }
  })();

  // Voyage AI — instrumented separately (see _instrumentVoyage). Optional peer
  // dep; auto-discovery via app-first resolution.
  //
  // The user-supplied module goes FIRST, and the order is load-bearing:
  // _wrapMethod dedups on `voyageai:prototype:${method}`, a key that
  // carries no module identity — so whichever module record is wrapped FIRST
  // wins and every later copy is silently skipped. The app's own imported copy
  // is the one its calls actually run through, and in the ESM case
  // instrumentModules exists to solve, the resolveProviderModule() / import()
  // copies below can be a different record entirely. Do not reorder.
  // _instrumentVoyage is fully self-guarded, so this can never throw into init().
  if (voyageModule) _instrumentVoyage(voyageModule);
  try {
    const voyageMod = resolveProviderModule("voyageai");
    if (voyageMod) _instrumentVoyage(voyageMod);
  } catch {
    // fail-safe
  }
  (async () => {
    try {
      // @ts-ignore — optional peer dep
      const mod = await import("voyageai");
      _instrumentVoyage(mod);
    } catch {
      /* fail-safe */
    }
  })();

  // LangChain — instrumented separately because CJS and ESM
  // @langchain/core/language_models/chat_models are distinct module records,
  // and consumers like @langchain/openai use the ESM one. Patch every copy
  // we can resolve.
  // 1. User-provided ESM module (recommended for first-call correctness).
  if (langChainChatModelsModule) {
    _instrumentLangChainChatModels(langChainChatModelsModule);
  }
  // 2. CJS top-level — patches the CJS prototype (covers CJS consumers).
  // App-first resolution so the copy patched is the app's, not the SDK's.
  try {
    const lcChatMod = resolveProviderModule(
      "@langchain/core/language_models/chat_models",
    );
    if (lcChatMod) _instrumentLangChainChatModels(lcChatMod);
  } catch {
    /* fail-safe */
  }
  // 3. ESM top-level — async best-effort. May settle after the first LLM
  // call, so apps that need first-call coverage should pass the module
  // via instrumentModules.langChain.chatModelsModule.
  (async () => {
    try {
      // @ts-ignore — optional peer dep; not part of the SDK's bundled types.
      const mod = await import("@langchain/core/language_models/chat_models");
      _instrumentLangChainChatModels(mod);
    } catch {
      /* fail-safe */
    }
  })();

  // LlamaIndex — each provider lives in its own subpackage. Patch the LLM
  // class prototypes (chat / complete) for each provider we can resolve.
  // 1. User-provided ESM modules (recommended for first-call correctness).
  if (llamaIndexModules.openai) {
    _instrumentLlamaIndexProvider(llamaIndexModules.openai, ["OpenAI", "OpenAIResponses"]);
  }
  if (llamaIndexModules.anthropic) {
    _instrumentLlamaIndexProvider(llamaIndexModules.anthropic, ["Anthropic"]);
  }
  if (llamaIndexModules.google) {
    _instrumentLlamaIndexProvider(llamaIndexModules.google, ["Gemini", "GoogleGenAI"]);
  }
  // 2. CJS fallback — try require() so auto-discovery works without
  // instrumentModules. Each call is independent and fail-safe.
  for (const [pkg, classes] of [
    ["@llamaindex/openai", ["OpenAI", "OpenAIResponses"]],
    ["@llamaindex/anthropic", ["Anthropic"]],
    ["@llamaindex/google", ["Gemini", "GoogleGenAI"]],
  ] as Array<[string, string[]]>) {
    try {
      const mod = resolveProviderModule(pkg);
      if (mod) _instrumentLlamaIndexProvider(mod, classes);
    } catch {
      /* fail-safe */
    }
  }
  // 3. ESM dynamic import fallback — best-effort; may settle after first call.
  (async () => {
    for (const [pkg, classes] of [
      ["@llamaindex/openai", ["OpenAI", "OpenAIResponses"]],
      ["@llamaindex/anthropic", ["Anthropic"]],
      ["@llamaindex/google", ["Gemini", "GoogleGenAI"]],
    ] as Array<[string, string[]]>) {
      try {
        // @ts-ignore — optional peer dep; not part of the SDK's bundled types.
        const mod = await import(pkg);
        _instrumentLlamaIndexProvider(mod, classes);
      } catch {
        /* fail-safe */
      }
    }
  })();

  // LangChain Embeddings — same CJS+ESM dual-patch strategy as chat. No
  // OpenLLMetry instrumentor exists for embeddings; the framework wrapper
  // runs the single pre-flight check + holds inLangchain so the inner
  // provider wrapper (e.g. openai.embeddings.create) short-circuits.
  try {
    const lcEmbMod = resolveProviderModule("@langchain/core/embeddings");
    if (lcEmbMod) _instrumentLangChainEmbeddings(lcEmbMod);
  } catch {
    /* fail-safe */
  }
  (async () => {
    try {
      // @ts-ignore — optional peer dep
      const mod = await import("@langchain/core/embeddings");
      _instrumentLangChainEmbeddings(mod);
    } catch {
      /* fail-safe */
    }
  })();

  // LlamaIndex Embedding subclasses — per provider, mirroring the chat
  // pattern. Embedding class names per LlamaIndex Node conventions.
  for (const [pkg, classes, originalProvider] of _LI_EMBEDDING_PACKAGES) {
    try {
      const mod = resolveProviderModule(pkg);
      if (mod) _instrumentLlamaIndexEmbeddings(mod, classes, originalProvider);
    } catch {
      /* fail-safe */
    }
  }
  (async () => {
    for (const [pkg, classes, originalProvider] of _LI_EMBEDDING_PACKAGES) {
      try {
        // @ts-ignore — optional peer dep
        const mod = await import(pkg);
        _instrumentLlamaIndexEmbeddings(mod, classes, originalProvider);
      } catch {
        /* fail-safe */
      }
    }
  })();

  _isInstrumented = true;
  logger.debug("TokenPolice: Pre-flight enforcement hooks applied.");
}

/**
 * Restores the original SDK methods, removing TokenPolice pre-flight enforcement.
 * Useful for testing or dynamic reconfiguration.
 */
export function uninstrument(): void {
  if (!_isInstrumented) return;

  // Primary restore path: run every thunk captured at its patch site. Each is
  // isolated in its own try/catch so one broken restore can't stop the rest.
  // This is what makes probe-derived surfaces (Cohere v2, OpenRouter, Voyage,
  // the LangChain/LlamaIndex framework classes, Anthropic Batches) — and every
  // root-exported surface too — restore correctly.
  for (const thunk of _restoreThunks) {
    try {
      thunk();
    } catch (e) {
      logger.debug(
        `Failed to run restore thunk: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }
  _restoreThunks.length = 0;

  // Legacy string-key fallback, kept for any un-migrated root-exported surface.
  // For thunk-covered keys this is a harmless idempotent second write; for
  // probe-derived keys `require(moduleName)` + root path can't resolve the
  // patched object, so it no-ops (the thunk above already restored it).
  for (const [key, original] of _originals) {
    // HuggingFace patches require-cache submodule exports — restored separately
    // below (they have no resolvable path on the package root).
    if (key.startsWith("@huggingface/inference:")) continue;
    // Mistral's Chat class isn't root-exported, so we can't reach
    // Chat.prototype via _resolvePath on the package — restored from
    // _mistralRestore below.
    if (key.startsWith("@mistralai/mistralai:")) continue;
    try {
      const parts = key.split(":");
      const moduleName = parts[0];
      const pathStr = parts[1];
      const methodName = parts[2];

      // Resolved through the same cache _wrapMethod populated, so the restore
      // lands on the EXACT module object that was wrapped — a fresh require
      // could hit a different copy on file:/pnpm topologies.
      const mod = resolveProviderModule(moduleName);
      const obj = mod ? _resolvePath(mod, pathStr.split(".")) : null;
      if (obj) {
        obj[methodName] = original;
      }
    } catch (e) {
      logger.debug(
        `Failed to uninstrument ${key}: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  }

  // Restore HuggingFace source-submodule exports directly.
  for (const { exportsObj, method, original } of _hfRestore) {
    try {
      exportsObj[method] = original;
    } catch {
      // ignore
    }
  }
  _hfRestore.length = 0;

  // Restore Mistral Chat.prototype methods (Chat is not root-exported).
  for (const { proto, method, original } of _mistralRestore) {
    try {
      proto[method] = original;
    } catch {
      // ignore
    }
  }
  _mistralRestore.length = 0;

  // Restore Vercel AI SDK LanguageModel prototypes (classes not root-exported).
  for (const { proto, method, original } of _aiSdkRestore) {
    try {
      proto[method] = original;
    } catch {
      // ignore
    }
  }
  _aiSdkRestore.length = 0;
  // WeakSet has no clear() — reassign so re-instrumenting works.
  _patchedAiSdkProtos = new WeakSet<object>();

  // Restore Anthropic Messages/AsyncMessages `.stream` + clear the
  // idempotency marker so a later re-instrument re-wraps.
  for (const { proto, method, original } of _anthropicStreamRestore) {
    try {
      proto[method] = original;
      delete (proto as any).__tpStreamPatched;
    } catch {
      // ignore
    }
  }
  _anthropicStreamRestore.length = 0;

  _originals.clear();
  // Reset the zero-wrap audit counts so a later re-init recomputes them from
  // scratch (stale counts would suppress a legitimate warning).
  _wrappedTargetCounts.clear();
  _isInstrumented = false;

  try {
    const { unsetupOpenTelemetry } = require("./telemetry");
    unsetupOpenTelemetry();
  } catch (e) {
    logger.debug(
      `Failed to unsetup opentelemetry: ${e instanceof Error ? e.message : String(e)}`,
    );
  }

  logger.debug("TokenPolice: Pre-flight enforcement hooks removed.");
}

// Test-only re-export of latency internals. Not part of the public API; kept
// at module scope so the vitest suite can assert the stream-wrapper fail-safety
// and TTFT math without going through full SDK instrumentation.
export const __test__ = {
  _buildStreamLatency,
  _streamAccHasContent,
  _newStreamAccumulator,
  _accumulateStreamChunk,
  _wrapManualStream,
  _wrapBedrockConverseStream,
  _chunkHasUsage,
  _tapStreamUsageForOnEnd,
  _runAsyncCheck,
  _handleBedrockEmbeddingInvoke,
  _makeFrameworkEmbeddingWrapper,
  _instrumentLlamaIndexEmbeddings,
  _LI_EMBEDDING_PACKAGES,
  MODALITY_HANDLERS,
  _buildIntent,
  _audioFileSeconds,
  _wantsGoogleAudioOut,
  _isGoogleAudioOutput,
  _resolveGoogleLogOperation,
  _resolveFailOperation,
  _stashAttemptContext,
  _emitCallFailureLog,
  _imageSizeFromDims,
  _parseImageSizeStr,
  _imageDimsFromBinary,
  _resolveImageSize,
  _extractResponsesImageToolConfig,
  _listResponsesImageCalls,
  _logResponsesImageChildren,
  _extractRawUsage,
  _instrumentMistral,
  _instrumentHuggingFace,
  _instrumentCohere,
  _instrumentOpenRouter,
  _instrumentVoyage,
  _instrumentAnthropicBatches,
  _instrumentLangChainChatModels,
  _instrumentLlamaIndexProvider,
  _restoreThunks,
  // Lets the uninstrument()-restore suite flip the guard so uninstrument()
  // actually runs after a single _instrumentX() call (which doesn't set it).
  _setInstrumented: (v: boolean): void => {
    _isInstrumented = v;
  },
  _applyReroute,
  _detectProvider,
  _effectiveProvider,
  _wireParseKey,
  _resolveServingProvider,
  _resolveServingFromBaseUrl,
  _matchHostToProvider,
  _extractHost,
  _resolveUsageShape,
  _streamAccumulatorToResponse,
  _captureCompositionAt,
  _aiSdkRunStream,
  _captureLangchainResponse,
  _captureLangchainResponseFromMessage,
  _captureLlamaIndexResponseAt,
  _isAiSdkParse,
  _mapAiSdkProvider,
  _TARGET_METHODS,
  // App-first resolution + zero-wrap audit internals (enforcerResolution /
  // enforcerZeroWrapWarning suites).
  resolveProviderModule,
  _resolveWithAnchors,
  _resolvedModules,
  _wrappedTargetCounts,
  _appCanResolve,
  _requireCache,
  _getAppRequire,
  _stashGatewayRequestModel,
  // Non-streaming openai-wire verbatim usage stash (G3-14-2) — exposed so the
  // nonStreamVerbatimUsage suite can drive the gates directly instead of
  // standing up a full OpenAI SDK + instrumentor.
  _hasUsageDetailObjects,
  _stashNonStreamVerbatimUsage,
  _stashProviderOverride,
  _stashApiBase,
  _extractLlamaIndexUsage,
  _logLlamaIndex,
  _buildLlamaIndexVerbatimUsage,
  _maybeTapOpenAITtsSse,
  _parseOpenAITtsSseBuffer,
  _callWithInjectedStreamUsage,
  _shouldRetryWithoutInjection,
  _injectStreamUsageOption,
  TP_CAPTURED_USAGE,
};
