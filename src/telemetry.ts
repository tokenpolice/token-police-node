/**
 * Passive Token Extraction Layer.
 *
 * Sets up a LOCAL OpenTelemetry provider with ZERO EXPORTERS.
 * Attaches a SpanProcessor that intercepts gen_ai.usage.* attributes
 * and routes them to the TokenPolice API via the client.
 *
 * PRIVACY GUARANTEE: No telemetry data ever leaves the process via
 * OpenTelemetry. Only aggregated token counts are sent to TokenPolice.
 *
 * Instrumentation strategy (inspired by lmnr-ts, adapted for zero-export):
 * 1. Creates a private `BasicTracerProvider`. If no real global provider exists
 * yet, registers ours globally; if a customer already installed one, leaves it
 * in place and attaches our TokenPoliceSpanProcessor to it instead (resolving
 * the real provider through the global proxy's delegate). See setupOpenTelemetry
 * below; coexistence is covered by tests/otelCoexistence.test.ts.
 * 2. Constructs instrumentors directly and calls setTracerProvider(ours) —
 * registerInstrumentations() is never used. Note the InstrumentationBase
 * CONSTRUCTOR is what registers the require-in-the-middle / import-in-the-middle
 * loader hooks, so hooks are live the moment we `new` the instrumentor; the
 * later enable() call is a no-op. See hardenInstrumentorPatches below.
 * 3. Supports two modes:
 * a) instrumentModules: User passes module references directly (works in ESM/CJS)
 * b) Auto-discovery: Dynamically resolves installed instrumentors (CJS fallback)
 *
 * Behavior is kept in parity with the TokenPolice Python SDK.
 */

import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { AsyncLocalStorage } from "node:async_hooks";

import { trace, context, propagation, ROOT_CONTEXT, SpanStatusCode, type Context, type Tracer } from "@opentelemetry/api";
import {
  BasicTracerProvider,
  type SpanProcessor,
  type ReadableSpan,
} from "@opentelemetry/sdk-trace-base";
import type { Span } from "@opentelemetry/api";

import {
  getClient,
  drainObservations,
  getCurrentObsKey,
  beginObservationsDrainAll,
  endObservationsDrainAll,
} from "./state";
import {
  getCurrentSession,
  consumePendingSpanName,
  consumeReservedSpanOrder,
  anthropicStreamOtelSuppressActive,
  llamaIndexOtelSuppressActive,
  _setAgentTracerFactory,
  randomHex16,
} from "./context";
import { claimLocalDecision } from "./localDecisionStore";
import { TP_ROUTING_ATTR, stampRoutingMarker } from "./routingMarkerStore";
import { createHash } from "node:crypto";
import { scrubErrorMessage, resolveErrorDetail, toDurationMs } from "./_classify";
import { codePointLength, hashLen } from "./composition";

// ── Module state ──────────────────────────────────────────────────

let _isSetup = false;
let _tracerProvider: BasicTracerProvider | undefined;
// Whether setup registered OUR globals (so teardown disables only what we
// installed and never the customer's). Setup sets the global tracer provider
// (no propagator — see setupOpenTelemetry); the context manager is installed
// separately and only "wins" when no customer manager already exists.
let _registeredGlobalProvider = false;
let _registeredGlobalContextManager = false;
/** Tracked instrumentations for cleanup */
let _activeInstrumentations: any[] = [];

const logger = {
  debug: (msg: string | (() => string)) => {
    try {
      if ((getClient() as any)?.logErrors) {
        console.log(`[TokenPolice Debug] ${typeof msg === "function" ? msg() : msg}`);
      }
    } catch { /* fail-open: silent */ }
  },
  warning: (msg: string) => console.warn(`[TokenPolice Warning] ${msg}`),
};

// ── Instrumentor resolution ──────────────────────────────────────
//
// Instrumentors are resolved first from the application's node_modules
// (process.cwd()), then — as a fallback — from the SDK's OWN location. The
// fallback matters because @traceloop/instrumentation-openai / -anthropic are
// now optionalDependencies of token-police: with npm/yarn they hoist to the
// app's top-level node_modules (cwd resolution finds them), but pnpm / yarn-PnP
// keep them nested under token-police, where only an SDK-relative require can
// see them. esbuild/tsup substitutes import.meta.url in the CJS output, so this
// is safe across both the ESM and CJS dist files.
let sdkRequire: NodeRequire;
try {
  sdkRequire = createRequire(import.meta.url);
} catch {
  // `require` exists in the CJS build; in the unlikely event neither is
  // available, fall back to a cwd-seeded require so this never throws.
  try {
    sdkRequire = require;
  } catch {
    sdkRequire = createRequire(join(process.cwd(), "node_modules"));
  }
}

/**
 * Resolve an instrumentor package, trying the app's node_modules first and the
 * SDK's own location second. Returns undefined if neither can load it (never
 * throws — capture must fail open).
 */
function resolveInstrumentor(
  appRequire: NodeRequire,
  pkg: string,
): any | undefined {
  try {
    return appRequire(pkg);
  } catch {
    /* not in app node_modules — try SDK-relative */
  }
  try {
    return sdkRequire(pkg);
  } catch {
    return undefined;
  }
}

/** Whether a package (target SDK or instrumentor) is resolvable at all. */
function canResolve(appRequire: NodeRequire, pkg: string): boolean {
  try {
    appRequire.resolve(pkg);
    return true;
  } catch {
    /* not in app node_modules */
  }
  try {
    sdkRequire.resolve(pkg);
    return true;
  } catch {
    return false;
  }
}

/** Providers we've already warned about (one warning per provider, per process). */
const _warnedProviders = new Set<string>();

/**
 * `_warnedProviders` key for the manual-mode LangChain opt-in notice (see
 * _applyLangChainInstrumentation). Namespaced so it can never collide with a
 * real INSTRUMENTOR_REGISTRY moduleKey.
 */
const LANGCHAIN_OPT_IN_WARN_KEY = "langChain:manual-opt-in";

// ── Tool-span recognition (cross-framework) ──────────────────────
// No single discriminator works across frameworks/versions, so we OR several:
// gen_ai.operation.name="execute_tool" → Node LangChain, OTel-native, OpenAI Agents
// traceloop.span.kind="tool" → Python LangChain, older LlamaIndex
// openinference.span.kind="TOOL" → Arize-instrumented apps
// name "execute_tool <tool>" / "<tool>.tool" → last-resort name heuristic
function isToolSpan(attrs: Record<string, unknown>, name: string): boolean {
  try {
    if (attrs["gen_ai.operation.name"] === "execute_tool") return true;
    if (String(attrs["traceloop.span.kind"] ?? "").toLowerCase() === "tool") return true;
    if (String(attrs["openinference.span.kind"] ?? "").toUpperCase() === "TOOL") return true;
    const n = name || "";
    if (n.startsWith("execute_tool ")) return true;
    if (n.endsWith(".tool")) return true;
  } catch {
    // fall through
  }
  return false;
}

/**
 * True for an OpenLLMetry embeddings span — mirrors Python's
 * embedding-span detector for parity with the Python SDK.
 *
 * opentelemetry-instrumentation-openai / -cohere patch embeddings and emit a
 * span tagged `llm.request.type="embedding"` (span names "openai.embeddings" /
 * "cohere.embed"). TokenPolice captures embeddings authoritatively via its own
 * manual wrapper, which creates NO OTel span — so any embedding span that
 * reaches the SpanProcessor is the instrumentor's spurious duplicate (it would
 * otherwise be mis-logged with the generic *_chat shape). Used to drop it.
 * Covered by embeddingSpanDedup.test.ts.
 */
export function isInstrumentorEmbeddingSpan(
  attrs: Record<string, unknown>,
  name: string,
): boolean {
  try {
    if (String(attrs["llm.request.type"] ?? "").toLowerCase() === "embedding") return true;
    const n = name || "";
    if (n === "openai.embeddings" || n === "cohere.embed" || n.endsWith(".embeddings")) return true;
    // opentelemetry-instrumentation-bedrock emits a chat/converse-shaped
    // span on the embeddings InvokeModel call (no llm.request.type=embedding, no
    // *.embeddings name), tagged with the vendor-stripped embedding model id.
    // TokenPolice captures Bedrock embeddings on the manual path, so this span is
    // the spurious duplicate. Match by model id — mirrors
    // enforcer._BEDROCK_EMBEDDING_MODEL_PREFIXES (amazon.titan-embed /
    // cohere.embed / voyage.voyage). No Bedrock chat model id contains "embed" or
    // starts with "voyage", so chat spans are never dropped. Falsy-coalescing (||,
    // not ??) to match Python's `str(a or b or "")`: an
    // empty-string primary key falls through to the legacy key.
    const modelId = String(
      attrs["gen_ai.request.model"] || attrs["llm.request.model"] || "",
    ).toLowerCase();
    if (modelId !== "" && (modelId.includes("embed") || modelId.startsWith("voyage"))) return true;
  } catch {
    // fall through
  }
  return false;
}

/** Tool name lives in different keys per framework — read all of them. */
function toolNameFromSpan(attrs: Record<string, unknown>, name: string): string {
  const tn = attrs["gen_ai.tool.name"] ?? attrs["traceloop.entity.name"] ?? attrs["tool.name"];
  if (tn) return String(tn);
  const n = name || "";
  if (n.startsWith("execute_tool ")) return n.slice("execute_tool ".length);
  if (n.endsWith(".tool")) return n.slice(0, -".tool".length);
  return n || "tool";
}

/**
 * Correct LangChain (JS) tool-span names. Covered by langchain-tool-naming.test.ts.
 *
 * `@traceloop/instrumentation-langchain`'s `handleToolStart` derives the tool
 * name from `tool.id[last]` — the LangChain serialization *class path* — so JS
 * tools created as `DynamicStructuredTool`/`StructuredTool` (which are not
 * serializable, so `toJSON()` yields only the class id) are mislabeled with the
 * CLASS name (e.g. "DynamicStructuredTool") instead of their real instance name
 * (e.g. "getCustomerInfo"). The real name is passed as the 7th arg `runName`
 * (which `@langchain/core` defaults to the tool's `.name`) but the instrumentor
 * discards it. The Python instrumentor uses the real name, so Python is fine.
 *
 * This wraps `handleToolStart` on the shared callback-handler prototype and,
 * after the original runs, fixes the span name + `traceloop.entity.name` /
 * `gen_ai.tool.name` (the keys `toolNameFromSpan` reads) **only when a better
 * name is available** — otherwise upstream behavior is left untouched. Guarded,
 * idempotent, and fully fail-open so it can never break instrumentation.
 *
 * @returns true if the class is (already) patched, false if it could not be.
 */
export function patchLangChainToolNaming(HandlerClass: any): boolean {
  try {
    const proto = HandlerClass?.prototype;
    if (!proto || typeof proto.handleToolStart !== "function") return false;
    if (proto.__tpToolNamePatched) return true;
    const original = proto.handleToolStart;
    proto.handleToolStart = async function (
      tool: any,
      input: any,
      runId: any,
      parentRunId?: any,
      tags?: any,
      metadata?: any,
      runName?: any,
    ) {
      const ret = await original.call(this, tool, input, runId, parentRunId, tags, metadata, runName);
      try {
        const classish = Array.isArray(tool?.id) ? tool.id[tool.id.length - 1] : undefined;
        const real =
          (typeof runName === "string" && runName) ||
          (typeof tool?.name === "string" && tool.name) ||
          (typeof tool?.kwargs?.name === "string" && tool.kwargs.name) ||
          undefined;
        if (real && real !== classish) {
          const span = this?.spans?.get?.(runId)?.span;
          if (span) {
            if (typeof span.updateName === "function") span.updateName(`execute_tool ${real}`);
            span.setAttribute?.("traceloop.entity.name", real);
            span.setAttribute?.("gen_ai.tool.name", real);
          }
        }
      } catch {
        // fail-open: never break instrumentation over a cosmetic span name
      }
      return ret;
    };
    proto.__tpToolNamePatched = true;
    return true;
  } catch {
    return false;
  }
}

// `hashLen` (sha1-16 + code-point length for a tool arg/result) now lives in
// composition.ts as the single shared copy — imported above.

// ── Provider definition for supported instrumentModules ───────────

/**
 * Map of provider keys to their @traceloop/instrumentation-* package and class.
 * Used for both auto-discovery and manual instrumentModules mode.
 */
const INSTRUMENTOR_REGISTRY: Array<{
  /** Key in InstrumentModules interface */
  moduleKey: string;
  /** npm package name */
  packageName: string;
  /** Exported class name */
  className: string;
  /** The npm package name of the target SDK (for auto-discovery fallback) */
  targetModule: string;
  /**
   * Coerces the user's `instrumentModules[moduleKey]` value into the shape THIS
   * instrumentor's `manuallyInstrument(module)` reads, before it is handed
   * over. A per-entry function — never a shared switch — because the two
   * instrumentors want OPPOSITE shapes and handing over the wrong one throws
   * inside manuallyInstrument(), silently disabling metering:
   *
   * - @traceloop/instrumentation-openai (0.26) does
   *   `this._wrap(openaiModule.Chat.Completions.prototype, "create", …)` and
   *   `this._wrap(openaiModule.Completions.prototype, "create", …)`. Those
   *   resource statics live on the `OpenAI` CLASS, not on the namespace — so
   *   this entry normalizes TO THE CLASS (_pickOpenAIClass).
   * - @traceloop/instrumentation-anthropic (0.27) does
   *   `this._wrap(module.Anthropic.Completions.prototype, "create", …)`,
   *   `…module.Anthropic.Messages.prototype…` and
   *   `…module.Anthropic.Beta.Messages.prototype…`. It needs an object with an
   *   `.Anthropic` property carrying those resource statics — i.e. the module
   *   NAMESPACE. Handing it the class throws "Cannot read properties of
   *   undefined (reading 'prototype')", because `Anthropic.Anthropic` is the
   *   inherited `BaseAnthropic` static self-ref, which does NOT carry
   *   `.Messages`. So this entry normalizes TO THE NAMESPACE
   *   (_pickAnthropicNamespace), the exact opposite of openAI.
   *
   * Because the detectors are SDK-specific (one probes `.OpenAI`, the other
   * `.Anthropic`), a NEW registry entry must supply its own — there is
   * deliberately no "class"/"namespace" enum to select one of these by, since
   * that would let a third provider silently inherit OpenAI's probe. Every
   * implementation must return its input verbatim on no-match and must never
   * throw (golden rule).
   *
   * Consumed ONLY by _applyManualInstrumentations. _applyAutoDiscoveredInstrumentations
   * must NEVER call this: there, the `require()` that produced the module has
   * ALREADY driven the loader hook, so the copy is patched. Normalizing would
   * make its eager manuallyInstrument() succeed too and stack a SECOND
   * Traceloop layer on the same prototypes — double-counted spans and cost. Its
   * eager attempt is expected to fail on a shape mismatch, which is exactly why
   * that failure is handled as debug-only and non-fatal.
   */
  normalizeManualTarget: (mod: any) => any;
}> = [
  {
    moduleKey: "openAI",
    packageName: "@traceloop/instrumentation-openai",
    className: "OpenAIInstrumentation",
    targetModule: "openai",
    // Hoisted function declaration — safe to reference from this initializer.
    normalizeManualTarget: (mod) => _pickOpenAIClass(mod),
  },
  {
    moduleKey: "anthropic",
    packageName: "@traceloop/instrumentation-anthropic",
    className: "AnthropicInstrumentation",
    targetModule: "@anthropic-ai/sdk",
    normalizeManualTarget: (mod) => _pickAnthropicNamespace(mod),
  },
  // NOTE: No registry entry for Cohere. @traceloop/instrumentation-cohere only
  // instruments the legacy v1 CohereClient — it does not support the current
  // CohereClientV2. The enforcer's manual-telemetry path covers Cohere v2 by
  // reading response.usage directly.
  // NOTE: No registry entry for AWS Bedrock. @traceloop/instrumentation-bedrock
  // creates spans for the Converse API but does not populate gen_ai.usage.*
  // token attributes, so the enforcer's manual-telemetry path covers Bedrock
  // by reading response.usage directly.
  // NOTE: No registry entry for Google. @traceloop/instrumentation-google-generativeai
  // does now instrument @google/genai (class GenAIInstrumentation), but only
  // versions ">=1.0.0 <2.0.0" — it does not support the current @google/genai 2.x.
  // The enforcer's manual-telemetry path covers @google/genai across all
  // versions by reading response.usageMetadata directly.
  // NOTE: No registry entry for LangChain — @traceloop/instrumentation-langchain
  // takes `{ callbackManagerModule }` (not a single SDK module), and its init()
  // is a no-op so enable() does NOT set up auto require-in-the-middle hooks.
  // Handled separately by _applyLangChainInstrumentation below.
];

// ── InstrumentModules interface ───────────────────────────────────

/**
 * Modules to instrument for token usage extraction.
 *
 * Pass the imported module reference directly to solve ESM import ordering.
 * This works regardless of whether the user imports before or after init().
 *
 * @example
 * ```typescript
 * import OpenAI from "openai";
 * import * as tp from "token-police";
 *
 * tp.init({
 * apiKey: "tp_sk_...",
 * instrumentModules: { openAI: OpenAI },
 * });
 * ```
 */
export interface InstrumentModules {
  /**
   * The OpenAI module from 'openai'. EITHER form is accepted — the default
   * export (the `OpenAI` class, `import OpenAI from "openai"`) or the module
   * namespace (`import * as OpenAI from "openai"`). The SDK normalizes to
   * whichever shape the OpenLLMetry instrumentor needs (it reads
   * `Chat.Completions.prototype`, which lives on the class).
   */
  openAI?: any;
  /**
   * The Anthropic module from '@anthropic-ai/sdk'. EITHER form is accepted —
   * the namespace import (`import * as Anthropic from "@anthropic-ai/sdk"`) or
   * the default export (the `Anthropic` class). The SDK normalizes to whichever
   * shape the OpenLLMetry instrumentor needs (it reads
   * `module.Anthropic.Messages.prototype`, which only the namespace carries).
   *
   * The namespace import is the RECOMMENDED form: on modern @anthropic-ai/sdk
   * releases it also carries the root `APIPromise` export, which is what keeps
   * the instrumentor's streaming path usable. Without that export the SDK must
   * route STREAMED calls around the instrumentor layer, and what survives
   * differs by surface:
   * - `messages.create` / `beta.messages.create` — still fully covered: the
   *   enforcer has its own wrapper on both (pre-flight check + manual
   *   telemetry), so streamed calls stay metered, just not via a Traceloop span.
   * - `completions.create` — the legacy Text Completions surface has NO
   *   enforcer wrapper at all, so a streamed call there is metered by NOTHING:
   *   no pre-flight check and no usage row. It runs, and it runs invisibly.
   */
  anthropic?: any;
  /**
   * The Cohere module (namespace import of 'cohere-ai').
   * @traceloop/instrumentation-cohere only supports the legacy v1 CohereClient,
   * so Cohere v2 token usage is extracted directly by the enforcer's
   * manual-telemetry path (it reads response.usage). Passing this module lets
   * the enforcer patch `CohereClientV2.prototype.chat` / `.chatStream`.
   */
  cohere?: any;
  /**
   * The AWS Bedrock module (namespace import of '@aws-sdk/client-bedrock-runtime').
   * @traceloop/instrumentation-bedrock does not emit token usage for the
   * Converse API, so usage is extracted directly by the enforcer's
   * manual-telemetry path (it reads response.usage). Passing this module lets
   * the enforcer patch `BedrockRuntimeClient.prototype.send`.
   */
  bedrock?: any;
  /**
   * The Google GenAI module (namespace import of '@google/genai').
   * Token usage is extracted directly by the enforcer's manual-telemetry path
   * (it reads response.usageMetadata) — passing this module lets the enforcer
   * patch it. This works across all @google/genai versions; the Traceloop
   * instrumentor only supports the 1.x line.
   */
  googleGenAI?: any;
  /**
   * The native OpenRouter SDK module (namespace import of '@openrouter/sdk').
   * There is no OpenLLMetry instrumentor for it, so token usage is extracted
   * directly by the enforcer's manual-telemetry path (it reads
   * response.usage). Passing this module lets the enforcer patch
   * `Chat.prototype.send`.
   */
  openRouter?: any;
  /**
   * The native Cerebras SDK module (namespace import of
   * '@cerebras/cerebras_cloud_sdk'). There is no OpenLLMetry instrumentor for
   * it, so token usage is extracted directly by the enforcer's
   * manual-telemetry path (it reads response.usage). Passing this module lets
   * the enforcer patch `Cerebras.Chat.Completions.prototype.create`.
   */
  cerebras?: any;
  /**
   * The native Together.ai SDK module (default + namespace import of
   * 'together-ai'). There is no OpenLLMetry-JS instrumentor for Together, so
   * token usage is extracted directly by the enforcer's manual-telemetry path
   * (it reads response.usage on non-stream calls and the final usage-bearing
   * chunk on streams). Passing this module lets the enforcer patch
   * `Together.Chat.Completions.prototype.create`.
   */
  together?: any;
  /**
   * The native Groq SDK module ('groq-sdk') — the default export (the `Groq`
   * class) or a namespace import both work; the enforcer normalizes either via
   * `_pickClassExport`. Groq ships its own Stainless-generated package — it is
   * NOT the `openai` package, so no OpenLLMetry-JS instrumentor patches it and
   * token usage is extracted directly by the enforcer's manual-telemetry path
   * (non-stream `.usage` is OpenAI-shaped; on streams the usage rides the final
   * chunk under `chunk.x_groq.usage`). Passing this module lets the enforcer
   * patch `Groq.Chat.Completions.prototype.create`.
   */
  groq?: any;
  /**
   * The Mistral SDK module (namespace import of '@mistralai/mistralai' — its
   * `Chat` class is not root-exported, so the namespace is required). There is
   * no OpenLLMetry-JS instrumentor for it, so token usage is extracted directly
   * by the enforcer's manual-telemetry path. The enforcer constructs a
   * throw-away `Mistral` instance to reach the prototypes and patches
   * `Chat.prototype.complete` / `.stream` (plus embeddings, OCR and
   * transcriptions) — see _instrumentMistral.
   */
  mistral?: any;
  /**
   * The Voyage AI SDK module (namespace import of 'voyageai').
   * No OpenLLMetry instrumentor exists for Voyage, so token usage is
   * extracted directly by the enforcer's manual-telemetry path with
   * operation="embedding". Passing this module lets the enforcer patch
   * `VoyageAIClient.prototype.embed` / `.multimodalEmbed`, and it is patched
   * BEFORE the auto-discovery copies so the record your app actually calls is
   * the one that gets wrapped.
   */
  voyageai?: any;
  /**
   * LangChain core modules. @langchain/core ships dual builds (CJS + ESM)
   * as distinct module records — consumers like @langchain/openai
   * (ESM) import the ESM versions, which are different objects from what
   * `require()` returns. Pass the ESM modules here for guaranteed first-call
   * coverage; the SDK also patches whatever it can resolve via require() and
   * async import() as fallbacks.
   *
   * ```ts
   * import * as lcChatModels from '@langchain/core/language_models/chat_models';
   * import * as lcCallbacks from '@langchain/core/callbacks/manager';
   *
   * tp.init({
   * instrumentModules: {
   * langChain: {
   * chatModelsModule: lcChatModels, // enforcer patches BaseChatModel.{generate,stream}
   * callbackManagerModule: lcCallbacks, // telemetry patches CallbackManager
   * },
   * },
   * });
   * ```
   */
  langChain?:
    | {
        /** Namespace import of @langchain/core/language_models/chat_models. */
        chatModelsModule?: any;
        /** Namespace import of @langchain/core/callbacks/manager. */
        callbackManagerModule?: any;
      }
    | any;
  /**
   * The HuggingFace SDK module (namespace import of '@huggingface/inference').
   * There is no OpenLLMetry instrumentor for it, so token usage is extracted
   * directly by the enforcer's manual-telemetry path (it reads response.usage —
   * the SDK is OpenAI-compatible). `InferenceClient.chatCompletion` /
   * `.chatCompletionStream` are non-configurable per-instance fields, so the
   * enforcer patches the package's `tasks/nlp/chatCompletion(Stream)` source
   * submodules via the require cache. Must be passed (or the SDK installed)
   * before the app constructs its InferenceClient.
   */
  huggingFace?: any;
  /**
   * LlamaIndex provider modules. Each provider lives in its own subpackage
   * (@llamaindex/openai, @llamaindex/anthropic, @llamaindex/google) — pass
   * the namespace import for each provider you use. The enforcer patches
   * `chat` / `complete` on each provider class prototype, runs the single
   * pre-flight check + captures composition from LlamaIndex's ChatMessage
   * objects, and holds the inLlamaIndex guard so the inner provider wrapper
   * stays inert. Telemetry flows through the inner provider's OpenLLMetry
   * span (or, for Gemini, the manual-telemetry path on @google/genai).
   *
   * ```ts
   * import * as liOpenAI from '@llamaindex/openai';
   * import * as liAnthropic from '@llamaindex/anthropic';
   * import * as liGoogle from '@llamaindex/google';
   *
   * tp.init({
   * instrumentModules: {
   * llamaIndex: {
   * openaiModule: liOpenAI,
   * anthropicModule: liAnthropic,
   * geminiModule: liGoogle,
   * },
   * },
   * });
   * ```
   */
  llamaIndex?: {
    /** Namespace import of @llamaindex/openai. */
    openaiModule?: any;
    /** Namespace import of @llamaindex/anthropic. */
    anthropicModule?: any;
    /** Namespace import of @llamaindex/google. */
    geminiModule?: any;
  };
  /**
   * The Vercel AI SDK xAI provider module (namespace import of '@ai-sdk/xai').
   * `createXai({apiKey})("grok-...")` returns a `LanguageModelV2` whose
   * concrete class isn't root-exported; the enforcer constructs a probe
   * instance to reach the prototype and patches `doGenerate` / `doStream` on
   * the manual-telemetry path (there is no OpenLLMetry-JS instrumentor for
   * the AI SDK). Pass this before constructing your provider so the patches
   * are in place at first call.
   *
   * Superseded by `aiSdkProviders` (which accepts any AI SDK provider, xAI
   * included) — kept working for existing integrations.
   */
  xai?: any;
  /**
   * Vercel AI SDK (`ai` package) providers — generic. Known first-party
   * packages (@ai-sdk/openai, @ai-sdk/anthropic, @ai-sdk/google,
   * @ai-sdk/mistral, @ai-sdk/groq, @ai-sdk/xai, @ai-sdk/gateway,
   * @ai-sdk/openai-compatible, plus the `ai` package's bundled gateway) are
   * auto-discovered without this key. Pass entries here for providers the
   * SDK's own require() cannot reach — most notably COMMUNITY providers
   * (e.g. 'vercel-minimax-ai-provider'), which bundle their own nested
   * @ai-sdk/* copies whose LanguageModel classes are different objects from
   * the top-level packages.
   *
   * Each entry may be a provider factory/singleton, a constructed model
   * instance (any kind — language, embedding, image, speech, transcription,
   * video), or a package namespace (its `create*` exports are probed). All
   * modality entry points (`languageModel`, `embeddingModel`, `imageModel`,
   * `speechModel`, `transcriptionModel`, `videoModel`) are instrumented:
   *
   * ```ts
   * import { createMinimax } from 'vercel-minimax-ai-provider';
   * const minimax = createMinimax({ apiKey: process.env.MINIMAX_API_KEY! });
   *
   * tp.init({
   * apiKey: 'tp_sk_...',
   * firewall: 'enforce',
   * instrumentModules: { aiSdkProviders: [minimax] },
   * });
   * ```
   *
   * Alternative for bundled/edge runtimes: wrap individual models with
   * `tokenPoliceAiSdkMiddleware()` via the AI SDK's `wrapLanguageModel`.
   */
  aiSdkProviders?: any[];
}

// ── Context manager ───────────────────────────────────────────────

/**
 * Minimal AsyncLocalStorage-backed OTel ContextManager.
 *
 * The SDK uses a `BasicTracerProvider` and sets it as the global provider
 * itself, which installs NO async context manager (only `NodeTracerProvider`'s
 * `register()` does). Without one, the
 * global active context is the NoopContextManager and `startActiveSpan()`
 * cannot make our agent span active — so OpenLLMetry LLM spans never nest
 * under it. We register this manager (only if none exists) so context
 * propagation works without pulling in `@opentelemetry/context-async-hooks`.
 */
class TPContextManager {
  private _als = new AsyncLocalStorage<Context>();

  active(): Context {
    return this._als.getStore() ?? ROOT_CONTEXT;
  }

  with<A extends unknown[], F extends (...args: A) => ReturnType<F>>(
    ctx: Context,
    fn: F,
    thisArg?: ThisParameterType<F>,
    ...args: A
  ): ReturnType<F> {
    const cb = thisArg == null ? fn : (fn.bind(thisArg) as F);
    return this._als.run(ctx, cb as (...a: A) => ReturnType<F>, ...args);
  }

  bind<T>(ctx: Context, target: T): T {
    if (typeof target === "function") {
      const self = this;
      const bound = function (this: unknown, ...args: unknown[]) {
        return self.with(ctx, () =>
          (target as (...a: unknown[]) => unknown).apply(this, args),
        );
      };
      return bound as unknown as T;
    }
    return target;
  }

  enable(): this {
    return this;
  }

  disable(): this {
    this._als.disable();
    return this;
  }
}

// ── Orphan-rewrite: surviving-ancestor resolution ──────────────────────────
// LangChain/LangGraph emit a stack of intermediate orchestration spans between
// our structural agent/chain anchor and the real LLM/tool call ("invoke_agent
// X", "execute_task RunnableSequence", "LangGraph.workflow", …). None carry a
// model, so they are dropped and never stored — a kept LLM/tool child
// whose parent_span_id points at one would be orphaned (flat, cost not rolled
// up). We record EVERY non-structural span's (native span_id -> parent span_id)
// in onStart, then at emission hop from the child's immediate parent up through
// these intermediates to the nearest span NOT in the map. The structural anchor
// returns out of onStart BEFORE recording, so it is never a key — making it the
// natural terminator (the walk stops exactly at the agent/chain root, or any
// other kept ancestor). For a correctly-linked call the parent is already the
// agent (absent from the map) so the walk is a no-op and the result is
// byte-identical to before. Fixed cap + FIFO eviction => constant memory; an
// evicted entry only degrades that one lookup to the root fallback, never an
// error. Single-threaded (event loop) so no lock is needed.
const _SPAN_PARENT_CAP = 4096;
const _spanParent = new Map<string, string>();

// OTel JS SDK 1.x exposed the ended span's parent id as `ReadableSpan.parentSpanId`
// (a 16-char hex string). SDK 2.x REMOVED that field and moved the parent id under
// `parentSpanContext?: SpanContext`, so it now lives at `parentSpanContext.spanId`
// (also a 16-char hex string — identical shape, no conversion needed). Reading only
// the 1.x field returns `undefined` on 2.x, silently orphaning every span. Read
// whichever the running SDK exposes; tolerate absence and never throw.
function _parentIdOf(span: unknown): string | undefined {
  try {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const s = span as any;
    return s?.parentSpanId ?? s?.parentSpanContext?.spanId ?? undefined;
  } catch {
    return undefined;
  }
}

/**
 * Resolves the `plan_source` to report for a row rebuilt from span attributes.
 *
 * `onStart` stamps `tp.plan_source` next to `tp.paid_plan`, so a span produced
 * by this SDK carries both. Rules:
 *   - a valid stamped source ("app" / "default") is forwarded verbatim;
 *   - no `tp.paid_plan` attribute at all → the reader's own `?? "free"` default
 *     is about to fire, which IS an SDK-synthesized plan → "default";
 *   - anything else (plan stamped but source missing/garbled — e.g. a span from
 *     an older SDK build) → `undefined`, so the field is OMITTED on the wire and
 *     the server records it as "unknown" rather than a guess.
 * Pure property reads on an already-dereferenced attrs object; never throws.
 */
function _planSourceFromAttrs(
  attrs: Record<string, unknown>,
): "app" | "default" | undefined {
  const raw = attrs["tp.plan_source"];
  if (raw === "app" || raw === "default") return raw;
  if (attrs["tp.paid_plan"] == null) return "default";
  return undefined;
}

function _recordSpanParent(span: Span | ReadableSpan): void {
  try {
    const sctx = span.spanContext();
    const sid = sctx?.spanId;
    if (!sid || sid === "0000000000000000") return;
    const pid = String(_parentIdOf(span) ?? "");
    _spanParent.set(sid, pid);
    if (_spanParent.size > _SPAN_PARENT_CAP) {
      // Map preserves insertion order — evict oldest.
      const oldest = _spanParent.keys().next().value;
      if (oldest !== undefined) _spanParent.delete(oldest);
    }
  } catch {
    // fail-open — telemetry bookkeeping must never throw
  }
}

function _resolveKeptParent(parentSpanId: string, rootSpanId: string): string {
  // Hop from an immediate parent up through dropped framework ancestors to the
  // nearest surviving span id (a kept ancestor — typically the agent/chain root,
  // never recorded). For a correctly-linked span (parent already kept, absent
  // from the map) this returns parentSpanId unchanged. Bounded (<=64 hops).
  // rootSpanId is empty for unanchored throwaway sessions (onStart stamps ""
  // when session._anchored is false) so this never invents a phantom parent.
  // Kept-ancestor hop logic is unchanged.
  try {
    let p = parentSpanId;
    let hops = 0;
    while (p && _spanParent.has(p) && hops < 64) {
      p = _spanParent.get(p) as string;
      hops += 1;
    }
    return p || rootSpanId;
  } catch {
    return parentSpanId || rootSpanId;
  }
}

// ── SpanProcessor ─────────────────────────────────────────────────

/**
 * Per-span marker: set in `onStart` on the OTel anthropic instrumentor's
 * duplicate `.stream` span (the one that starts inside a manual `.stream()`
 * wrapper's suppression window), consumed in `onEnd` to drop that span so it
 * does not double-log alongside the wrapper's manual row. A Symbol key keeps it
 * off the exported attribute surface. Same span instance flows onStart→onEnd, so
 * the tag survives. Per-call by construction (see anthropicStreamOtelSuppress).
 */
const TP_DROP_ANTHROPIC_STREAM_OTEL = Symbol("tp.dropAnthropicStreamOtelSpan");

/**
 * Per-span marker: set in `onStart` on a provider-instrumentor genai span that
 * starts inside a LlamaIndex wrapper's per-call OTel-suppression window
 * (runWithLlamaIndexOtelSuppress), consumed in `onEnd` to drop that span. The
 * LlamaIndex wrapper meters the call itself (`_logLlamaIndex` manual row /
 * `_emitCallFailureLog` on failure) and short-circuits the inner ENFORCER
 * wrapper via the inLlamaIndex guard — but a Traceloop OTel provider
 * instrumentor wrapping the same underlying SDK still emits its own span,
 * which would double-log the call (run-23 F-23-1: exact 2× billing). A Symbol
 * key keeps it off the exported attribute surface. Same span instance flows
 * onStart→onEnd, so the tag survives. Per-call by construction (see
 * llamaIndexOtelSuppressActive).
 */
const TP_DROP_LLAMAINDEX_INNER_OTEL = Symbol("tp.dropLlamaIndexInnerOtelSpan");

/**
 * Per-span marker: the owning call's observations key, stamped in `onStart`
 * (the span starts inside the enforcer wrapper's per-call obs scope, so the
 * key read there is the call's own) and consumed by the deferred drain in
 * `onEnd` — which runs on a later tick, outside the scope, so it must use
 * this captured value rather than re-read the ALS. Same span instance flows
 * onStart→onEnd in the Node OTel SDK (BasicTracerProvider hands the SAME
 * Span to both hooks), so an on-object tag survives — a DELIBERATE mechanism
 * divergence from the Python SDK, whose Span.end() builds a fresh
 * ReadableSpan for on_end and therefore uses a span-id side map
 * (telemetry.py `_span_obs_keys`) instead. A Symbol key keeps it off the
 * exported attribute surface and out of every /log payload.
 */
const TP_OBS_KEY = Symbol("tp.obsCallKey");

/**
 * Promises for in-flight process.nextTick deferred LLM log callbacks.
 * Module-level so every TokenPoliceSpanProcessor instance (private provider +
 * customer piggyback) shares one drain queue for flush/shutdown.
 */
const _deferredLogs = new Set<Promise<void>>();

/**
 * @internal Drain nextTick-deferred LLM logs so client.log() can register POSTs
 * before flush/close seals the client. Fail-open: never throws.
 */
export async function forceFlushTokenPoliceSpans(
  options?: { greedy?: boolean },
): Promise<void> {
  // Greedy observations window (depth-counted) for terminal flushes only:
  // client flush()/close() and processor shutdown must ship EVERY queued
  // observation (old drain-all behavior) — a keyed drain inside the deferred
  // callbacks below would strand other calls' tagged entries = silent audit
  // loss, the worst regression class. Depth counter (not a boolean): the
  // beforeExit fire-and-forget flush can overlap a user's awaited close()
  // (and two concurrent close() calls both enter before either seals) — a
  // boolean's first finally would clear the flag mid-way through the other
  // call's still-open window. CAVEAT: SpanProcessor.forceFlush passes
  // `{greedy: false}` — it is a PUBLIC per-request OTel API (platform
  // integrations call it on live traffic) and a greedy window there would
  // reopen cross-call stealing against concurrent calls; its orphans ride
  // the staleness fallback instead. Window closed in the finally even on
  // error.
  const greedy = options?.greedy !== false;
  if (greedy) {
    try {
      beginObservationsDrainAll();
    } catch {
      /* fail-open */
    }
  }
  try {
    await Promise.allSettled([..._deferredLogs]);
  } catch {
    /* fail-open */
  } finally {
    if (greedy) {
      try {
        endObservationsDrainAll();
      } catch {
        /* fail-open */
      }
    }
  }
}

/**
 * Custom SpanProcessor that listens to finished spans locally.
 * Extracts gen_ai.usage attributes and logs them to TokenPolice.
 *
 * PRIVACY: This processor runs entirely in-process. No data is exported via
 * OpenTelemetry exporters. Only aggregated token counts are sent to TokenPolice.
 */
export class TokenPoliceSpanProcessor implements SpanProcessor {
  /**
   * Eagerly capture session context before async boundaries can erase it.
   * Called when a span starts — stamps TP session data as span attributes.
   */
  onStart(span: Span, _parentContext?: Context): void {
    // Our own structural anchor span (agent/chain, opened by
    // session()/agent()/chain()/workflow()). It is emitted as an `agent`/`chain`
    // row in onEnd; it must NOT consume a span_order or be treated as an LLM span
    // here. Its tp.* attributes were stamped at creation.
    {
      const k = (span as any).attributes?.["tp.kind"];
      if (k === "agent" || k === "chain") {
        return;
      }
    }
    // ── Tag the Traceloop anthropic `.stream` duplicate span for suppression ──
    // The manual `.stream()` wrapper meters this call itself; the inner OTel
    // anthropic instrumentor (wrapped inner of our patch) emits a duplicate span
    // for the same call. That span STARTS here, inside the wrapper's per-call
    // suppression window (runWithAnthropicStreamOtelSuppress), so we tag exactly
    // it — `onEnd` then drops the tagged span. Async-context scoping makes this
    // per-call: concurrent same-session streams each tag their OWN duplicate, so
    // no call can consume another's suppression (the former session one-shot
    // boolean could — eating the wrong span and double-counting the intended
    // duplicate). Gated on an anthropic signal as defense-in-depth. Fully
    // fail-open: a tagging failure can never crash span start.
    try {
      if (anthropicStreamOtelSuppressActive()) {
        const _a = (span as any).attributes ?? {};
        const _sys = String(
          _a["gen_ai.system"] ?? _a["gen_ai.provider.name"] ?? "",
        ).toLowerCase();
        const _scope = String(
          (span as any).instrumentationScope?.name ??
            (span as any).instrumentationLibrary?.name ??
            "",
        ).toLowerCase();
        if (_sys.includes("anthropic") || _scope.includes("anthropic")) {
          Object.defineProperty(span, TP_DROP_ANTHROPIC_STREAM_OTEL, {
            value: true,
            configurable: true,
            enumerable: false,
          });
        }
      }
    } catch {
      // fail-open: never crash span start on suppression tagging
    }
    // ── Tag the provider-instrumentor duplicate span under LlamaIndex ──
    // The LlamaIndex wrapper is the sole billable emitter for calls made
    // through @llamaindex/* provider classes (manual row; inner ENFORCER
    // wrapper passes through on inLlamaIndex). When a Traceloop provider
    // instrumentor ALSO wraps the underlying SDK, it starts a span for the
    // same physical call — HERE, inside the wrapper's per-call suppression
    // window (runWithLlamaIndexOtelSuppress) — so we tag exactly it for the
    // `onEnd` drop (run-23 F-23-1 double-billing). Async-context scoping
    // makes this per-call: a concurrent same-session DIRECT provider call is
    // outside every window and keeps its span billable. Gated on a genai
    // signal as defense-in-depth — Node registers OTel instrumentors for
    // OpenAI + Anthropic ONLY (see the "Intentional divergence" note in
    // onEnd); all other providers are manual-tap and emit no span. No early
    // return after tagging (mirrors the anthropic block): the span still
    // flows through normal onStart stamping, and drop placement in onEnd
    // protects a mis-tagged tool/structural span. Fully fail-open: a tagging
    // failure can never crash span start.
    try {
      if (llamaIndexOtelSuppressActive()) {
        const _a = (span as any).attributes ?? {};
        const _sys = String(
          _a["gen_ai.system"] ?? _a["gen_ai.provider.name"] ?? "",
        ).toLowerCase();
        const _scope = String(
          (span as any).instrumentationScope?.name ??
            (span as any).instrumentationLibrary?.name ??
            "",
        ).toLowerCase();
        if (
          _sys !== "" ||
          _scope.includes("anthropic") ||
          _scope.includes("openai")
        ) {
          Object.defineProperty(span, TP_DROP_LLAMAINDEX_INNER_OTEL, {
            value: true,
            configurable: true,
            enumerable: false,
          });
        }
      }
    } catch {
      // fail-open: never crash span start on suppression tagging
    }
    // Record this NON-structural span's parent link for orphan-rewrite. The
    // structural anchor returned above, so it is never recorded — that makes it
    // the natural terminator when a kept LLM/tool child hops over dropped
    // LangChain/LangGraph framework ancestors in onEnd. (See _spanParent.)
    _recordSpanParent(span);
    // LangChain's OpenLLMetry instrumentor emits framework spans (chains,
    // agents, tools) alongside the LLM spans. Skip enriching/counting those:
    // only LLM spans are logged in onEnd, and counting framework spans would
    // inflate span_order and let a chain span steal the pending span name set
    // via tp.setSpanName(). Python convention is "<Class>.<kind>"; JS
    // convention is "<kind> <name>" (space-separated). LLM spans use "chat".
    const _spanName = (span as any).name ?? "";
    // Tool execution span → promote to a `tool` row (emitted in onEnd) instead
    // of dropping. Stamp minimal session context; do NOT consume a span_order
    // or steal the pending setSpanName().
    if (isToolSpan((span as any).attributes ?? {}, String(_spanName))) {
      try {
        const session = getCurrentSession();
        span.setAttribute("tp.kind", "tool");
        span.setAttribute("tp.trace_id", session.traceId);
        span.setAttribute("tp.user_id", session.userId);
        span.setAttribute("tp.paid_plan", session.paidPlan);
        span.setAttribute("tp.plan_source", session.planSource);
        span.setAttribute("tp.workflow_name", session.workflowName);
        span.setAttribute("tp.session_id", session.sessionId);
        // Custom session metadata → tp.meta.* attributes so onEnd can read it
        // back into the tool row (mirrors the LLM branch below).
        if (session.metadata) {
          for (const [k, v] of Object.entries(session.metadata)) {
            if (
              typeof v === "string" ||
              typeof v === "number" ||
              typeof v === "boolean"
            ) {
              span.setAttribute(`tp.meta.${k}`, v);
            } else if (v != null) {
              let serialized: string;
              try {
                serialized = JSON.stringify(v);
              } catch {
                serialized = String(v); // circular ref → best-effort
              }
              span.setAttribute(`tp.meta.${k}`, serialized);
            }
          }
        }
      } catch {
        // Fail-open
      }
      return;
    }
    if (typeof _spanName === "string") {
      if (
        // Python convention
        _spanName.endsWith(".workflow") ||
        _spanName.endsWith(".task") ||
        _spanName.endsWith(".tool") ||
        _spanName.endsWith(".agent") ||
        // JS / @traceloop/instrumentation-langchain convention
        _spanName.startsWith("workflow ") ||
        _spanName.startsWith("task ") ||
        _spanName.startsWith("invoke_agent ") ||
        _spanName.startsWith("execute_tool ") ||
        _spanName.startsWith("chain ")
      ) {
        return;
      }
    }
    // ── Per-call observations key ──
    // This LLM span starts inside its enforcer wrapper's obs scope (the
    // wrapper opens the scope before dispatching the provider call), so the
    // ALS read here yields the call's own key. Captured as a NON-EXPORTED
    // Symbol marker (never a span attribute → never in a payload) for the
    // deferred drain in onEnd, which runs outside the scope. Fully fail-open.
    try {
      const _obsKey = getCurrentObsKey();
      if (_obsKey) {
        Object.defineProperty(span, TP_OBS_KEY, {
          value: _obsKey,
          configurable: true,
          enumerable: false,
        });
      }
    } catch {
      // fail-open: an unmarked span drains as an unknown claimant (null)
    }
    try {
      const session = getCurrentSession();
      if (session.workflowName) {
        span.setAttribute("tp.workflow_name", session.workflowName);
      }
      span.setAttribute("tp.user_id", session.userId);
      span.setAttribute("tp.paid_plan", session.paidPlan);
      span.setAttribute("tp.plan_source", session.planSource);
      span.setAttribute("tp.session_id", session.sessionId);

      // Span hierarchy. Only stamp root when the session was anchored by a
      // real structural agent/chain span; otherwise "" so onEnd never parents
      // onto a throwaway rootSpanId that was never logged as an agent row.
      span.setAttribute("tp.trace_id", session.traceId);
      let _root = "";
      try {
        _root = session._anchored ? session.rootSpanId : "";
      } catch {
        _root = "";
      }
      span.setAttribute("tp.root_span_id", _root);
      // The instrumented wrapper reserves this span's order before the provider
      // call — so concurrent same-session calls can't cross-attribute their
      // pre/post-call composition stashes — and this onStart consumes it. When
      // no reservation is present (e.g. a span opened outside any wrapped call),
      // fall back to allocating a fresh order exactly as before.
      let _spanOrder: number;
      try {
        const _reserved = consumeReservedSpanOrder();
        _spanOrder = _reserved != null ? _reserved : session.nextSpanOrder();
      } catch {
        _spanOrder = session.nextSpanOrder();
      }
      span.setAttribute("tp.span_order", _spanOrder);

      // Consume pending span name (set by user via tp.setSpanName())
      const pendingName = consumePendingSpanName();
      if (pendingName) {
        span.setAttribute("tp.span_name", pendingName);
      }

      if (session.metadata) {
        for (const [k, v] of Object.entries(session.metadata)) {
          if (
            typeof v === "string" ||
            typeof v === "number" ||
            typeof v === "boolean"
          ) {
            span.setAttribute(`tp.meta.${k}`, v);
          } else if (v != null) {
            // Object-valued metadata (e.g. `_tp_routing` reroute
            // provenance) must be JSON-serialized — `String(obj)` yields the
            // useless literal "[object Object]" and loses from→to/ruleId.
            let serialized: string;
            try {
              serialized = JSON.stringify(v);
            } catch {
              serialized = String(v); // circular ref → best-effort
            }
            span.setAttribute(`tp.meta.${k}`, serialized);
          }
        }
      }
    } catch {
      // Fail-open: never crash on span start
    }
  }

  /**
   * Called when a span ends. Extracts token usage from gen_ai.* attributes
   * and sends them to the TokenPolice API as a fire-and-forget log.
   */
  onEnd(span: ReadableSpan): void {
    try {
      const attrs = span.attributes || {};
      // Lazy: the attribute-key enumeration + join only runs when debug is on.
      logger.debug(() => `Span ended: ${span.name} (attributes: ${Object.keys(attrs).join(", ")})`);

      // ── Structural anchor span → emit a zero-token `agent`/`chain` row ──
      // These anchor the trace tree (workflows, chains, and nested sub-agents).
      // They carry native W3C trace/span ids and parent onto the enclosing span.
      if (attrs["tp.kind"] === "agent" || attrs["tp.kind"] === "chain") {
        this._logStructuralSpan(span, attrs, String(attrs["tp.kind"]));
        return;
      }

      // ── Tool execution span → emit a zero-usage `tool` row ──
      // tp.kind is set in onStart for the common frameworks; the discriminator
      // is re-checked here so spans whose tool attributes only appear at end
      // (e.g. OpenInference) still emit.
      if (attrs["tp.kind"] === "tool" || isToolSpan(attrs, String(span.name ?? ""))) {
        this._logToolSpan(span, attrs, String(span.name ?? ""));
        return;
      }

      // ── Drop instrumentor-emitted embedding spans ──
      // opentelemetry-instrumentation-openai / -cohere patch embeddings too and
      // emit a span tagged llm.request.type="embedding". TokenPolice captures
      // embeddings authoritatively via its own manual wrapper (operation=embedding,
      // *_embeddings/*_embed shape) which creates NO OTel span — so any embedding
      // span here is the instrumentor's duplicate, which would otherwise be
      // mis-logged as a *_chat row.
      if (isInstrumentorEmbeddingSpan(attrs, String(span.name ?? ""))) {
        return;
      }

      // ── Drop the Traceloop Anthropic `.stream` duplicate OTel span ──
      // `client.messages.stream()` is instrumented manually by the enforcer
      // (pre-flight + one `/log` from finalMessage()). On newer @anthropic-ai/sdk
      // versions the Traceloop anthropic instrumentor ALSO patches `.stream` and
      // emits a duplicate span for the same call that would double-log. That span
      // was tagged in `onStart` — but ONLY while THIS call's manual-stream
      // suppression window was active (runWithAnthropicStreamOtelSuppress), so
      // the tag is per-call. Drop exactly the tagged span. This replaced a
      // session-wide one-shot boolean whose FIRST-anthropic-span-wins consume
      // mis-fired under concurrent same-session streams (ate the wrong span → its
      // row lost, the intended duplicate surviving → double-count). Parity with
      // the Python SDK's intent (Python drops in on_start; Node tags in
      // on_start, drops in on_end — accepted parity-of-intent divergence).
      // Reading a Symbol prop can't throw; the whole body is inside the outer
      // try/catch regardless.
      if ((span as any)[TP_DROP_ANTHROPIC_STREAM_OTEL]) {
        return; // drop the duplicate stream span (per-call tagged in onStart)
      }

      // ── Drop the provider-instrumentor duplicate span under LlamaIndex ──
      // The LlamaIndex wrapper meters the call itself (`_logLlamaIndex`
      // manual row / `_emitCallFailureLog` on failure); the inner provider-
      // instrumentor span tagged during that call's suppression window
      // (runWithLlamaIndexOtelSuppress) is a duplicate that would double-bill
      // the customer (run-23 F-23-1: exact 2× cost — twin rows with identical
      // parent/model/tokens). Per-call ALS scoping keeps a concurrent
      // same-session DIRECT provider call's span billable (a session-scoped
      // flag would eat it → under-billing). This drop deliberately sits AFTER
      // the structural-anchor emit, tool-span emit, and instrumentor-embedding
      // drop (same placement rationale as the anthropic `.stream` drop above):
      // a mis-tagged non-llm span still emits its row. Reading a Symbol prop
      // can't throw; the whole body is inside the outer try/catch regardless.
      if ((span as any)[TP_DROP_LLAMAINDEX_INNER_OTEL]) {
        return; // drop the duplicate inner span (per-call tagged in onStart)
      }

      // ── Intentional divergence from the Python SDK's on_end drop set ──
      // The Python SDK drops several extra
      // double-log classes that Node deliberately does NOT port because they
      // are unreachable here — porting them 1:1 would be dead code:
      // • scope-name drops (`mistralai_sdk_tracer`, `xai_sdk*`,
      // `opentelemetry.instrumentation.google_genai` under LangChain), and
      // • framework-context drops (`in_pydantic_ai()`, `in_litellm()`).
      // Node registers OTel instrumentors for OpenAI + Anthropic ONLY (see the
      // INSTRUMENTOR_REGISTRY NOTEs above); Mistral/xAI/Google/etc. are
      // manual-tap and emit NO OTel span, so there is nothing to double-log.
      // pydantic_ai has no JS runtime (Python-only), so it is never
      // instrumented here. LiteLLM is NOT Python-only — a JS port (npm
      // `litellm` = litellmjs) exists, and the dominant LiteLLM-on-Node
      // topology (openai JS client → a LiteLLM proxy `base_url`) is already
      // metered by the OpenAI instrumentor; the manual `provider:"litellm"`
      // path in enforcer.ts covers raw-HTTP tp.protect wrappers. In all of
      // these there is no PyPI/OTel LiteLLM instrumentor emitting a duplicate
      // span, so no on_end drop is needed. (The native litellmjs library
      // called directly is an un-auto-instrumented metering gap, not a
      // double-log concern.) The one Python drop with a reachable Node
      // trigger —
      // instrumentor-embedding spans — IS handled by the guard directly above.
      // The LangChain-JS ChatOpenAI/ChatAnthropic path is
      // verified single-row (no inner-instrumentor duplicate) in cross-framework
      // regression runs, so no drop
      // is needed there either. Do NOT "restore parity" by porting these guards:
      // Node auto-instruments OpenAI and Anthropic only — every other provider
      // is manual-tap and emits no OTel span, so there is no duplicate to drop
      // for those providers, and adding drops for spans that cannot exist here
      // would be dead code.

      // Support both new gen_ai and legacy llm attributes
      // `let` (not `const`): for streaming spans whose instrumentor emits no
      // usage, the enforcer's stream tap stashes token usage in
      // `_pendingCompositions[compKey].usage`, picked up inside the deferred
      // nextTick below (the stash only becomes available after the customer
      // drains the stream — i.e. after this synchronous onEnd returns).
      let inputTokens = Number(attrs["gen_ai.usage.input_tokens"] ?? attrs["llm.usage.prompt_tokens"] ?? 0);
      let outputTokens = Number(attrs["gen_ai.usage.output_tokens"] ?? attrs["llm.usage.completion_tokens"] ?? 0);

      // Check for cached tokens (Standard OpenLLMetry + Provider Specific).
      // OpenLLMetry's Anthropic instrumentor emits dotted keys
      // (gen_ai.usage.cache_read.input_tokens); older versions used
      // anthropic.usage.cache_*_input_tokens.
      // Cache READ hits only. Cache *creation* (write) tokens bill at a different
      // (higher) rate and are forwarded separately as cache_creation_input_tokens
      // below — they must not be folded in here, or downstream cost calculation
      // would count creation tokens twice (once as read, once as write).
      let cachedTokens = Number(attrs["gen_ai.usage.cache_read_tokens"] ?? 0);
      if (cachedTokens === 0) {
        cachedTokens = Number(
          attrs["gen_ai.usage.cache_read.input_tokens"] ??
            attrs["anthropic.usage.cache_read_input_tokens"] ??
            0,
        );
      }

      // Only log if there's actual token usage and it's an LLM span.
      // We are more liberal here to ensure we don't miss spans
      const isLlmSpan = 
        "gen_ai.system" in attrs ||
        "gen_ai.provider.name" in attrs ||
        "llm.system" in attrs ||
        "gen_ai.operation.name" in attrs ||
        "telemetry.sdk.name" in attrs ||
        span.name.includes("chat");

      // Enter for any LLM span. The usage>0 decision is deferred to the nextTick
      // below so a streaming span's tapped usage (stashed after the stream
      // drains — see telemetry note on inputTokens above) can be applied first.
      // Usage-less spans early-return inside the nextTick, preserving the old
      // "only log spans with real token usage" behavior.
      if (isLlmSpan) {
        const tp = getClient();
        if (!tp) {
          logger.debug("No TokenPolice client found, skipping log.");
          return;
        }
        logger.debug(`Logging span ${span.name} with ${inputTokens} input and ${outputTokens} output tokens.`);

        // Extract eagerly captured TP attributes from span
        const userId = String(attrs["tp.user_id"] ?? "anonymous");
        const paidPlan = String(attrs["tp.paid_plan"] ?? "free");
        const planSource = _planSourceFromAttrs(attrs);
        const workflowName = String(attrs["tp.workflow_name"] ?? "default");
        const sessionId = String(attrs["tp.session_id"] ?? "");
        let model = String(attrs["gen_ai.request.model"] ?? "unknown");
        // Gemini via LangChain reports "models/gemini-2.5-flash" — strip the
        // resource prefix so price lookup matches the canonical model id.
        if (model.startsWith("models/")) model = model.slice("models/".length);
        // LangChain's instrumentor reports gen_ai.system with mixed case
        // ("openai" / "Anthropic" / "Google"); providers are normalized to
        // lowercase before being reported (canonical provider slugs)
        let provider = String(
          attrs["gen_ai.system"] ?? attrs["gen_ai.provider.name"] ?? "",
        );
        const _pl = provider.toLowerCase();
        if (_pl.includes("gemini") || _pl.includes("google")) provider = "google";
        else if (_pl.includes("bedrock") || _pl === "aws") provider = "bedrock";
        else if (_pl.includes("mistral")) provider = "mistral";
        else if (_pl.includes("together")) provider = "together";
        else provider = _pl;

        // Span hierarchy
        const traceId = String(attrs["tp.trace_id"] ?? "");
        const rootSpanId = String(attrs["tp.root_span_id"] ?? "");
        const spanOrder = Number(attrs["tp.span_order"] ?? 0);
        // Span name defaults to the (post-override) model below; for now,
        // record any user-set name. We finalize the default after the
        // compData lookup so model overrides are reflected.
        const userSpanName = attrs["tp.span_name"];

        const startMs = span.startTime[0] * 1000 + span.startTime[1] / 1000000;
        const endMs = span.endTime[0] * 1000 + span.endTime[1] / 1000000;

        const startTimeISO = new Date(startMs).toISOString();
        const endTimeISO = new Date(endMs).toISOString();

        // Reconstruct metadata from span attributes (eagerly — no session needed)
        const metadata: Record<string, unknown> = { workflow_name: workflowName };
        if (sessionId) metadata.session_id = sessionId;
        for (const [k, v] of Object.entries(attrs)) {
          // B4: the reroute marker was snapshotted off session metadata at
          // span START, so it rides EVERY span opened after ANY reroute — not
          // just the rerouted call's. Strip it here; the deferred block below
          // re-adds it from the per-call store when this span's own obs key
          // says this row belongs to the call that was rerouted.
          if (k === TP_ROUTING_ATTR) continue;
          if (k.startsWith("tp.meta.")) {
            metadata[k.slice(8)] = v;
          }
        }

        // ── Deferred log: use process.nextTick to let the enforcer's ──
        // ── post-hook capture response composition before we send. ──
        // The enforcer wraps AROUND OpenLLMetry, so its post-hook runs
        // after on_end returns. By deferring to nextTick, we read the
        // composition map after the enforcer has populated it.
        // Track the deferred Promise so flush/shutdown can await it
        // (otherwise the last LLM span races seal and is silently dropped).
        let resolveDeferred!: () => void;
        const deferred = new Promise<void>((r) => {
          resolveDeferred = r;
        });
        _deferredLogs.add(deferred);
        process.nextTick(() => {
          // Deferred Promise settles in `finally` when this callback finishes.
          // Do not fire-and-forget async work here: if the callback returns before
          // tp.log() runs, forceFlush resolves early and close() can seal first —
          // reopening the last-span drop. (A true async callback with await is fine
          // only if finally still wraps the whole body after log.)
          try {
            // Extract composition from session (deferred — response should be captured now)
            let promptComp: unknown[] = [];
            let responseComp: unknown[] = [];
            // The enforcer stashes the real provider for SDK calls whose
            // gen_ai.system is misleading (e.g. the OpenAI SDK pointed at
            // OpenRouter reports gen_ai.system="openai").
            let effProvider = provider;
            // The enforcer stashes the model name when the instrumentor span
            // doesn't expose it (e.g. ChatGoogleGenerativeAI in JS).
            let effectiveModel = model;
            // Serving endpoint the enforcer stashed — forwarded so the deployer
            // can be identified (host → provider) on the
            // auto-instrumented path too (not just the manual path).
            let apiBase = "";
            // Vendor head of a gateway-routed model slug ("openai/gpt-4.1-nano"
            // -> "openai") the enforcer stashed on the detected-gateway path —
            // forwarded as `model_extras.original_provider` when non-empty so the
            // deployer can be priced/attributed. Independent of apiBase.
            let gwOriginalProvider = "";
            // Provider-reported service tier ("batch"/"flex"/"priority") the
            // enforcer post-hook / stream tap stashed — forwarded as usage.tier
            // so tier-specific pricing applies.
            let serviceTier = "";
            // Client-side latency primitives (is_streaming/ttft) the enforcer's
            // stream tap stashed — forwarded so streamed instrumented calls
            // carry TTFT/throughput like the manual paths do.
            let latencyData: {
              is_streaming: boolean;
              ttft_ms: number | null;
              total_ms: number;
              generation_ms: number | null;
              output_tokens: number | null;
              clock: string;
            } | null = null;
            // Set only when the OpenAI-SDK stream-stash fallback (below) supplies
            // the token numbers because the span attrs carried none. Gates the
            // cache-inclusive adjustment of raw.prompt_tokens to the N1 stash
            // path, leaving the attr path bit-identical. Never set for
            // usage_source:"langchain_message" (LC input may be cache-inclusive).
            let usedStashNumbers = false;
            // The provider's verbatim chunk usage the enforcer stashed
            // (cache-INCLUSIVE prompt_tokens + prompt_tokens_details) — emitted
            // as raw when present so the OpenAI-family mappers don't double-net.
            let stashRawUsage: any = undefined;
            // The provider's verbatim NON-streaming usage the enforcer stashed
            // (openai-wire only) — carries the token detail objects (reasoning,
            // cached) the bare Traceloop attrs drop. Distinct from the stream
            // tap's raw above: here the span attrs DO carry token numbers, so
            // it is applied only after an exact-count consistency check below.
            let nonStreamVerbatimRaw: any = undefined;
            // Cache-write tokens from LangChain message stash (attrs often omit).
            let stashCacheCreationTokens = 0;
            let usedLangchainMessageStash = false;
            // G3-O1: LC-wrapper reasoning stash — fallback source for the
            // always-zero gen_ai.usage.reasoning_tokens attr on LC spans.
            let stashReasoningTokens = 0;
            try {
              const session = getCurrentSession();
              const compKey = `${traceId}:${spanOrder}`;
              // A tap-detected stream failure already emitted the llm failure
              // row for this call — the span row (which may carry partial
              // pre-failure usage in its attrs and would be stamped success)
              // must not land as a second, mislabeled row. delete() doubles
              // as marker cleanup, so the Set never outgrows in-flight failed
              // streams; the pending entry is also dropped (idempotent if
              // _emitCallFailureLog already consumed it) so nothing leaks.
              // Must run BEFORE the compData merge and the zero-usage gate.
              // (Enforcer-owned dynamic field — not on TPSession's typed surface.)
              const failedStreamKeys = (session as any)._failedStreamCompKeys;
              if (failedStreamKeys instanceof Set && failedStreamKeys.delete(compKey)) {
                delete session._pendingCompositions[compKey];
                return;
              }
              const compData = session._pendingCompositions[compKey];
              if (compData) {
                promptComp = compData.prompt ?? [];
                responseComp = compData.response ?? [];
                if (compData.provider) effProvider = compData.provider;
                // Model override (e.g. LangChain JS instrumentor omits
                // gen_ai.request.model for ChatGoogleGenerativeAI).
                if (compData.model) {
                  effectiveModel = compData.model;
                }
                if (compData.api_base) apiBase = compData.api_base;
                // Gateway-routed call: forward the stashed vendor head below in
                // model_extras. Only the gateway stash sets this key; every
                // non-gateway span leaves it unset.
                if (compData.original_provider) {
                  gwOriginalProvider = compData.original_provider;
                }
                if (compData.service_tier) serviceTier = compData.service_tier;
                if (compData.latency) latencyData = compData.latency;
                // Pulled UNCONDITIONALLY (not inside the token-attr branch
                // below — non-streaming attrs DO carry tokens); the entry is
                // deleted after this read, so widen the copy-out here.
                if (compData.nonStreamVerbatimRawUsage) {
                  nonStreamVerbatimRaw = compData.nonStreamVerbatimRawUsage;
                }
                // Slot-level LC reasoning stash (independent of compData.usage
                // — the generate path writes it without a usage stash).
                if (compData.reasoning_tokens) {
                  stashReasoningTokens = Number(compData.reasoning_tokens) || 0;
                }
                // Streaming usage merge:
                // 1) LangChain streamIterator stashes concat'd AIMessage.usage_metadata
                // (usage_source:"langchain_message") — prefer when it repairs
                // under-billing (last-chunk Traceloop llmOutput). Never apply
                // OpenAI N1 add-cached-back for this source.
                // 2) OpenAI-SDK stream tap (no usage_source) — fill only when
                // span usage attrs are both zero; usedStashNumbers → N1 path.
                if (compData.usage) {
                  const sIn = Number(compData.usage.input_tokens ?? 0);
                  const sOut = Number(compData.usage.output_tokens ?? 0);
                  const sCached = Number(compData.usage.cached_tokens ?? 0);
                  const sCacheCreation = Number(
                    compData.usage.cache_creation_tokens ?? 0,
                  );
                  const isLcMessage =
                    compData.usage.usage_source === "langchain_message";
                  if (isLcMessage) {
                    const preferStash =
                      sIn + sOut > inputTokens + outputTokens ||
                      (inputTokens === 0 && sIn > 0) ||
                      (!inputTokens && !outputTokens);
                    if (preferStash) {
                      inputTokens = sIn;
                      outputTokens = sOut;
                      cachedTokens = sCached;
                      stashCacheCreationTokens = sCacheCreation;
                      usedLangchainMessageStash = true;
                      // deliberately NOT usedStashNumbers — LC Gemini input is
                      // cache-inclusive; N1 add-cached-back would over-bill.
                    }
                  } else if (!inputTokens && !outputTokens) {
                    inputTokens = sIn;
                    outputTokens = sOut;
                    if (!cachedTokens) cachedTokens = sCached;
                    usedStashNumbers = true;
                    // The stashed positional input_tokens is cache-EXCLUSIVE; carry
                    // the provider's verbatim (inclusive) chunk usage when present
                    // so the usage-block construction below can emit it as raw.
                    if (compData.usage.raw) stashRawUsage = compData.usage.raw;
                  }
                }
                delete session._pendingCompositions[compKey];
              }
            } catch {
              // Composition is best-effort
            }

            // Usage-less LLM span (e.g. a streaming span whose tapped usage never
            // arrived, or an empty/failed completion): skip — preserves the prior
            // gate behavior of only logging spans with real token usage.
            if (!inputTokens && !outputTokens) {
              return;
            }

            // Build composition from span attributes (fallback for ESM/CJS mismatch).
            //
            // Two attribute formats coexist depending on the instrumentor /
            // semantic-convention version:
            // - legacy: gen_ai.{prompt,completion}.N.{content,role}
            // - 1.x: gen_ai.{input,output}.messages = JSON array of
            // { role, parts: [{ type:"text", content }] }
            try {
              const normalizeRole = (raw: any, dflt: string): string => {
                let r = String(raw ?? dflt);
                if (r.toLowerCase() === "unknown") r = dflt;
                return r;
              };

              if (promptComp.length === 0) {
                const entries: unknown[] = [];
                for (const [key, val] of Object.entries(attrs)) {
                  if (key.startsWith("gen_ai.prompt.") && key.endsWith(".content")) {
                    const idx = key.split(".")[2];
                    const role = normalizeRole(attrs[`gen_ai.prompt.${idx}.role`], "user");
                    const content = String(val).trim();
                    const h = createHash("sha1").update(content).digest("hex").slice(0, 16);
                    // Code points (not UTF-16 units) so astral chars match the
                    // primary composition path and the Python SDK.
                    entries.push({ role, type: "text", length: codePointLength(content), hash: h });
                  }
                }
                if (entries.length === 0 && typeof attrs["gen_ai.input.messages"] === "string") {
                  try {
                    const parsed = JSON.parse(attrs["gen_ai.input.messages"] as string);
                    if (Array.isArray(parsed)) {
                      for (const m of parsed) {
                        const role = normalizeRole(m?.role, "user");
                        const parts = Array.isArray(m?.parts) ? m.parts : [];
                        for (const part of parts) {
                          if (part?.type === "text" && typeof part?.content === "string" && part.content) {
                            const content = part.content.trim();
                            const h = createHash("sha1").update(content).digest("hex").slice(0, 16);
                            // Code points (not UTF-16 units) — parity with the
                            // primary path and the Python SDK.
                            entries.push({ role, type: "text", length: codePointLength(content), hash: h });
                          }
                        }
                      }
                    }
                  } catch { /* not JSON */ }
                }
                if (entries.length > 0) promptComp = entries;
              }

              if (responseComp.length === 0) {
                const entries: unknown[] = [];
                for (const [key, val] of Object.entries(attrs)) {
                  if (key.startsWith("gen_ai.completion.") && key.endsWith(".content")) {
                    const idx = key.split(".")[2];
                    const role = normalizeRole(attrs[`gen_ai.completion.${idx}.role`], "assistant");
                    const content = String(val).trim();
                    const h = createHash("sha1").update(content).digest("hex").slice(0, 16);
                    // Code points (not UTF-16 units) so astral chars match the
                    // primary composition path and the Python SDK.
                    entries.push({ role, type: "text", length: codePointLength(content), hash: h });
                  }
                }
                if (entries.length === 0 && typeof attrs["gen_ai.output.messages"] === "string") {
                  try {
                    const parsed = JSON.parse(attrs["gen_ai.output.messages"] as string);
                    if (Array.isArray(parsed)) {
                      for (const m of parsed) {
                        const role = normalizeRole(m?.role, "assistant");
                        const parts = Array.isArray(m?.parts) ? m.parts : [];
                        let textAdded = false;
                        for (const part of parts) {
                          if (part?.type === "text" && typeof part?.content === "string" && part.content) {
                            const content = part.content.trim();
                            const h = createHash("sha1").update(content).digest("hex").slice(0, 16);
                            // Code points (not UTF-16 units) — parity with the
                            // primary path and the Python SDK.
                            entries.push({ role, type: "text", length: codePointLength(content), hash: h });
                            textAdded = true;
                          }
                        }
                        // The newer JS LangChain instrumentor reports
                        // tool-calling assistant turns as `parts: [{type:"text",
                        // content:""}]` with `finish_reason: "tool_call"` —
                        // the tool name + args are NOT in the attribute. Emit
                        // a placeholder so the UI still shows the call.
                        if (!textAdded && m?.finish_reason === "tool_call") {
                          entries.push({ role, type: "tool_call" });
                        }
                      }
                    }
                  } catch { /* not JSON */ }
                }
                if (entries.length > 0) responseComp = entries;
              }
            } catch {
              // Best-effort
            }

            // Build the span payload now that overrides are resolved.
            // Use the OTel span's NATIVE W3C ids so the parent pointer resolves
            // to the real parent row (the enclosing agent span, or a nested
            // OTel span). `trace_id` equals session.traceId (set from the agent
            // span), so the composition key above still matches.
            const spanName = String(userSpanName ?? effectiveModel);
            const sctx = span.spanContext();
            const nativeTraceId = sctx?.traceId && sctx.traceId !== "00000000000000000000000000000000"
              ? sctx.traceId
              : traceId;
            const nativeSpanId = sctx?.spanId && sctx.spanId !== "0000000000000000"
              ? sctx.spanId
              : randomHex16();
            // Rewrite over any dropped framework ancestors (LangChain/LangGraph
            // intermediate spans) so this LLM call nests under the surviving
            // agent/chain root instead of being orphaned. No-op when the parent
            // is a kept span.
            const parentSpanId = _resolveKeptParent(
              String(_parentIdOf(span) ?? ""),
              rootSpanId ?? "",
            );
            const spanObj = {
              trace_id: nativeTraceId,
              span_id: nativeSpanId,
              parent_span_id: parentSpanId,
              span_kind: "llm" as const,
              span_name: spanName,
              span_order: spanOrder,
              start_time: startTimeISO,
              end_time: endTimeISO,
            };

            // Phase B: synthesise a `usage` block from the gen_ai.* semconv
            // attributes the OpenLLMetry instrumentor stamped on this span.
            // Downstream, an unrecognized shape is treated as openai_compatible_chat
            // unless the effective provider has a more specific shape.
            //
            // N1 — provider and shape are TWO AXES, and only one of them is the
            // serving vendor:
            // • payload `provider` = SERVING slug: the host the bytes
            // were actually billed by, e.g. an Anthropic SDK client pointed
            // at api.minimax.io reports "minimax". Never weaken this.
            // • `usage.shape` = WIRE surface: which client SDK /
            // instrumentor produced these token fields.
            // Keying the shape off the serving slug (the switch below, on its
            // own) mislabels Anthropic-wire usage as openai_compatible_chat
            // whenever the host remaps — "minimax" has no case, so it falls to
            // the default. The server then runs the OpenAI mapper over
            // Anthropic fields: `cache_creation_input_tokens` is unknown there
            // so cache WRITES land unpriced in extra_units ($0), and Anthropic's
            // `input_tokens` is already cache-EXCLUSIVE so the OpenAI mapper
            // subtracts cache reads a second time (under-counted text input).
            //
            // The wire-first override below is deliberately ANTHROPIC-ONLY.
            // Anthropic is the one family whose wire shape is not
            // OpenAI-compatible while still being reachable through a remapped
            // host. Widening it to google/bedrock/cohere would regress paths
            // that correctly emit openai_compatible_chat today (notably the
            // LangChain-Gemini spans, which carry a Google `gen_ai.system` but
            // OpenAI-shaped token fields).
            const usageShape = (() => {
              // Wire axis signal 1: the pre-override provider (line ~1010),
              // read from gen_ai.system / gen_ai.provider.name and never
              // reassigned — the stash override only ever touches effProvider.
              let wireIsAnthropic = provider === "anthropic";
              if (!wireIsAnthropic) {
                // Signal 2: the emitting instrumentation scope, for spans that
                // carry no gen_ai.system attr at all. Same read (and same
                // fail-open contract) as the stream-suppression tag in onStart.
                try {
                  const scope = String(
                    (span as any).instrumentationScope?.name ??
                      (span as any).instrumentationLibrary?.name ??
                      "",
                  ).toLowerCase();
                  // Node scope: "@traceloop/instrumentation-anthropic";
                  // Python scope: "opentelemetry.instrumentation.anthropic".
                  wireIsAnthropic =
                    scope.includes("instrumentation-anthropic") ||
                    scope.includes("instrumentation.anthropic");
                } catch {
                  // Ignore the signal — fall through to the switch below.
                }
              }
              if (wireIsAnthropic) return "anthropic_messages";
              switch ((effProvider || "").toLowerCase()) {
                case "anthropic":   return "anthropic_messages";
                case "openai":      return "openai_chat";
                case "google":
                case "gemini":      return "google_genai";
                case "bedrock":     return "bedrock_converse";
                case "cohere":      return "cohere_chat";
                case "openrouter":  return "openrouter_routed";
                default:            return "openai_compatible_chat";
              }
            })();
            const attrReasoningTokens = Number(attrs["gen_ai.usage.reasoning_tokens"] ?? 0);
            // G3-O1: LC spans never carry the reasoning attr — fall back to
            // the enforcer's LangChain stash, clamped to the final output
            // count so a concat over-sum can never exceed what is billed.
            // Shape-gated to the two LC arms with a reasoning source
            // (anthropic_messages emits nothing new — langchain_anthropic has
            // no reasoning data); the attr path (every non-LC span) stays
            // bit-identical for every shape.
            const reasoningTokens =
              attrReasoningTokens > 0
                ? attrReasoningTokens
                : usageShape === "openai_chat" || usageShape === "google_genai"
                  ? Math.min(Math.max(0, stashReasoningTokens), outputTokens)
                  : 0;
            // Prefer LangChain message stash for cache-write when applied (span
            // attrs often lack creation tokens on the LC/Traceloop path).
            const cacheCreationTokens = usedLangchainMessageStash
              ? stashCacheCreationTokens
              : Number(
                  attrs["gen_ai.usage.cache_creation_input_tokens"]
                  ?? attrs["gen_ai.usage.cache_creation.input_tokens"]
                  ?? attrs["anthropic.usage.cache_creation_input_tokens"]
                  ?? 0
                );
            // raw.prompt_tokens MUST be cache-INCLUSIVE for the OpenAI-family
            // shapes (openai_chat / openrouter_routed / openai_compatible_chat):
            // their mapper computes text_input = prompt_tokens - cached, so a
            // cache-EXCLUSIVE prompt_tokens double-subtracts cached. The attr
            // path already carries the provider's verbatim inclusive
            // prompt_tokens (locked by sdk-usage-parity-fixture.json); only the
            // stream-stash fallback holds a netted (exclusive) value. So: when
            // the stash captured the provider's verbatim chunk usage, emit it as
            // raw; else, only if the numbers came from the stash, add cached back
            // to make prompt_tokens inclusive. The attr path (usedStashNumbers
            // false, stashRawUsage absent) stays bit-identical.
            const rawHasPositiveCount = (u: any): boolean => {
              try {
                return (
                  Boolean(u) &&
                  typeof u === "object" &&
                  (Number(u.prompt_tokens) > 0 ||
                    Number(u.completion_tokens) > 0 ||
                    Number(u.input_tokens) > 0 ||
                    Number(u.output_tokens) > 0)
                );
              } catch {
                return false;
              }
            };
            let usageRaw: any;
            // The OpenAI stream guard stashes the provider's verbatim chunk
            // usage as a size-capped JSON attr (TP_STREAM_RAW_USAGE_ATTR).
            // Used ONLY when the enforcer stash didn't supply raw and the
            // token numbers came from span attrs (no stash / LC overrides):
            // on that path the attr numbers and this JSON come from the same
            // usage object, so raw stays consistent with the counts. Guarded
            // parse; any failure falls through to the constructed raw.
            let guardRaw: any;
            if (!usedStashNumbers && !usedLangchainMessageStash) {
              try {
                const rawJson = attrs[TP_STREAM_RAW_USAGE_ATTR];
                if (typeof rawJson === "string" && rawJson.length <= 4096) {
                  const parsed = JSON.parse(rawJson);
                  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
                    guardRaw = parsed;
                  }
                }
              } catch {
                guardRaw = undefined;
              }
            }
            // Detail-bearing re-check for the non-streaming verbatim stash
            // (same predicate as the enforcer side, re-run on the clone).
            const rawHasDetailObjects = (u: any): boolean => {
              try {
                const isObj = (v: any): boolean => v != null && typeof v === "object";
                return (
                  Boolean(u) &&
                  (isObj(u.prompt_tokens_details) ||
                    isObj(u.completion_tokens_details) ||
                    isObj(u.promptTokensDetails) ||
                    isObj(u.completionTokensDetails))
                );
              } catch {
                return false;
              }
            };
            if (usedStashNumbers && rawHasPositiveCount(stashRawUsage)) {
              usageRaw = stashRawUsage;
            } else if (rawHasPositiveCount(guardRaw)) {
              usageRaw = guardRaw;
            } else if (
              nonStreamVerbatimRaw &&
              rawHasDetailObjects(nonStreamVerbatimRaw) &&
              // Consistency gate: the constructed raw below would emit exactly
              // these prompt/completion counts (see inclusivePrompt), and the
              // Traceloop instrumentor set the span attrs from the SAME usage
              // object this stash cloned — so any mismatch means the stash
              // belongs to a different call (mis-attribution). Exact equality;
              // reject → fall through to the constructed raw.
              Number(nonStreamVerbatimRaw.prompt_tokens) ===
                (usedStashNumbers ? inputTokens + cachedTokens : inputTokens) &&
              Number(nonStreamVerbatimRaw.completion_tokens) === outputTokens
            ) {
              // Non-streaming openai-wire call whose usage carries detail
              // objects the bare token attrs drop (reasoning_tokens,
              // cached_tokens): forward the provider's verbatim usage
              // wholesale. Positional counts stay attr-derived as today.
              usageRaw = nonStreamVerbatimRaw;
            } else {
              const inclusivePrompt = usedStashNumbers ? inputTokens + cachedTokens : inputTokens;
              usageRaw = {
                prompt_tokens: inclusivePrompt,
                completion_tokens: outputTokens,
                input_tokens: inclusivePrompt,
                output_tokens: outputTokens,
                prompt_tokens_details: cachedTokens ? { cached_tokens: cachedTokens } : undefined,
                completion_tokens_details: reasoningTokens ? { reasoning_tokens: reasoningTokens } : undefined,
                cache_read_input_tokens: cachedTokens || undefined,
                cache_creation_input_tokens: cacheCreationTokens || undefined,
              };
              // G3-O1 LC-Gemini: LC's output_tokens is thoughts-INCLUSIVE
              // while the google_genai mapper's candidates_token_count is
              // EXCLUSIVE-of-thoughts (additive) — emit the split so thoughts
              // bill at the reasoning rate with total output unchanged.
              // Stash-sourced only: an attr-sourced count's output-inclusion
              // semantics are unverified, so that path stays bit-identical.
              if (
                usageShape === "google_genai" &&
                reasoningTokens > 0 &&
                attrReasoningTokens <= 0
              ) {
                usageRaw.thoughts_token_count = reasoningTokens;
                usageRaw.candidates_token_count = Math.max(
                  0,
                  outputTokens - reasoningTokens,
                );
              }
            }
            const usageBlock = {
              shape: usageShape,
              raw: usageRaw,
              ...(serviceTier ? { tier: serviceTier } : {}),
            };

            // Success-call duration: streamed spans carry the tap's monotonic
            // total; otherwise derive from the span's own start/end.
            // call_outcome.status defaults to 'success' server-side, so omitting
            // it never flips a row to failed (failures never reach this path —
            // usage-less spans early-return above).
            let durationMs = 0;
            try {
              // Prefer stream tap total_ms when it is a finite positive number;
              // else wall delta. isFinite rejects NaN/±Infinity so a corrupt
              // total falls back to wall (old `Number(x)||wall` kept Infinity
              // as truthy → JSON null). toDurationMs: sub-ms positive → 1.
              const latTotal = Number((latencyData as any)?.total_ms);
              durationMs =
                Number.isFinite(latTotal) && latTotal > 0
                  ? toDurationMs(latTotal)
                  : toDurationMs(endMs - startMs);
            } catch {
              durationMs = 0;
            }

            // ── This call's own reroute marker ──
            // Stripped from the span attributes above; re-added here ONLY when
            // the span's captured obs key matches the key `_applyReroute`
            // stashed under — i.e. this row belongs to the call that was
            // actually rerouted. Same key the local_decision claim below uses.
            // The peek never removes the record, so a call that emits several
            // rows keeps the marker on all of them. Serialized to JSON to
            // preserve the wire shape this path has always produced for
            // object-valued metadata (the attribute hop stringified it).
            try {
              let _routingKey: string | null = null;
              try {
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                const _mk = (span as any)[TP_OBS_KEY];
                _routingKey = typeof _mk === "string" && _mk ? _mk : null;
              } catch {
                _routingKey = null;
              }
              stampRoutingMarker(metadata, getCurrentSession(), _routingKey, true);
            } catch {
              // fail-open: the row simply carries no reroute provenance
            }

            // Fire and forget
            tp.log(
              userId,
              paidPlan,
              workflowName,
              sessionId,
              effectiveModel,
              effProvider,
              inputTokens,
              outputTokens,
              cachedTokens,
              metadata,
              spanObj,
              promptComp,
              responseComp,
              (() => {
                // Drain shadow/dry_run observations on the SUCCESS path too
                // (parity with Python). Without this, a dry_run app whose calls
                // all succeed would never ship its would-block/would-reroute
                // observations. The drain is KEYED by the span's captured obs
                // key (stamped in onStart, inside the call's obs scope — this
                // deferred callback itself runs outside it): a call's entries
                // go to whichever of ITS OWN drain paths runs first (success
                // here, or _emitCallFailureLog/_emitLocalBlockLog), so success
                // and failure drains stay mutually exclusive per call and
                // cross-call theft is impossible before the staleness window.
                // An unmarked span drains as an unknown claimant (null →
                // untagged + stale entries only).
                let _obsKey: string | null = null;
                try {
                  const _mk = (span as any)[TP_OBS_KEY];
                  _obsKey = typeof _mk === "string" && _mk ? _mk : null;
                } catch {
                  _obsKey = null;
                }
                const observations = drainObservations(_obsKey);
                // Also ship applied local_decision so /log can emit the
                // authoritative REQUEST_REROUTED (server no longer treats
                // /check as applied confirmation). Re-resolve session here —
                // the earlier composition-stash `session` is try-block scoped.
                // Claimed with the SAME span-recorded obs key as the drain
                // above: the decision goes to its OWN call's row, so a burst
                // of concurrent calls emits one REQUEST_REROUTED each instead
                // of one for the whole burst on an arbitrary row.
                let localDecision: Record<string, unknown> | undefined;
                try {
                  const sess: any = getCurrentSession();
                  localDecision = claimLocalDecision(sess, _obsKey);
                } catch {
                  localDecision = undefined;
                }
                // model_extras: api_base and original_provider are INDEPENDENT
                // keys (mirrors the Python SDK's handling — two separate `if`
                // blocks under one `if _extras:` gate). Build the object, add each
                // key only when truthy, and emit the block ONLY when it has ≥1 key
                // (never `model_extras: {}`). Do NOT nest one key under the other:
                // an absent api_base must not drop original_provider.
                const modelExtras: Record<string, string> = {};
                if (apiBase) modelExtras.api_base = apiBase;
                if (gwOriginalProvider) {
                  modelExtras.original_provider = gwOriginalProvider;
                }
                return {
                  usage: usageBlock,
                  ...(Object.keys(modelExtras).length > 0
                    ? { model_extras: modelExtras }
                    : {}),
                  ...(latencyData ? { latency: latencyData } : {}),
                  ...(durationMs > 0
                    ? { call_outcome: { status: "success", duration_ms: durationMs } }
                    : {}),
                  ...(localDecision ? { local_decision: localDecision } : {}),
                  ...(observations.length > 0 ? { observations } : {}),
                  planSource,
                };
              })(),
            );
          } catch {
            // Fail-open: never crash on deferred log
          } finally {
            _deferredLogs.delete(deferred);
            resolveDeferred();
          }
        });
      }
    } catch {
      // Fail-open: never crash on span end
    }
  }

  /**
   * Emits an `agent` span row for a workflow/sub-agent anchor span. Zero
   * tokens, no model, no composition — it exists only to give the trace tree
   * a real root/branch node. All fields come from span attributes + native
   * context (no session lookup, so it is timing-independent).
   */
  private _logStructuralSpan(
    span: ReadableSpan,
    attrs: Record<string, unknown>,
    kind: string = "agent",
  ): void {
    try {
      const tp = getClient();
      if (!tp) return;

      // Only `agent` / `chain` are valid structural anchors; anything unexpected
      // falls back to `agent` (the historical default) — never an LLM kind.
      const structuralKind = kind === "chain" ? "chain" : "agent";

      const userId = String(attrs["tp.user_id"] ?? "anonymous");
      const paidPlan = String(attrs["tp.paid_plan"] ?? "free");
      const planSource = _planSourceFromAttrs(attrs);
      const workflowName = String(attrs["tp.workflow_name"] ?? "default");
      const sessionId = String(attrs["tp.session_id"] ?? "");

      const metadata: Record<string, unknown> = { workflow_name: workflowName };
      if (sessionId) metadata.session_id = sessionId;
      for (const [k, v] of Object.entries(attrs)) {
        // B4: agent / chain / tool rows execute no model call, so reroute
        // provenance can never be theirs. Stripped here and never re-added.
        if (k === TP_ROUTING_ATTR) continue;
        if (k.startsWith("tp.meta.")) metadata[k.slice(8)] = v;
      }

      const startMs = span.startTime[0] * 1000 + span.startTime[1] / 1000000;
      const endMs = span.endTime[0] * 1000 + span.endTime[1] / 1000000;
      const sctx = span.spanContext();

      const spanObj = {
        trace_id: sctx?.traceId ?? "",
        span_id: sctx?.spanId ?? randomHex16(),
        parent_span_id: _parentIdOf(span) ?? "",
        span_kind: structuralKind,
        span_name: workflowName,
        span_order: 0,
        start_time: new Date(startMs).toISOString(),
        end_time: new Date(endMs).toISOString(),
      };

      // Map the anchor span's OTel status → call_outcome so a workflow / agent /
      // chain whose body threw uncaught reports status='failed' instead of the
      // server's default 'success'. Mirrors `_logToolSpan`; reading the
      // ReadableSpan here never affects the customer's already-propagating error.
      const failed = span.status?.code === SpanStatusCode.ERROR;
      const callOutcome: Record<string, unknown> = {
        status: failed ? "failed" : "success",
        duration_ms: toDurationMs(endMs - startMs),
      };
      if (failed) {
        // Route the raw span status message through the scrub helper (no
        // pre-stringify); default 'redacted' ships a hash, not raw text.
        Object.assign(callOutcome, scrubErrorMessage(span.status?.message, resolveErrorDetail()));
        const ek = attrs["error.type"];
        if (ek) callOutcome.error_kind = String(ek);
      }

      // Zero-usage structural row (agent/chain): excluded from budget/loop
      // evaluation and analytics by span_kind, per the wire contract.
      tp.log(
        userId,
        paidPlan,
        workflowName,
        sessionId,
        "",
        "",
        0,
        0,
        0,
        metadata,
        spanObj,
        [],
        [],
        { call_outcome: callOutcome, planSource },
      );
    } catch {
      // Fail-open: never crash on agent-span logging.
    }
  }

  /**
   * Emits a `tool` span row for a tool / function execution. Zero tokens, no
   * model, no cost — it slots into the trace tree between LLM calls. Tool
   * args/results are reduced to (sha1-16 hash, length); raw content never
   * leaves the process.
   */
  private _logToolSpan(
    span: ReadableSpan,
    attrs: Record<string, unknown>,
    name: string,
  ): void {
    try {
      const tp = getClient();
      if (!tp) return;

      const toolName = toolNameFromSpan(attrs, name);
      const toolType = String(attrs["gen_ai.tool.type"] ?? "function");
      // OTel frameworks (LangChain/LangGraph/CrewAI/Agno) rarely set
      // gen_ai.tool.call.id. Fall back to the pending stash populated from the
      // preceding LLM response (name-exact FIFO). Prefer the attr when present;
      // never invent ids.
      let toolCallId = String(attrs["gen_ai.tool.call.id"] ?? "");
      if (!toolCallId) {
        try {
          toolCallId = getCurrentSession().popPendingToolCallId(toolName) || "";
        } catch {
          toolCallId = "";
        }
      }
      const [paramHash, paramLen] = hashLen(
        attrs["traceloop.entity.input"] ??
          attrs["gen_ai.tool.call.arguments"] ??
          attrs["tool.parameters"],
      );
      const [resultHash, resultLen] = hashLen(
        attrs["traceloop.entity.output"] ?? attrs["gen_ai.tool.call.result"],
      );

      const userId = String(attrs["tp.user_id"] ?? "anonymous");
      const paidPlan = String(attrs["tp.paid_plan"] ?? "free");
      const planSource = _planSourceFromAttrs(attrs);
      const workflowName = String(attrs["tp.workflow_name"] ?? "default");
      const sessionId = String(attrs["tp.session_id"] ?? "");
      const metadata: Record<string, unknown> = { workflow_name: workflowName };
      if (sessionId) metadata.session_id = sessionId;
      for (const [k, v] of Object.entries(attrs)) {
        // B4: agent / chain / tool rows execute no model call, so reroute
        // provenance can never be theirs. Stripped here and never re-added.
        if (k === TP_ROUTING_ATTR) continue;
        if (k.startsWith("tp.meta.")) metadata[k.slice(8)] = v;
      }

      const startMs = span.startTime[0] * 1000 + span.startTime[1] / 1000000;
      const endMs = span.endTime[0] * 1000 + span.endTime[1] / 1000000;
      const sctx = span.spanContext();
      const traceId = String(attrs["tp.trace_id"] ?? "");
      const nativeTraceId =
        sctx?.traceId && sctx.traceId !== "00000000000000000000000000000000"
          ? sctx.traceId
          : traceId;
      const nativeSpanId =
        sctx?.spanId && sctx.spanId !== "0000000000000000"
          ? sctx.spanId
          : randomHex16();
      // Rewrite over dropped framework ancestors so this tool span nests under
      // the surviving agent/chain root (orphan-rewrite). No-op when the parent
      // is a kept span. Tool spans carry no tp.root_span_id, so the fallback is
      // "" — identical to today's default.
      const parentSpanId = _resolveKeptParent(
        String(_parentIdOf(span) ?? ""),
        String(attrs["tp.root_span_id"] ?? ""),
      );

      const failed = span.status?.code === SpanStatusCode.ERROR;
      const callOutcome: Record<string, unknown> = {
        status: failed ? "failed" : "success",
        duration_ms: toDurationMs(endMs - startMs),
      };
      if (failed) {
        // Route the raw span status message through the scrub helper
        // (no pre-stringify); default 'redacted' ships a hash, not raw text.
        Object.assign(callOutcome, scrubErrorMessage(span.status?.message, resolveErrorDetail()));
        const ek = attrs["error.type"];
        if (ek) callOutcome.error_kind = String(ek);
      }

      const spanObj = {
        trace_id: nativeTraceId,
        span_id: nativeSpanId,
        parent_span_id: parentSpanId,
        span_kind: "tool" as const,
        span_name: toolName,
        span_order: 0,
        start_time: new Date(startMs).toISOString(),
        end_time: new Date(endMs).toISOString(),
      };

      tp.log(
        userId,
        paidPlan,
        workflowName,
        sessionId,
        "",
        "",
        0,
        0,
        0,
        metadata,
        spanObj,
        [],
        [],
        {
          tool: {
            name: toolName,
            type: toolType,
            call_id: toolCallId,
            param_hash: paramHash,
            param_length: paramLen,
            result_hash: resultHash,
            result_length: resultLen,
          },
          call_outcome: callOutcome,
          planSource,
        },
      );
    } catch {
      // Fail-open: never crash on tool-span logging.
    }
  }

  async forceFlush(): Promise<void> {
    // NON-greedy: forceFlush is a public per-request OTel API (platform
    // integrations like @vercel/otel call it on live traffic). It still
    // awaits the deferred logs, but their drains stay KEYED — a greedy
    // window here would reopen cross-call observation stealing against
    // concurrent in-flight calls on every request. Orphans whose owning
    // /log never fires ride the staleness fallback instead.
    await forceFlushTokenPoliceSpans({ greedy: false });
  }

  async shutdown(): Promise<void> {
    // Drain deferred LLM logs only — never call client.close() from here
    // (customer OTel may invoke processor.shutdown independently). Terminal
    // path → greedy window (default), so no queued observation is stranded.
    await forceFlushTokenPoliceSpans();
  }
}

// ── Setup ─────────────────────────────────────────────────────────

/**
 * Initializes a LOCAL TracerProvider with ZERO EXPORTERS.
 * If no real global OTel provider exists yet, ours is registered globally.
 * If a customer already installed one (resolved via the global proxy's delegate),
 * ours is NOT registered — that would fail silently and clobber the customer's
 * global propagator — and our TokenPoliceSpanProcessor is attached to the
 * customer's provider instead. A customer provider lacking addSpanProcessor is
 * left untouched (metering still flows through our privately-bound instrumentors).
 * Coexistence is covered by tests/otelCoexistence.test.ts.
 *
 * @param instrumentModules - Optional map of module references to instrument.
 * When provided, calls manuallyInstrument() with the passed module — solves
 * ESM import ordering. When omitted, falls back to auto-discovery.
 */
export function setupOpenTelemetry(
  instrumentModules?: InstrumentModules,
): void {
  if (_isSetup) return;

  // ── Create our own TracerProvider ──────────────────────────────
  // We build our own provider here; whether we register() it globally or
  // instead piggyback onto an existing one is decided below (see the
  // "register globally as fallback" block) so we don't conflict with any
  // existing OTel provider (Sentry, Datadog, Vercel, etc.).
  // OTel SDK 2.x: processors are constructor-only (addSpanProcessor is gone).
  const provider = new BasicTracerProvider({
    spanProcessors: [new TokenPoliceSpanProcessor()],
  });
  _tracerProvider = provider;

  // Expose our tracer to context.session()/workflow() so they can open the
  // "agent" anchor span. We use OUR private provider's tracer so agent rows
  // always reach our processor; LLM-span parenting still works because it
  // flows through the global context manager, not the provider.
  _setAgentTracerFactory(() => _tracerProvider?.getTracer("token-police"));

  // ── Attach instrumentations ───────────────────────────────────
  if (instrumentModules && Object.keys(instrumentModules).length > 0) {
    _applyManualInstrumentations(provider, instrumentModules);
  } else {
    _applyAutoDiscoveredInstrumentations(provider);
  }

  // ── LangChain — special-case (different manuallyInstrument shape) ──
  _applyLangChainInstrumentation(provider, instrumentModules);

  // ── Register globally, or piggyback onto the customer's provider ──
  // The JS OpenTelemetry API always registers a singleton ProxyTracerProvider
  // globally and keeps the real provider behind it, so trace.getTracerProvider()
  // returns a ProxyTracerProvider whether or not a customer already installed
  // one. We therefore resolve the REAL provider through the proxy's delegate:
  // - no real provider yet (the delegate is the Noop sentinel) → set ours as
  // the global tracer provider so instrumentors that create tracers via
  // trace.getTracer() route spans through our processor. (SDK 1.x did this via
  // BasicTracerProvider.register(), which ALSO installed a global W3C
  // propagator; 2.x removed register() from BasicTracerProvider and we
  // deliberately install no propagator — the SDK never injects or extracts
  // trace headers, so a global propagator was a side effect on the customer's
  // process with no benefit.)
  // - a real customer provider exists → do NOT touch the globals. Registering
  // a second provider fails silently (a duplicate-registration diagnostic).
  // Instead we attach our span processor directly to their provider.
  // - a customer provider without addSpanProcessor (every OTel SDK 2.x
  // provider — processors are constructor-only there) → skip silently: our
  // instrumentors are already bound to our own private provider via
  // setTracerProvider, so token metering still works. This is the accepted
  // best-effort residual for such providers.
  // All detection is fail-open: any probe error degrades to registering our own
  // provider (today's behavior). Covered by tests/otelCoexistence.test.ts.
  let realProvider: any = null;
  try {
    const proxy: any = trace.getTracerProvider();
    realProvider =
      typeof proxy?.getDelegate === "function" ? proxy.getDelegate() : proxy;
  } catch {
    realProvider = null;
  }

  if (!realProvider || realProvider.constructor?.name === "NoopTracerProvider") {
    trace.setGlobalTracerProvider(provider);
    // We set the global tracer provider; record it so teardown can undo
    // exactly this (and not a customer's).
    _registeredGlobalProvider = true;
  } else if (typeof realProvider.addSpanProcessor === "function") {
    realProvider.addSpanProcessor(new TokenPoliceSpanProcessor());
  }
  // else: real provider without addSpanProcessor — skip silently (see above).

  // ── Ensure a global context manager exists ────────────────────
  // session()/workflow() open an "agent" span via startActiveSpan and rely on
  // the active context propagating to OpenLLMetry's LLM spans (so they nest
  // under the agent) and to manual spans (via trace.getActiveSpan()).
  // BasicTracerProvider.register() does NOT install an async context manager
  // (only NodeTracerProvider does), so without this the active context is the
  // NoopContextManager and nothing nests. We install a minimal
  // AsyncLocalStorage-based manager. setGlobalContextManager is a no-op when a
  // real manager is already registered (e.g. the customer's), so this never
  // clobbers an existing OTel setup. Fail-open — never throw on setup.
  try {
    // Returns false when a manager is already registered (customer-owned), in
    // which case ours did NOT take effect and teardown must leave it alone.
    if (context.setGlobalContextManager(new TPContextManager())) {
      _registeredGlobalContextManager = true;
    }
  } catch {
    // A context manager is already registered (customer-owned) — fine.
  }

  _isSetup = true;
  logger.debug(
    "TokenPolice: Local OpenTelemetry extraction layer initialized.",
  );
}

// ── Loader-hook hardening ────────────────────────────────────────

/** Marks an already-wrapped patch/unpatch so re-hardening never double-wraps. */
const HARDENED_PATCH = Symbol("tp-hardened-patch");

/**
 * Per-instrumentor identity set of modules already patched via
 * manuallyInstrument(). See markManuallyInstrumented below.
 */
const MANUALLY_INSTRUMENTED = Symbol("tp-manually-instrumented-modules");

/**
 * Records that `userModule` was successfully patched by this instrumentor's
 * manuallyInstrument(), so the loader hook can skip re-patching the SAME
 * module later (patch idempotency by module identity).
 *
 * Why this matters: in instrumentModules (manual) mode the user's module was
 * loaded BEFORE the instrumentor's require-in-the-middle hook armed, so the
 * hook's per-file cache has never seen it. The app's first post-init
 * require() of that module fires the hook and applies a SECOND Traceloop
 * patch layer — on top of the enforcer wrapper installed over the manual
 * patch. That inverts the load-bearing layering (enforcer must stay on top):
 * the enforcer's anthropicStreamBypass sits below the new layer, so on
 * Anthropic SDKs with no root APIPromise export every streamed
 * messages.create() throws "APIPromise is not a constructor" INTO CUSTOMER
 * CODE; for other providers the duplicate layer double-counts spans/cost.
 *
 * Both identities are recorded (module + its `default`) because the hook may
 * see the CJS exports object or an ESM namespace while the user passed the
 * default-export class per the ESM guidance. A miss is fail-open: the hook
 * patches as before (status quo), never worse.
 */
export function markManuallyInstrumented(
  instrumentor: unknown,
  userModule: unknown,
): void {
  try {
    if (!instrumentor) return;
    if (
      userModule === null ||
      (typeof userModule !== "object" && typeof userModule !== "function")
    ) {
      return;
    }
    const inst = instrumentor as any;
    const set: Set<unknown> =
      inst[MANUALLY_INSTRUMENTED] ?? (inst[MANUALLY_INSTRUMENTED] = new Set());
    set.add(userModule);
    // Guarded: `.default` may be a hostile getter on exotic module objects.
    const def = (userModule as any).default;
    if (def !== null && (typeof def === "object" || typeof def === "function")) {
      set.add(def);
    }
  } catch {
    // fail-open: unmarked → hook behavior unchanged
  }
}

/**
 * True when `exports` identity-matches a module this instrumentor already
 * patched via manuallyInstrument(). Never throws.
 */
function _isManuallyInstrumentedModule(
  instrumentor: unknown,
  exports: unknown,
): boolean {
  try {
    const set = (instrumentor as any)?.[MANUALLY_INSTRUMENTED] as
      | Set<unknown>
      | undefined;
    if (!set || !set.size || exports === null || exports === undefined) {
      return false;
    }
    if (set.has(exports)) return true;
    if (typeof exports !== "object" && typeof exports !== "function") {
      return false;
    }
    // Guarded: `.default` may be a hostile getter.
    const def = (exports as any).default;
    return def !== null && def !== undefined && set.has(def);
  } catch {
    return false;
  }
}

// ── Anthropic in-band stream guard (nested/duplicate SDK copies) ──

/** Marks an installed stream-guard wrapper so re-install is a no-op. */
const IN_BAND_STREAM_GUARD = Symbol("tp-anthropic-in-band-stream-guard");

/**
 * True when an @anthropic-ai/sdk exports object is "in-band": the Traceloop
 * anthropic instrumentor patches it successfully (Anthropic.* resolves) but its
 * STREAMING branch throws at call time — it runs
 * `new moduleExports.APIPromise(...)` and these SDK versions (~0.28 → 0.41)
 * have no root APIPromise export. Mirrors the enforcer's anthropicStreamBypass
 * probe (enforcer.ts _wrapMethod). Never throws; a hostile getter is treated as
 * NOT in-band so the guard is skipped (status quo).
 */
function _isInBandAnthropicExports(exports: any): boolean {
  try {
    return (
      typeof (
        exports?.APIPromise ??
        exports?.default?.APIPromise ??
        exports?.Anthropic?.APIPromise
      ) !== "function"
    );
  } catch {
    return false;
  }
}

/**
 * After the Traceloop anthropic loader hook patches an IN-BAND copy of
 * @anthropic-ai/sdk, install a minimal stream guard over each patched `create`
 * so `create({stream: true})` routes AROUND the Traceloop layer straight to the
 * real method. Everything else still flows through the Traceloop layer, so
 * non-streaming telemetry is untouched.
 *
 * Why this exists: the enforcer only wraps the app's MAIN copy of the SDK (its
 * anthropicStreamBypass handles streaming there), but the armed loader hook
 * also patches NESTED/duplicate copies — e.g. under @langchain/anthropic when
 * conflicting version pins prevent hoisting. Such a copy carries the Traceloop
 * layer with no enforcer above it, so every streamed messages.create() throws
 * "TypeError: moduleExports.APIPromise is not a constructor" INTO CUSTOMER
 * CODE. With the guard, streamed calls on that copy run un-metered instead of
 * crashing (they produce no telemetry today either — they throw).
 *
 * Main-copy interactions (why this can't regress existing behavior):
 * - Manual mode: the main copy is deduped by markManuallyInstrumented, so the
 * LOADER HOOK never re-reaches it. _applyManualInstrumentations does call this
 * installer directly on the main copy right after its manuallyInstrument(),
 * because a class-form `instrumentModules.anthropic` is normalized into a
 * synthesized namespace that has no root APIPromise — and unlike Messages /
 * Beta.Messages, `Anthropic.Completions` has NO enforcer target above it to
 * bypass streaming, so an unguarded streamed completions.create() would throw
 * into customer code. Namespace-form input on a modern SDK still carries
 * APIPromise, so the _isInBandAnthropicExports gate makes that call a no-op
 * and the main copy is untouched exactly as before.
 * - Auto mode, in-band main copy: the guard installs, but the eager
 * manuallyInstrument() that follows uses the node-platform `_wrap`, which is
 * unwrap-first — it pops the guard (via its shimmer-compatible __unwrap) and
 * re-wraps the RAW method, restoring today's exact layering; the enforcer's
 * bypass then covers streaming as before.
 * - Safe copies (root APIPromise present, ≥0.50): not in-band — never touched.
 *
 * The guard is shimmer-compatible (__wrapped/__original/__unwrap) so OTel's
 * isWrapped()/unwrap(), the Traceloop unpatch, and the enforcer's one-level
 * bypass unwrap all treat it exactly like an instrumentor layer whose original
 * is the REAL method. Fully fail-open: no path in here may throw into the
 * customer's require()/import.
 */
export function installAnthropicInBandStreamGuard(exports: unknown): void {
  try {
    if (
      exports === null ||
      (typeof exports !== "object" && typeof exports !== "function")
    ) {
      return;
    }
    if (!_isInBandAnthropicExports(exports)) return;
    // Exactly the surfaces the Traceloop instrumentor patches. `.Anthropic`
    // must exist here — its patch() would have thrown otherwise and this
    // installer only runs after a successful patch.
    let root: any;
    try {
      root = (exports as any).Anthropic;
    } catch {
      return; // hostile getter — nothing was patched, nothing to guard
    }
    const protos: any[] = [];
    try { const p = root?.Completions?.prototype; if (p) protos.push(p); } catch { /* skip surface */ }
    try { const p = root?.Messages?.prototype; if (p) protos.push(p); } catch { /* skip surface */ }
    try { const p = root?.Beta?.Messages?.prototype; if (p) protos.push(p); } catch { /* skip surface */ }
    for (const proto of protos) {
      try {
        const patched = proto?.create;
        if (
          typeof patched !== "function" ||
          (patched as any)[IN_BAND_STREAM_GUARD] ||
          (patched as any).__wrapped !== true ||
          typeof (patched as any).__original !== "function"
        ) {
          continue; // not a freshly Traceloop-wrapped method — leave alone
        }
        const real = (patched as any).__original;
        const guard = function tpAnthropicInBandStreamGuard(
          this: any,
          ...args: any[]
        ) {
          // Only the `.stream` READ is guarded — never the dispatch itself.
          // Wrapping the apply too would swallow a synchronous throw from the
          // real method and re-execute the call through the Traceloop layer
          // (double invocation, and its streaming branch would then hit the
          // very APIPromise crash this guard exists to prevent).
          let isStream = false;
          try {
            isStream = (args?.[0] as any)?.stream === true;
          } catch {
            // Hostile `.stream` getter — fall through to the patched layer;
            // both it and the raw SDK read `.stream` too, so status quo.
          }
          if (isStream) {
            // Skip the Traceloop layer: its streaming branch would throw
            // `new moduleExports.APIPromise(...)` synchronously. `real` is
            // the SDK's own method — its errors (401s, network) are the
            // app's legitimate provider errors, exactly as if uninstrumented.
            return real.apply(this, args);
          }
          return patched.apply(this, args);
        };
        (guard as any)[IN_BAND_STREAM_GUARD] = true;
        // shimmer-compatible markers. __original is the RAW method (not the
        // Traceloop layer) so (a) the enforcer's one-level bypass unwrap lands
        // on the real create, and (b) __unwrap / unpatch / unwrap-first re-wrap
        // all restore the raw method in one hop — the same end state the
        // instrumentor's own unpatch produces.
        (guard as any).__wrapped = true;
        (guard as any).__original = real;
        (guard as any).__unwrap = () => {
          try {
            if (proto.create === guard) proto.create = real;
          } catch {
            /* frozen prototype — leave as is */
          }
        };
        proto.create = guard;
        logger.debug(
          () =>
            "TokenPolice: installed anthropic in-band stream guard on a " +
            "loader-hook-patched SDK copy (streamed calls route around the " +
            "instrumentor layer).",
        );
      } catch {
        // fail-open per surface: frozen prototype / hostile getter → status quo
      }
    }
  } catch {
    // fail-open: guard install must never break the customer's require()
  }
}

// ── OpenAI stream guard (accumulator crash + lost stream usage) ──

/** Marks a guarded `_streamingWrapPromise` so re-install is a no-op. */
const OPENAI_STREAM_GUARD = Symbol("tp-openai-stream-guard");

/**
 * Span attribute carrying the provider's verbatim stream-usage JSON, set by
 * the OpenAI stream guard when a streamed chat chunk carries `usage`.
 * Consumed ONLY by TokenPoliceSpanProcessor.onEnd as the row's raw usage
 * block, and only when the enforcer's stream stash didn't already supply one.
 * Never leaves the process: the provider has zero exporters, and the row
 * metadata loop copies only `tp.meta.*` keys — this key is neither.
 */
export const TP_STREAM_RAW_USAGE_ATTR = "tp.stream_raw_usage";

/** Size cap for the raw-usage JSON attr — larger payloads are skipped. */
const _STREAM_RAW_USAGE_MAX_LEN = 2048;

/** One fallback warning per process (mirrors `_warnedProviders` intent). */
let _openaiStreamGuardWarnedFallback = false;

/** Per-call synthetic tool-call index state (see the sanitizer below). */
type _ToolCallIndexState = {
  nextIndex: number;
  lastIndex: number;
  seenAny: boolean;
};

/**
 * Sanitize ONE streamed chat chunk for the Traceloop OpenAI accumulator
 * (`_streamingWrapPromise`'s chat branch, which has NO try/catch upstream —
 * the text-completion branch does; upstream oversight). Returns the chunk
 * itself when it is already safe, or a spread-shallow CLONE along the mutated
 * path only. NEVER mutates the input — the customer app must receive the
 * byte-identical original objects.
 *
 * Known crash shapes (all confirmed against the customer's Gemini
 * OpenAI-compat stream and the 0.26.0 dist):
 * - chunk with no `choices` array → `chunk.choices[0]` TypeError.
 * - `choices[0]` present but no `delta` → `chunk.choices[0]?.delta.content`
 *   TypeError (the optional chain stops at `[0]`, not at `.delta`).
 * - tool-call deltas missing a numeric `index` (Gemini omits it on EVERY
 *   delta) → `length < NaN + 1` never pushes a slot, then
 *   `tool_calls[undefined].id` TypeError. Synthetic indexes are assigned
 *   per-call: a delta with a truthy `id` (or the first delta ever) opens a
 *   new slot; an id-less delta continues the last one. Deltas that DO carry
 *   a numeric index keep it, re-syncing the counters.
 *
 * Fail-open: the caller wraps this in try/catch and falls back to passing
 * the original chunk through unmodified (today's behavior).
 */
function _sanitizeOpenAIChunkForInstrumentor(
  chunk: any,
  tc: _ToolCallIndexState,
): any {
  if (chunk === null || typeof chunk !== "object") return chunk;
  const choices = chunk.choices;
  if (!Array.isArray(choices)) {
    // The accumulator reads `chunk.choices[0]` UNchained → TypeError.
    return { ...chunk, choices: [] };
  }
  if (choices.length === 0) return chunk; // usage-only terminal frame — safe
  const c0 = choices[0];
  if (c0 === null || typeof c0 !== "object") return chunk; // `[0]` IS chained
  const delta = c0.delta;
  if (delta === null || delta === undefined) {
    return { ...chunk, choices: [{ ...c0, delta: {} }, ...choices.slice(1)] };
  }
  const toolCalls = typeof delta === "object" ? delta.tool_calls : undefined;
  if (!Array.isArray(toolCalls) || toolCalls.length === 0) return chunk;
  let changed = false;
  const fixedCalls = toolCalls.map((t: any) => {
    if (t === null || typeof t !== "object") return t; // unknown shape → fallback covers
    const idx = t.index;
    if (typeof idx === "number" && Number.isFinite(idx)) {
      // Provider-supplied index wins; keep the counters in sync with it.
      tc.nextIndex = Math.max(tc.nextIndex, idx + 1);
      tc.lastIndex = idx;
      tc.seenAny = true;
      return t;
    }
    const assigned = t.id || !tc.seenAny ? tc.nextIndex++ : tc.lastIndex;
    tc.lastIndex = assigned;
    tc.seenAny = true;
    changed = true;
    return { ...t, index: assigned };
  });
  if (!changed) return chunk;
  return {
    ...chunk,
    choices: [{ ...c0, delta: { ...delta, tool_calls: fixedCalls } }, ...choices.slice(1)],
  };
}

/**
 * Stamp streamed-usage attributes on the (still-open) instrumentor span,
 * using the EXACT names TokenPoliceSpanProcessor.onEnd reads. Overwrites on
 * every sighting, so cumulative-usage compat endpoints resolve to
 * last-chunk-wins. Also stashes the verbatim usage JSON (size-capped) as
 * TP_STREAM_RAW_USAGE_ATTR for the row's raw usage block. Never throws.
 */
function _applyStreamUsageAttrs(span: any, usage: any): void {
  try {
    if (!span || typeof span.setAttribute !== "function") return;
    if (!usage || typeof usage !== "object") return;
    const input = Number(usage.prompt_tokens ?? usage.input_tokens);
    const output = Number(usage.completion_tokens ?? usage.output_tokens);
    if (Number.isFinite(input) && input >= 0) {
      span.setAttribute("gen_ai.usage.input_tokens", input);
    }
    if (Number.isFinite(output) && output >= 0) {
      span.setAttribute("gen_ai.usage.output_tokens", output);
    }
    const cached = Number(usage.prompt_tokens_details?.cached_tokens);
    if (Number.isFinite(cached) && cached > 0) {
      span.setAttribute("gen_ai.usage.cache_read_tokens", cached);
    }
    const reasoning = Number(usage.completion_tokens_details?.reasoning_tokens);
    if (Number.isFinite(reasoning) && reasoning > 0) {
      span.setAttribute("gen_ai.usage.reasoning_tokens", reasoning);
    }
    try {
      const json = JSON.stringify(usage);
      if (
        typeof json === "string" &&
        json.length > 0 &&
        json.length <= _STREAM_RAW_USAGE_MAX_LEN
      ) {
        span.setAttribute(TP_STREAM_RAW_USAGE_ATTR, json);
      }
    } catch {
      // raw attr is optional — positional attrs above already carry metering
    }
  } catch {
    // fail-open: usage capture must never break the customer's iteration
  }
}

/** Remember + stamp a chunk's usage (if any). Never throws. */
function _captureStreamChunkUsage(span: any, state: any, chunk: any): void {
  try {
    const usage = chunk?.usage; // hostile getter → caught below
    if (usage === null || usage === undefined) return;
    state.lastUsage = usage; // last one wins (cumulative-usage endpoints)
    _applyStreamUsageAttrs(span, usage);
  } catch {
    // fail-open
  }
}

/**
 * Defensively end the instrumentor span if Traceloop didn't (crash fallback,
 * provider error, early consumer return — its `_endSpan` at the end of the
 * generator body never runs on those paths, leaking the span today).
 * Re-applies the last-seen usage attrs first, and never double-ends: a span
 * Traceloop already ended is left untouched (`_ended` / isRecording probe,
 * fail-open to attempting end(), which OTel treats as a diag-warned no-op).
 */
function _endGuardedStreamSpan(span: any, state: any): void {
  try {
    if (!span || typeof span.end !== "function") return;
    let ended = false;
    try {
      if ((span as any)._ended === true) ended = true;
      else if (typeof span.isRecording === "function" && !span.isRecording()) {
        ended = true;
      }
    } catch {
      ended = false;
    }
    if (ended) return;
    if (state?.lastUsage) _applyStreamUsageAttrs(span, state.lastUsage);
    span.end();
  } catch {
    // fail-open
  }
}

/** One warning per process when the accumulator crashes and we direct-drive. */
function _warnOpenAIStreamGuardFallback(err: unknown): void {
  try {
    if (_openaiStreamGuardWarnedFallback) return;
    _openaiStreamGuardWarnedFallback = true;
    logger.warning(
      "TokenPolice: the OpenAI stream instrumentor crashed while accumulating a " +
        "streamed chat response (typically an OpenAI-compatible endpoint — e.g. " +
        "Gemini — emitting an unexpected chunk shape). Your app is unaffected: " +
        "TokenPolice delivered the remaining chunks directly and metered the " +
        "stream from its usage chunk. " +
        `Cause: ${(err as Error)?.message ?? err}`,
    );
  } catch {
    // fail-open
  }
}

/**
 * The async generator the customer app receives instead of Traceloop's.
 * Drives the inner (Traceloop) generator explicitly, pairing each of its
 * yields 1:1 with the ORIGINAL chunk from the tee queue so the app receives
 * byte-identical provider objects while Traceloop accumulates sanitized ones.
 *
 * Error contract (golden rule):
 * - a rejection identity-equal to the recorded provider error is rethrown
 *   UNCHANGED (apps depend on provider errors) after defensively ending the
 *   leaked span;
 * - any other rejection is an instrumentor crash → warn once, then
 *   direct-drive the remaining chunks from the queue + real iterator so the
 *   app sees a complete, uninterrupted stream; the span is ended with the
 *   captured usage attrs;
 * - early consumer return()/throw() lands in `finally`: the inner generator
 *   and the real iterator are closed (errors swallowed) and the span ended.
 */
async function* _mkGuardedStreamGenerator(
  inner: any,
  state: any,
  span: any,
): AsyncGenerator<any, void, unknown> {
  try {
    // ── Normal drive ──
    let crashed = false;
    while (!crashed) {
      let r: IteratorResult<any>;
      try {
        r = await inner.next();
      } catch (err) {
        if (state.hasProviderError && err === state.providerError) {
          throw err; // genuine provider error — identity preserved; finally ends span
        }
        _warnOpenAIStreamGuardFallback(err);
        crashed = true;
        break;
      }
      if (r.done) return; // Traceloop ended the span itself
      const chunk = state.queue.length > 0 ? state.queue.shift() : r.value;
      _captureStreamChunkUsage(span, state, chunk);
      yield chunk;
    }
    // ── Crash fallback: direct-drive ──
    // The accumulator crashes on the inner.next() that processes the PREVIOUS
    // chunk, before the next pull — so everything already pulled sits in the
    // queue and the rest is still in the real iterator. No chunk is lost.
    while (state.queue.length > 0) {
      const chunk = state.queue.shift();
      _captureStreamChunkUsage(span, state, chunk);
      yield chunk;
    }
    if (state.realIter) {
      for (;;) {
        // A rejection here is the provider's own mid-stream error — propagate
        // (identity preserved); finally ends the span.
        const r = await state.realIter.next();
        if (r.done) break;
        _captureStreamChunkUsage(span, state, r.value);
        yield r.value;
      }
    }
  } finally {
    // Runs on normal completion, provider-error rethrow, crash-fallback
    // completion, and consumer early return()/throw(). Both closes are
    // swallow-all: cleanup must never mask or replace the in-flight result.
    try {
      const p = inner?.return?.(undefined);
      if (p && typeof p.then === "function") await p.then(() => {}, () => {});
    } catch {
      // fail-open
    }
    try {
      // The forwarder deliberately does NOT forward return() to the real
      // iterator (see installOpenAIStreamGuard) — this is the single owner
      // of real-stream cleanup.
      const p = state?.realIter?.return?.(undefined);
      if (p && typeof p.then === "function") await p.then(() => {}, () => {});
    } catch {
      // fail-open
    }
    _endGuardedStreamSpan(span, state);
  }
}

/**
 * Install the OpenAI stream guard on a freshly constructed
 * OpenAIInstrumentation INSTANCE by wrapping `_streamingWrapPromise` (a plain
 * prototype method at runtime — TS `private` is erased). One seam covers
 * auto-discovery, instrumentModules, AND nested loader-hook-patched SDK
 * copies (they all dispatch through this instance), and dodges the CJS/ESM
 * dual-build identity problem since we wrap whichever instance we built.
 *
 * Fixes two confirmed customer bugs in @traceloop/instrumentation-openai
 * (>=0.26.0 <0.28.0), which we cannot edit:
 * - Bug 1: the streamed-chat accumulator NEVER copies `chunk.usage` into its
 *   result, so streamed spans end with no gen_ai.usage.* attrs and the span
 *   processor drops them (metering lost wherever the enforcer wrapper isn't
 *   above this copy — e.g. hook-patched nested/duplicate copies). The guard
 *   captures each chunk's usage and stamps the attrs before `_endSpan`.
 * - Bug 2: the chat accumulator has no try/catch and crashes INTO CUSTOMER
 *   CODE on Gemini-shaped tool-call deltas (no `index`), chunks with no
 *   `choices`, or choices with no `delta` — and the span leaks. The guard
 *   tees the provider stream, feeds Traceloop sanitized clones (the app
 *   always receives the originals), and direct-drives the remaining chunks
 *   if the accumulator still crashes on an unknown shape.
 *
 * Unlike the Anthropic in-band guard this must NOT route around the
 * instrumentor (that would destroy the span and all metering) — here the
 * instrumentor works at dispatch time and only its accumulator is unsafe.
 *
 * Shimmer-compatible (__wrapped/__original/__unwrap) and fully fail-open: no
 * path in here may throw into init() or the customer's request path.
 */
export function installOpenAIStreamGuard(instrumentor: unknown): void {
  try {
    const inst = instrumentor as any;
    if (!inst || (typeof inst !== "object" && typeof inst !== "function")) {
      return;
    }
    let original: any;
    try {
      original = inst._streamingWrapPromise;
    } catch {
      return; // hostile getter — nothing to guard
    }
    if (typeof original !== "function") {
      logger.debug(
        () =>
          "OpenAI stream guard skipped: the instrumentor has no " +
          "_streamingWrapPromise (upstream internals changed).",
      );
      return;
    }
    if ((original as any)[OPENAI_STREAM_GUARD]) return; // idempotent

    const guarded = function tpOpenAIStreamGuard(this: any, ...args: any[]) {
      // SETUP is fully fail-open: any unexpected shape delegates verbatim.
      // (Delegating from the catch is safe — generators are lazy, so a
      // half-built inner generator was never driven and never touched the
      // provider stream.)
      try {
        const arg = args[0];
        if (!arg || typeof arg !== "object") return original.apply(this, args);
        let isChat = false;
        try {
          // The dist discriminates chat vs text-completion on this exact
          // value (GEN_AI_OPERATION_NAME_VALUE_CHAT === "chat"). The
          // text-completion branch already has its own try/catch upstream —
          // pass it (and anything unrecognized) through untouched.
          isChat = arg.type === "chat";
        } catch {
          isChat = false;
        }
        if (!isChat) return original.apply(this, args);
        const promise = arg.promise;
        if (!promise || typeof promise.then !== "function") {
          return original.apply(this, args);
        }
        const span = arg.span;
        const state: any = {
          queue: [] as any[],
          realIter: null as any,
          providerError: undefined as unknown,
          hasProviderError: false,
          lastUsage: undefined as any,
          tc: { nextIndex: 0, lastIndex: -1, seenAny: false } as _ToolCallIndexState,
        };
        // What Traceloop iterates: a forwarder over the real stream that
        // queues each ORIGINAL chunk 1:1 and hands Traceloop the original or
        // a sanitized shallow clone (never a mutation of the original).
        const forwarderIterable = {
          [Symbol.asyncIterator]() {
            return {
              next: async (): Promise<IteratorResult<any>> => {
                let r: IteratorResult<any>;
                try {
                  r = await state.realIter.next();
                } catch (err) {
                  // Genuine provider error — record identity so the outer
                  // generator rethrows it unchanged.
                  state.providerError = err;
                  state.hasProviderError = true;
                  throw err;
                }
                if (r.done) return r;
                const originalChunk = r.value;
                state.queue.push(originalChunk);
                let out = originalChunk;
                try {
                  out = _sanitizeOpenAIChunkForInstrumentor(originalChunk, state.tc);
                } catch {
                  out = originalChunk; // pass through unmodified — today's behavior
                }
                return { value: out, done: false };
              },
              // Deliberately NOT forwarded to the real iterator: tslib's
              // generator boilerplate calls return() on its source in the
              // finally that runs BEFORE an accumulator crash is rethrown —
              // forwarding would close the provider stream and defeat the
              // fallback direct-drive. _mkGuardedStreamGenerator's finally is
              // the single owner of real-iterator cleanup on every exit path.
              return: async (v?: any): Promise<IteratorResult<any>> => ({
                value: v,
                done: true,
              }),
              throw: async (e?: any): Promise<IteratorResult<any>> => {
                throw e;
              },
            } as AsyncIterator<any>;
          },
        };
        const teePromise = promise.then(
          (stream: any) => {
            try {
              if (stream && typeof stream[Symbol.asyncIterator] === "function") {
                state.realIter = stream[Symbol.asyncIterator]();
                return forwarderIterable;
              }
            } catch {
              // hostile iterator accessor — hand the stream through untouched
            }
            return stream; // non-iterable: instrumentor sees it verbatim (status quo)
          },
          (err: any) => {
            state.providerError = err;
            state.hasProviderError = true;
            throw err;
          },
        );
        // The tee is a derived promise: if the consumer abandons the
        // generator before its first pull, the rejection must never surface
        // as an unhandledRejection in the customer's process.
        try {
          teePromise.catch(() => {
            /* observed via the inner generator */
          });
        } catch {
          // fail-open
        }
        const inner = original.apply(this, [{ ...arg, promise: teePromise }]);
        if (!inner || typeof inner.next !== "function") return inner;
        return _mkGuardedStreamGenerator(inner, state, span);
      } catch {
        return original.apply(this, args);
      }
    };
    (guarded as any)[OPENAI_STREAM_GUARD] = true;
    // shimmer-compatible markers, mirroring the anthropic guard's contract.
    (guarded as any).__wrapped = true;
    (guarded as any).__original = original;
    (guarded as any).__unwrap = () => {
      try {
        if (inst._streamingWrapPromise === guarded) {
          inst._streamingWrapPromise = original;
        }
      } catch {
        /* frozen — leave as is */
      }
    };
    inst._streamingWrapPromise = guarded;
    logger.debug(
      () =>
        "TokenPolice: installed OpenAI stream guard on the instrumentor " +
        "instance (chunk sanitization + streamed usage capture).",
    );
  } catch {
    // fail-open: guard install must never break init()
  }
}

/** Minimal structural view of the upstream InstrumentationBase internals. */
type InstrumentationModuleDef = {
  name?: string;
  patch?: (...args: any[]) => any;
  unpatch?: (...args: any[]) => any;
  files?: InstrumentationModuleDef[];
  supportedVersions?: string[];
  moduleExports?: unknown;
};

/**
 * Wraps every `patch`/`unpatch` on an instrumentor's module definitions so a
 * throw inside third-party patch code can never escape into the customer's
 * `require()`/`import` of the provider SDK.
 *
 * Load-bearing constraints:
 * - require-in-the-middle / import-in-the-middle hooks are registered in the
 * InstrumentationBase CONSTRUCTOR, not in `enable()` — wrapping `enable()`
 * would fix nothing, and hardening must run immediately after `new`.
 * - The hook closure reads `.patch` off the LIVE `_modules` objects at call
 * time, so mutating them post-construction is effective. Never use
 * `getModuleDefinitions()` — it re-runs `init()` and returns NEW objects.
 * - On failure the wrapper MUST return the original `exports` so the
 * customer's require/import always completes with an unpatched (un-metered)
 * module rather than crashing the app.
 */
export function hardenInstrumentorPatches(
  instrumentor: unknown,
  label: string,
): void {
  try {
    const modules = (instrumentor as { _modules?: unknown } | null | undefined)
      ?._modules;
    if (!Array.isArray(modules)) {
      logger.debug(
        () => `Loader-hook hardening skipped for ${label}: no _modules array`,
      );
      return;
    }

    const wrap = (def: InstrumentationModuleDef, key: "patch" | "unpatch") => {
      const original = def?.[key];
      if (typeof original !== "function") return;
      if ((original as any)[HARDENED_PATCH]) return;

      const hardened = function (this: any, ...args: any[]) {
        try {
          // Patch idempotency by module identity: a module already patched via
          // manuallyInstrument() must NOT be re-patched by the loader hook — a
          // second layer would stack ABOVE the enforcer wrapper and break its
          // layering guarantees (see markManuallyInstrumented). Only `patch` is
          // deduped; `unpatch` must always run. Identity miss → patch as usual.
          if (key === "patch" && _isManuallyInstrumentedModule(instrumentor, args[0])) {
            try {
              logger.debug(
                () =>
                  `TokenPolice: ${label} loader hook skipped re-patch of a module ` +
                  `already instrumented via instrumentModules (identity match).`,
              );
            } catch { /* fail-open */ }
            return args[0];
          }
          const result = original.apply(this, args);
          // The anthropic loader hook also patches NESTED/duplicate copies of
          // @anthropic-ai/sdk that the enforcer never wraps. On IN-BAND copies
          // (patch succeeds, streaming would throw at call time) install a
          // stream guard so streamed calls can't crash customer code. Runs
          // ONLY after a successful patch; internally fail-open.
          if (key === "patch" && label === "anthropic" && def?.name === "@anthropic-ai/sdk") {
            installAnthropicInBandStreamGuard(args[0]);
          }
          return result;
        } catch (err) {
          // Logging runs on the customer's require() stack — guard it too.
          try {
            logger.warning(
              `TokenPolice: ${label} instrumentor ${key}() failed for ` +
                `${def?.name ?? "unknown module"} — calls to this provider will ` +
                `NOT be metered, but your app is unaffected. ` +
                `Cause: ${(err as Error)?.message ?? err}`,
            );
          } catch { /* fail-open: silent */ }
          // Return the module exports untouched so the caller's require/import
          // completes normally.
          return args[0];
        }
      };
      (hardened as any)[HARDENED_PATCH] = true;
      def[key] = hardened;
    };

    for (const def of modules as InstrumentationModuleDef[]) {
      if (!def || typeof def !== "object") continue;
      wrap(def, "patch");
      wrap(def, "unpatch");
      // Nested InstrumentationNodeModuleFile entries carry their own
      // patch/unpatch, invoked on the same loader-hook path.
      if (Array.isArray(def.files)) {
        for (const file of def.files) {
          if (!file || typeof file !== "object") continue;
          wrap(file, "patch");
          wrap(file, "unpatch");
        }
      }
    }
  } catch (err) {
    // Hardening is best-effort; an unexpected internal shape must never break
    // init(). Worst case we're back to the unhardened upstream behavior.
    logger.debug(
      () =>
        `Loader-hook hardening failed for ${label}: ${(err as Error)?.message ?? err}`,
    );
  }
}

// ── Version-gate probe ───────────────────────────────────────────

/**
 * `_warnedProviders` key prefix for the version-gate notice below. Namespaced
 * so it can never collide with a real INSTRUMENTOR_REGISTRY moduleKey.
 */
const VERSION_GATE_WARN_KEY_PREFIX = "versionGate:";

/**
 * Modules whose version gate accepted a copy at some point in this process.
 * NEVER cleared on teardown (like `_warnedProviders`): after unsetup + a second
 * init(), require-in-the-middle's per-module cache short-circuits before the
 * NEW instrumentor's hook, so its `moduleExports` stays unset even though the
 * version is fine. This memory is the only thing that tells those two causes
 * apart — clearing it would misreport re-init as an unsupported version.
 */
const _versionGateAccepted = new Set<string>();

/**
 * Best-effort installed version of a target SDK, for the notice below.
 *
 * `require("<pkg>/package.json")` alone is not enough: modern packages
 * (`openai` among them) omit `./package.json` from their `exports` map, so that
 * require throws ERR_PACKAGE_PATH_NOT_EXPORTED. The fallback walks up from the
 * resolved entry file to the owning manifest — checking `name` at every level,
 * because packages also ship stub manifests (`dist/package.json` carrying only
 * `{"type": ...}`) that would otherwise be mistaken for the real one.
 *
 * Purely cosmetic: returns undefined on any failure and never throws.
 */
export function resolveInstalledVersion(
  req: NodeRequire,
  targetModule: string,
): string | undefined {
  const asVersion = (value: unknown): string | undefined =>
    typeof value === "string" && value ? value : undefined;

  try {
    const direct = asVersion((req(`${targetModule}/package.json`) as any)?.version);
    if (direct) return direct;
  } catch { /* not exported — fall through to the directory walk */ }

  try {
    let dir = dirname(req.resolve(targetModule));
    // Bounded: the manifest sits a handful of levels above the entry file, and
    // an unbounded walk would climb out of node_modules to the filesystem root.
    for (let depth = 0; depth < 10; depth++) {
      try {
        const manifest = req(join(dir, "package.json")) as any;
        if (manifest?.name === targetModule) return asVersion(manifest.version);
      } catch { /* no readable manifest at this level — keep climbing */ }
      const parent = dirname(dir);
      if (parent === dir) break;
      dir = parent;
    }
  } catch { /* unresolvable — no version to report */ }

  return undefined;
}

/**
 * Warns once when `targetModule` ended up UNinstrumented, for either of the two
 * causes that leave no trace anywhere else:
 * - the loader hook ran but its semver gate REJECTED the installed copy (openai
 * 7 against an instrumentor declaring ">=4 <7"). Upstream is silent about
 * this at every diag level.
 * - unsetup + a second init(): require-in-the-middle's per-module cache
 * short-circuits before the freshly constructed instrumentor's hook, so the
 * module is left unpatched with no way to re-attach in this process.
 * `_versionGateAccepted` distinguishes them; either way metering is dead.
 *
 * Load-bearing constraint: in `InstrumentationBase._onRequire`, a def's
 * `moduleExports` is assigned IFF the module name matched AND
 * `isSupported(supportedVersions, version)` passed — before the `_enabled`
 * check and before `patch` runs. So "a matching def has `moduleExports` set"
 * ⇔ "the version gate accepted the installed copy", and we need no semver
 * parsing of our own. Only valid AFTER the module has been required at least
 * once; no matching def means the instrumentor declares no definition for this
 * target at all, so we cannot assess it and must NOT warn.
 */
export function warnIfInstrumentorSkippedModule(
  instrumentor: unknown,
  targetModule: string,
  getInstalledVersion?: () => string | undefined,
): void {
  try {
    const modules = (instrumentor as { _modules?: unknown } | null | undefined)
      ?._modules;
    if (!Array.isArray(modules)) return;

    const defs = (modules as InstrumentationModuleDef[]).filter(
      (def) => def && typeof def === "object" && def.name === targetModule,
    );
    if (defs.length === 0) return;
    if (defs.some((def) => def.moduleExports !== undefined)) {
      _versionGateAccepted.add(targetModule);
      return;
    }

    const warnKey = `${VERSION_GATE_WARN_KEY_PREFIX}${targetModule}`;
    if (_warnedProviders.has(warnKey)) return;
    _warnedProviders.add(warnKey);

    if (_versionGateAccepted.has(targetModule)) {
      logger.warning(
        `token capture for ${targetModule} stopped after re-initialization — the loader hook ` +
          `cannot re-attach in this process; calls to this provider will NOT be metered and ` +
          `budget counters will not accumulate. Pre-flight blocking of already-tripped rules ` +
          `still runs. Restart the process to restore metering.`,
      );
      return;
    }

    let installed = "";
    try {
      const version = getInstalledVersion?.();
      if (typeof version === "string" && version) installed = ` ${version}`;
    } catch { /* fail-open: the version is cosmetic */ }

    const ranges = defs
      .flatMap((def) =>
        Array.isArray(def.supportedVersions) ? def.supportedVersions : [],
      )
      .filter((range) => typeof range === "string" && range);
    const supported = ranges.length
      ? ` (its bundled instrumentor supports ${ranges.join(", ")})`
      : "";

    logger.warning(
      `token capture for ${targetModule} is disabled — the installed version${installed} is ` +
        `not yet supported${supported}; calls to this provider will NOT be metered and budget ` +
        `counters will not accumulate. Pre-flight blocking of already-tripped rules still ` +
        `runs. Pin ${targetModule} to a supported version until token-police adds support.`,
    );
  } catch (err) {
    logger.debug(
      () =>
        `Version-gate probe failed for ${targetModule}: ${(err as Error)?.message ?? err}`,
    );
  }
}

// ── Manual instrumentation (instrumentModules mode) ──────────────

/**
 * Reads one user-supplied `instrumentModules` key, tolerating case drift.
 *
 * What this guarantees is RECOGNITION parity, not precedence parity: after this
 * fix both seams accept the same SET of spellings for a key. They can still
 * disagree on which VALUE wins if an app passes two case-variants of one key
 * with DIFFERENT values — the enforcer's prep loop is
 * `for (const [key, mod] of Object.entries(instrumentModules))`, i.e.
 * source-order last-wins, while this reader is exact-first. So
 * `{ openai: OpenAI, openAI: undefined }` leaves the enforcer with an
 * unresolvable entry (no wrap) while this reader falls through to `openai` and
 * DOES instrument: metered but not enforced — fail-open, and strictly more
 * coverage than pre-fix, where that input produced neither seam. TypeScript
 * rejects it and it is not worth a behavior change; do not "fix" it by making
 * the prep loop exact-first or by dropping the fall-through below.
 *
 * The recognition gap this closes: enforcer.ts's
 * autoInstrument() prep loop lowercases EVERY key (`key.toLowerCase()`), and
 * the LangChain opt-in gate in _applyLangChainInstrumentation matches
 * case-insensitively too — but the manual-instrumentation lookup read
 * `modules[entry.moduleKey]` with EXACT case. So `{ openai: OpenAI }`
 * (lowercase) still patched the provider prototypes — /check fired, and that
 * also silenced client.ts's zero-wrap audit, so nothing warned — while the
 * Traceloop OpenAI instrumentor was never constructed. Auto-discovery could
 * not cover for it either, since `instrumentModules` was non-empty. The
 * `Chat.Completions` / `Completions` enforcer targets carry NO
 * `manualTelemetry`, so the whole of OpenAI metering hung off that Traceloop
 * span: the customer got an enforcing firewall and zero generation rows.
 *
 * Precedence — do NOT "simplify" this into a plain lowercase-only lookup,
 * which would silently change which value wins for an app passing both
 * spellings:
 * 1. EXACT match first. Every correctly-spelled app stays on a byte-for-byte
 *    identical path, and `{ openAI, openai }` together resolves to the
 *    canonical key — exactly one instrumentation, never two.
 * 2. Otherwise the FIRST readable key whose `toLowerCase()` matches, in
 *    `Object.keys()` order (a key whose getter throws is skipped, not fatal),
 *    so a duplicate-cased object always resolves the same way.
 *
 * This is a safety net for JS / tsx / no-typecheck callers, NOT a second
 * advertised API: `InstrumentModules` still declares one canonical spelling
 * per provider and TypeScript still rejects a misspelling at compile time.
 *
 * Every read is guarded. `modules` is customer input: it can be a revoked
 * Proxy, a null-prototype object, or carry hostile getters — and the old
 * exact-case read sat OUTSIDE _applyManualInstrumentations' per-entry try, so
 * a throwing getter escaped the entire function and killed metering for every
 * provider behind it. Returns undefined instead of throwing (golden rule).
 */
function _readModuleEntry(modules: any, key: string): any {
  try {
    if (
      !modules ||
      (typeof modules !== "object" && typeof modules !== "function")
    ) {
      return undefined;
    }
    try {
      // Present-but-undefined deliberately falls through to the scan below:
      // the callers all treat undefined as "not passed", so an app that wrote
      // `{ openAI: undefined, openai: OpenAI }` still gets instrumented.
      const exact = modules[key];
      if (exact !== undefined) return exact;
    } catch {
      /* hostile getter on the canonical key — fall through to the scan */
    }
    const target = key.toLowerCase();
    for (const k of Object.keys(modules)) {
      if (k === key) continue; // already attempted above
      if (typeof k !== "string" || k.toLowerCase() !== target) continue;
      try {
        return modules[k];
      } catch {
        /* hostile getter — treat this key as absent and KEEP SCANNING. An
           early return here would abandon a later, perfectly readable
           case-variant (e.g. { OpenAI: <throwing getter>, openai: OpenAI }),
           leaving the app enforced-but-unmetered through the very guard that
           exists to prevent it. */
        continue;
      }
    }
  } catch {
    /* revoked Proxy / exotic object — treat as "no such key" */
  }
  return undefined;
}

/**
 * Normalizes a user-supplied `openAI` module value to the shape
 * @traceloop/instrumentation-openai's manuallyInstrument() reads.
 *
 * That function's body is, in effect (0.26; paraphrased, not copied):
 * this._wrap(openaiModule.Chat.Completions.prototype, "create", …)
 * this._wrap(openaiModule.Completions.prototype, "create", …)
 * if (openaiModule.Images) { …Images.prototype… }
 * — i.e. it wants the `OpenAI` CLASS: `Chat`/`Completions` are static resource
 * classes on the class, not properties of the module namespace. A namespace
 * would throw "Cannot read properties of undefined (reading 'Completions')"
 * and metering for OpenAI would be silently lost.
 *
 * Feature-detected (never version-detected): a candidate is accepted only when
 * BOTH prototypes the instrumentor unconditionally touches are present, so we
 * can never hand it a partially-matching object that throws halfway through and
 * leaves a half-patched module behind. On no match, or on any throw (module
 * objects can carry hostile getters — see markManuallyInstrumented), the input
 * is returned VERBATIM so behaviour degrades to the pre-normalization attempt.
 * Never throws — golden rule.
 */
function _pickOpenAIClass(mod: any): any {
  try {
    const candidates = [mod, mod?.OpenAI, mod?.default, mod?.default?.OpenAI];
    for (const c of candidates) {
      if (typeof c !== "function") continue;
      if (!c?.Chat?.Completions?.prototype) continue;
      if (!c?.Completions?.prototype) continue;
      return c;
    }
  } catch {
    /* hostile getter — fall through to verbatim */
  }
  return mod;
}

/**
 * Normalizes a user-supplied `anthropic` module value to the shape
 * @traceloop/instrumentation-anthropic's manuallyInstrument() reads.
 *
 * That function's body is, in effect (0.27; paraphrased, not copied):
 * this._wrap(module.Anthropic.Completions.prototype, "create", …)
 * this._wrap(module.Anthropic.Messages.prototype, "create", …)
 * this._wrap(module.Anthropic.Beta.Messages.prototype, "create", …)
 * — i.e. it wants the module NAMESPACE (an object with `.Anthropic`), the exact
 * OPPOSITE of the openAI entry above. This is why each registry entry carries
 * its OWN normalizer and why `_pickClassExport` (enforcer.ts) must NOT be reused
 * here: it would collapse a working namespace down to the class and break every
 * app that passes `import * as Anthropic from "@anthropic-ai/sdk"`.
 *
 * The class-form case is the one this exists for: `@anthropic-ai/sdk`'s DEFAULT
 * export IS the `Anthropic` class, so `import Anthropic from "@anthropic-ai/sdk"`
 * (which the docs used to recommend) yields a value whose `.Anthropic` is the
 * inherited `BaseAnthropic` static self-ref — and BaseAnthropic does NOT carry
 * `.Messages`. manuallyInstrument() therefore throws "Cannot read properties of
 * undefined (reading 'prototype')" and ALL Anthropic telemetry is lost: the
 * enforcer's `["Anthropic","Messages","prototype"]` target has no
 * `manualTelemetry` fallback, so it depends on the Traceloop span.
 *
 * For that case we synthesize a namespace with `Object.create(cls)` so the
 * class's own statics (error classes, toFile, …) stay reachable by prototype
 * chain, and shadow the inherited `Anthropic` self-ref with the real class.
 *
 * All three prototypes are required before accepting a candidate because the
 * instrumentor touches all three unconditionally — a partial shape would throw
 * mid-way and leave the module half-patched.
 *
 * Returns the input VERBATIM on no match or on any throw. Never throws.
 */
function _pickAnthropicNamespace(mod: any): any {
  try {
    // Already namespace-shaped (the common, working case: `import * as …`).
    // Returned by identity so this path stays byte-for-byte what it was.
    for (const c of [mod, mod?.default]) {
      if (c === null || c === undefined) continue;
      if (!c?.Anthropic?.Completions?.prototype) continue;
      if (!c?.Anthropic?.Messages?.prototype) continue;
      if (!c?.Anthropic?.Beta?.Messages?.prototype) continue;
      return c;
    }
    // Class form: the default export IS the Anthropic class, carrying the
    // resource statics directly. Wrap it in a namespace-shaped view.
    if (
      typeof mod === "function" &&
      mod?.Completions?.prototype &&
      mod?.Messages?.prototype &&
      mod?.Beta?.Messages?.prototype
    ) {
      // Object.create(mod) — NOT a fresh object — so the synthesized namespace
      // still exposes everything else hanging off the class. `.Anthropic` is
      // then set as an OWN property to shadow the inherited BaseAnthropic
      // self-ref that would otherwise be walked into.
      const ns: any = Object.create(mod);
      ns.Anthropic = mod;
      return ns;
    }
  } catch {
    /* hostile getter — fall through to verbatim */
  }
  return mod;
}

/**
 * Applies instrumentations using user-provided module references.
 * This solves the ESM import ordering problem — users import the module
 * at the top of their file, then pass it to init().
 *
 * Instrumentors are created, set to use our private provider, and
 * manuallyInstrument() is called with the user-provided module.
 */
function _applyManualInstrumentations(
  provider: BasicTracerProvider,
  modules: InstrumentModules,
): void {
  let appRequire: NodeRequire;
  try {
    appRequire = createRequire(join(process.cwd(), "node_modules"));
  } catch {
    appRequire = require;
  }

  for (const entry of INSTRUMENTOR_REGISTRY) {
    const userModule = _readModuleEntry(modules, entry.moduleKey);
    if (!userModule) continue;

    try {
      const mod = resolveInstrumentor(appRequire, entry.packageName);
      if (!mod) continue;
      const InstrumentorClass = mod[entry.className] || mod.default;
      if (!InstrumentorClass) continue;

      const instrumentor = new InstrumentorClass();

      // The constructor already registered the loader hooks — harden before
      // anything else touches the instrumentor.
      hardenInstrumentorPatches(instrumentor, entry.moduleKey);

      // Guard the OpenAI streaming accumulator (crash + lost usage) at the
      // instance seam — covers this manual patch AND any loader-hook copies.
      if (entry.moduleKey === "openAI") {
        installOpenAIStreamGuard(instrumentor);
      }

      // Bind to our private provider so spans flow through our processor
      if (typeof instrumentor.setTracerProvider === "function") {
        instrumentor.setTracerProvider(provider);
      }

      // enable() sets up require-in-the-middle hooks for future loads
      if (typeof instrumentor.enable === "function") {
        instrumentor.enable();
      }

      // Patch the already-imported module directly
      if (typeof instrumentor.manuallyInstrument === "function") {
        // Coerce to the shape THIS instrumentor reads (class vs namespace —
        // see `normalizeManualTarget` on the registry entry). Without this, an
        // app that passes the `@anthropic-ai/sdk` default export (the Anthropic
        // class) makes manuallyInstrument() throw on
        // `module.Anthropic.Messages.prototype` and loses ALL Anthropic
        // telemetry — that target has no manualTelemetry fallback in the
        // enforcer. Returns the input verbatim when it already matches or is
        // unrecognized, so the namespace-import path is unchanged.
        const target = entry.normalizeManualTarget(userModule);
        instrumentor.manuallyInstrument(target);
        // The module was loaded BEFORE the loader hook armed, so the hook's
        // per-file cache misses it — the app's next require() of the same
        // module would stack a second patch layer above the enforcer wrapper
        // (breaking anthropicStreamBypass and double-counting spans). Record
        // its identity so the hardened hook skips that re-patch. Reached only
        // when manuallyInstrument() did not throw — an unpatched module is
        // never marked.
        //
        // BOTH identities are marked: `target` is what actually got patched,
        // but `userModule` is the identity the loader hook is likelier to see
        // (it's the value the app imported). Marking is a Set add — cheap, and
        // a miss would only mean the hook re-patches as it does today.
        markManuallyInstrumented(instrumentor, userModule);
        if (target !== userModule) {
          markManuallyInstrumented(instrumentor, target);
        }

        // Anthropic-only. The hazard is the Traceloop layer's streaming
        // branch: it runs `new moduleExports.APIPromise(...)`, which throws
        // synchronously INTO CUSTOMER CODE whenever the patched exports object
        // has no root `APIPromise`. The enforcer's anthropicStreamBypass
        // absorbs that on `Messages` and `Beta.Messages`, but there is NO
        // enforcer target for `Anthropic.Completions` at all — so that one
        // surface is unprotected. This installer covers it, and it does two
        // distinct jobs depending on what the app passed:
        //
        // - CLASS form: the normalization above is what makes the Traceloop
        //   layer exist here at all (previously manuallyInstrument() threw, so
        //   there was no layer and streamed completions.create() ran clean but
        //   UNMETERED). The synthesized namespace has no `APIPromise`, so
        //   without this guard the normalization would INTRODUCE a crash. With
        //   it, streamed completions.create() routes to the raw method: still
        //   unmetered, exactly as before, and still not crashing.
        // - NAMESPACE form on an older SDK (≤0.41, no root `APIPromise` — e.g.
        //   0.30.1, which most of the verification fleet pins): the Traceloop
        //   layer was ALREADY being installed here, with no enforcer target
        //   above it, so `completions.create({stream: true})` crashes with
        //   "APIPromise is not a constructor" TODAY. This is a latent
        //   production crash, and installing the guard fixes it.
        //
        // The guard self-gates on _isInBandAnthropicExports, so it is a no-op
        // whenever a real APIPromise is present (i.e. every namespace-import
        // app on a modern SDK) — that path stays byte-for-byte unchanged.
        if (entry.moduleKey === "anthropic") {
          installAnthropicInBandStreamGuard(target);
        }
      }

      _activeInstrumentations.push(instrumentor);
      logger.debug(
        `Instrumented ${entry.moduleKey} via ${entry.packageName} (manual)`,
      );
    } catch (err) {
      // A missing instrumentor package is already handled by `continue` above
      // (resolveInstrumentor returns undefined), so reaching here means the
      // instrumentor threw while patching — typically a value that is neither
      // the SDK's module namespace nor its default export (e.g. a client
      // INSTANCE, or a re-export barrel), which normalizeManualTarget above
      // cannot recognize and passes through verbatim. The remedy is
      // deliberately shape-agnostic: both forms are accepted per provider, and
      // naming one specific form here was wrong for anthropic (its instrumentor
      // needs the namespace, not the default export).
      // This silently disables metering for the provider, so warn loudly.
      logger.warning(
        `TokenPolice: failed to instrument ${entry.moduleKey} via ${entry.packageName} — ` +
          `calls to this provider will NOT be metered. ` +
          `Pass the imported module itself to instrumentModules (either the ` +
          `namespace import or the default export), not a client instance. ` +
          `Cause: ${(err as Error)?.message ?? err}`,
      );
    }
  }
}

// ── Auto-discovery (no instrumentModules) ────────────────────────

/**
 * Auto-discovers installed @traceloop/instrumentation-* packages from the
 * application's node_modules and registers them.
 *
 * Uses createRequire(process.cwd()) to resolve packages from the caller's
 * context, not the SDK's dist/ directory.
 */
function _applyAutoDiscoveredInstrumentations(
  provider: BasicTracerProvider,
): void {
  let appRequire: NodeRequire;
  try {
    appRequire = createRequire(join(process.cwd(), "node_modules"));
  } catch {
    appRequire = require;
  }

  for (const entry of INSTRUMENTOR_REGISTRY) {
    try {
      const mod = resolveInstrumentor(appRequire, entry.packageName);
      if (!mod) {
        // The instrumentor ships as an optionalDependency, so it should
        // normally be present. Warn ONLY if the app actually uses this
        // provider (its SDK resolves) but the instrumentor didn't install —
        // otherwise an OpenAI-only app would get a spurious Anthropic warning.
        // Note this layer cannot know whether the enforcer's own wraps landed
        // (that's audited separately at init() — see the zero-wrap warning in
        // client.ts), so make no claim about enforcement here.
        if (
          canResolve(appRequire, entry.targetModule) &&
          !_warnedProviders.has(entry.moduleKey)
        ) {
          _warnedProviders.add(entry.moduleKey);
          logger.warning(
            `token capture for ${entry.moduleKey} is disabled — its bundled instrumentor ` +
              `didn't load (optional dependencies may have been skipped at install); ` +
              `reinstall token-police to enable it. Until it loads, ${entry.moduleKey} ` +
              `calls may be missing token usage and cost data.`,
          );
        }
        continue;
      }
      const InstrumentorClass = mod[entry.className] || mod.default;
      if (!InstrumentorClass) continue;

      const instrumentor = new InstrumentorClass();

      // The constructor already registered the loader hooks — harden before
      // anything else touches the instrumentor.
      hardenInstrumentorPatches(instrumentor, entry.moduleKey);

      // Guard the OpenAI streaming accumulator (crash + lost usage) at the
      // instance seam — covers eager-manual, hook-patched, and nested copies.
      if (entry.moduleKey === "openAI") {
        installOpenAIStreamGuard(instrumentor);
      }

      // Bind to our private provider
      if (typeof instrumentor.setTracerProvider === "function") {
        instrumentor.setTracerProvider(provider);
      }

      // Hook future require() calls
      if (typeof instrumentor.enable === "function") {
        instrumentor.enable();
      }

      // Fallback: if target SDK is already loaded, patch it now.
      // The target SDK is the APP's own copy, so resolve it from cwd only.
      let targetMod: any;
      let targetInstalled = false;
      try {
        targetMod = appRequire(entry.targetModule);
        targetInstalled = true;
      } catch {
        // Target SDK not installed — enable() hook will catch future loads
      }

      if (targetInstalled) {
        let eagerManualSucceeded = false;
        if (typeof instrumentor.manuallyInstrument === "function") {
          try {
            instrumentor.manuallyInstrument(targetMod);
            eagerManualSucceeded = true;
            logger.debug(
              `Instrumented ${entry.targetModule} via ${entry.packageName} (auto)`,
            );
          } catch (err) {
            // Not every instrumentor accepts the module NAMESPACE `require`
            // returns (openai's expects the `.Chat` shape of the default
            // export). Not fatal: the require above already ran the loader
            // hook, which patches the real copy — so debug, never warn.
            logger.debug(
              () =>
                `Eager manual instrumentation of ${entry.targetModule} failed: ` +
                `${(err as Error)?.message ?? err}`,
            );
          }
        }

        // The require above drove the loader hook for this copy, so the module
        // defs now carry the version gate's verdict. A successful
        // manuallyInstrument bypasses that gate entirely — the provider IS
        // metered regardless of its verdict — so probing would be a false alarm.
        if (!eagerManualSucceeded) {
          warnIfInstrumentorSkippedModule(instrumentor, entry.targetModule, () =>
            resolveInstalledVersion(appRequire, entry.targetModule),
          );
        }
      }

      _activeInstrumentations.push(instrumentor);
    } catch {
      // Instrumentor package not installed — skip silently
    }
  }
}

// ── LangChain instrumentation (special-cased) ────────────────────
//
// @traceloop/instrumentation-langchain's manuallyInstrument signature is
// `manuallyInstrument({ callbackManagerModule })` — not the single-module
// shape every other instrumentor uses. Its init() returns []` so enable()
// alone does NOT register require-in-the-middle hooks; we MUST call
// manuallyInstrument with @langchain/core/callbacks/manager.

function _applyLangChainInstrumentation(
  provider: BasicTracerProvider,
  instrumentModules?: InstrumentModules,
): void {
  let appRequire: NodeRequire;
  try {
    appRequire = createRequire(join(process.cwd(), "node_modules"));
  } catch {
    appRequire = require;
  }

  // LangChain stays opt-in (heavy install), so a missing instrumentor is
  // expected — skip silently rather than warn.
  //
  // Resolve the TokenPolice-branded companion package `token-police-langchain`
  // first: customers install it instead of naming the third-party instrumentor
  // in their own package.json. It re-exports `LangChainInstrumentation` and
  // carries `@traceloop/instrumentation-langchain` as its own dependency, so it
  // resolves even under pnpm / yarn-PnP (the app depends on the companion
  // directly; the companion resolves the instrumentor from its own
  // node_modules). Installing the companion IS the opt-in, so it always wins.
  //
  // Falling back to the raw `@traceloop/instrumentation-langchain` stays as
  // back-compat for apps that installed it directly — but ONLY in
  // auto-discovery mode. In instrumentModules (manual) mode the integrator has
  // enumerated exactly what to instrument, so a raw instrumentor that merely
  // happens to be resolvable transitively (e.g. pulled in by
  // @traceloop/node-server-sdk) must NOT be applied: its constructor globally
  // patches LangChain's CallbackManager (first-patcher-wins), which would meter
  // LangChain without opt-in and can hijack the customer's own Traceloop
  // tracing. So in manual mode we require an explicit `langChain` key (matched
  // case-insensitively, mirroring the enforcer's key normalization) and never
  // even load the raw package otherwise.
  let pkg = resolveInstrumentor(appRequire, "token-police-langchain");
  if (!pkg) {
    const manualMode =
      !!instrumentModules && Object.keys(instrumentModules).length > 0;
    const langChainKeyPassed =
      !!instrumentModules &&
      Object.keys(instrumentModules).some(
        (k) => k.toLowerCase() === "langchain",
      );
    if (manualMode && !langChainKeyPassed) {
      // Warn only when the raw instrumentor is actually reachable — i.e. when
      // this gate really changed the outcome. `canResolve` only resolves a
      // path; it never loads or constructs the module.
      if (
        canResolve(appRequire, "@traceloop/instrumentation-langchain") &&
        !_warnedProviders.has(LANGCHAIN_OPT_IN_WARN_KEY)
      ) {
        _warnedProviders.add(LANGCHAIN_OPT_IN_WARN_KEY);
        logger.warning(
          `@traceloop/instrumentation-langchain was not applied: instrumentModules mode ` +
            `requires explicit opt-in for LangChain, and no langChain key was passed. ` +
            `Install token-police-langchain, or pass instrumentModules.langChain, to enable ` +
            `LangChain metering. Budgets are still enforced.`,
        );
      }
      return;
    }
    pkg = resolveInstrumentor(appRequire, "@traceloop/instrumentation-langchain");
  }
  if (!pkg) return; // Companion / instrumentor not installed — skip
  const InstrumentorClass = pkg.LangChainInstrumentation || pkg.default;
  if (!InstrumentorClass) return;

  // Correct JS tool-span names (DynamicStructuredTool → real tool name).
  // Patch the shared callback-handler prototype before instrumentation runs.
  try {
    if ((pkg as any).TraceloopCallbackHandler) {
      patchLangChainToolNaming((pkg as any).TraceloopCallbackHandler);
    }
  } catch {
    // fail-open
  }

  let instrumentor: any;
  try {
    instrumentor = new InstrumentorClass();
    // No-op today (LangChain's init() returns []), but kept uniform so a
    // future version that registers loader hooks is covered too.
    hardenInstrumentorPatches(instrumentor, "langChain");
    if (typeof instrumentor.setTracerProvider === "function") {
      instrumentor.setTracerProvider(provider);
    }
    if (typeof instrumentor.enable === "function") {
      instrumentor.enable();
    }
  } catch {
    return;
  }

  // @langchain/core/callbacks/manager exists as TWO objects: a CJS copy and
  // an ESM copy (distinct module records). The instrumentor's constructor
  // patches both, but resolves them from its OWN package location — which
  // can pull a nested duplicate (`node_modules/@traceloop/.../node_modules/
  // @langchain/core/...`) instead of the app's top-level copy that
  // `@langchain/openai` etc. actually use. We re-patch using the modules
  // resolved from the APP's perspective.
  const tryPatch = (mod: any) => {
    if (!mod) return;
    try {
      if (typeof instrumentor.manuallyInstrument === "function") {
        instrumentor.manuallyInstrument({ callbackManagerModule: mod });
      }
    } catch {
      /* fail-safe */
    }
  };

  // 1. User-provided module (recommended). Accept either:
  // - { callbackManagerModule, chatModelsModule } object — use the
  // callbackManagerModule sub-field.
  // - the bare module (back-compat) — treat as the callback manager module.
  // Case-insensitive fallback (exact `langChain` still wins) so this
  // "recommended, guaranteed first-call coverage" path is not lost on a
  // lowercase `langchain` key — the opt-in gate above already matched that
  // key case-insensitively, so without this the app passed the gate and then
  // silently fell through to the require()/import() fallbacks.
  const userLangChain = _readModuleEntry(instrumentModules, "langChain");
  if (userLangChain) {
    if (typeof userLangChain === "object" && userLangChain.callbackManagerModule) {
      tryPatch(userLangChain.callbackManagerModule);
    } else if (typeof userLangChain === "object" && "CallbackManager" in userLangChain) {
      tryPatch(userLangChain);
    }
  }

  // 2. CJS top-level from the app's node_modules, in case the app is CJS.
  try {
    tryPatch(appRequire("@langchain/core/callbacks/manager"));
  } catch {
    /* not installed */
  }

  // 3. ESM top-level via dynamic bare-specifier import. Async & best-effort —
  // covers ESM apps that did not pass the module via instrumentModules.
  // langChain.callbackManagerModule. May settle after the first LLM call,
  // in which case that call's span is dropped — apps that need
  // first-call coverage should pass the module via instrumentModules.
  (async () => {
    try {
      // @ts-ignore — optional peer dep; not part of the SDK's bundled types.
      const mod = await import("@langchain/core/callbacks/manager");
      tryPatch(mod);
    } catch {
      /* fail-safe */
    }
  })();

  _activeInstrumentations.push(instrumentor);
  logger.debug(
    "Instrumented langChain via @traceloop/instrumentation-langchain",
  );
}

// ── Teardown ──────────────────────────────────────────────────────

/**
 * Uninstalls OpenTelemetry instrumentors and shuts down the provider.
 */
export function unsetupOpenTelemetry(): void {
  if (!_isSetup) return;

  for (const instrumentor of _activeInstrumentations) {
    try {
      if (typeof instrumentor.disable === "function") {
        instrumentor.disable();
      }
    } catch {
      // ignore
    }
  }
  _activeInstrumentations = [];

  if (_tracerProvider) {
    _tracerProvider.shutdown().catch(() => {});
    _tracerProvider = undefined;
  }

  // Undo the global registrations we (and only we) performed in setup, so a
  // later re-init starts clean and no stale delegate points at our shut-down
  // provider. In piggyback mode these flags are false — we never touch the
  // customer's globals. Each disable is guarded so teardown never throws.
  if (_registeredGlobalProvider) {
    try {
      trace.disable();
    } catch {
      // ignore
    }
    try {
      // We install no propagator (see setup), but SDK 1.x's register() did;
      // resetting it in lockstep keeps teardown idempotent across versions.
      propagation.disable();
    } catch {
      // ignore
    }
    _registeredGlobalProvider = false;
  }
  if (_registeredGlobalContextManager) {
    try {
      context.disable();
    } catch {
      // ignore
    }
    _registeredGlobalContextManager = false;
  }

  _isSetup = false;
}
