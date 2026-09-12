/**
 * Session context propagation via Node.js AsyncLocalStorage.
 * Allows users to attach user_id, paid_plan, workflow_name, and metadata to all LLM calls
 * within a callback or decorator block, without passing them explicitly.
 *
 * Behavior is kept in parity with the TokenPolice Python SDK.
 * Uses AsyncLocalStorage (the Node.js equivalent of Python's contextvars.ContextVar).
 */

import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID, randomBytes } from "node:crypto";
import { trace, SpanStatusCode, type Tracer, type Span as OtelSpan } from "@opentelemetry/api";
import { getClient } from "./state";
import { scrubErrorMessage, resolveErrorDetail, toDurationMs, safeMonoNow } from "./_classify";
import { hashLen } from "./composition";
import { copySessionMetadata } from "./routingMarkerStore";

// ── W3C-format id helpers ─────────────────────────────────────────
// W3C Trace Context: trace_id = 32 lowercase hex chars (128-bit),
// span_id = 16 lowercase hex chars (64-bit). We mint these directly so
// the emitted span model matches the OpenTelemetry / W3C wire format even
// on paths that have no live OTel span (manual-telemetry providers).
/** 16 lowercase hex chars (64-bit) — a W3C-format span id. */
export function randomHex16(): string {
  return randomBytes(8).toString("hex");
}
/** 32 lowercase hex chars (128-bit) — a W3C-format trace id. */
export function randomHex32(): string {
  return randomBytes(16).toString("hex");
}

/**
 * Normalize a customer-supplied session id. Conversation threading is opt-in:
 * passing the same id across turns groups them. Defensive + fail-open — string-
 * coerces, trims, strips control chars, caps at 200 chars, and returns "" on any
 * failure or empty/nullish input (so the caller falls back to the per-run UUID).
 * Must never throw — an SDK failure must never propagate into the caller.
 */
export function sanitizeSessionId(v: unknown): string {
  try {
    if (v == null) return "";
    const raw = typeof v === "string" ? v : String(v);
    let out = "";
    // Strip control chars (code < 0x20 or 0x7f) without a control-char regex;
    // cap at 200 chars. Pure char-code scan — cannot throw.
    for (let i = 0; i < raw.length && out.length < 200; i++) {
      const code = raw.charCodeAt(i);
      if (code > 0x1f && code !== 0x7f) out += raw[i];
    }
    return out.trim();
  } catch {
    return "";
  }
}

/**
 * Factory that yields the SDK's OTel tracer, registered by the telemetry
 * layer at setup time. Kept as an injected setter (rather than importing
 * telemetry directly) to avoid a context↔telemetry import cycle.
 */
let _agentTracerFactory: (() => Tracer | undefined) | null = null;

/** @internal Called by telemetry.setupOpenTelemetry(). */
export function _setAgentTracerFactory(f: () => Tracer | undefined): void {
  _agentTracerFactory = f;
}

/**
 * Parent id for paths that cannot use a live OTel span.
 *
 * Returns `session.rootSpanId` only when this session was **anchored** by a
 * structural agent/chain span (real OTel ids bound). Otherwise `""` so we
 * never invent a phantom parent pointing at a throwaway root that was never
 * logged. Never throws.
 * @internal
 */
export function sessionParentSpanId(session: TPSession): string {
  try {
    if (session._anchored) return session.rootSpanId || "";
  } catch {
    // fall through
  }
  return "";
}

/**
 * Resolves the `{ trace_id, span_id, parent_span_id }` for a manually-built
 * span (Mode C/D providers that have no OTel span of their own). The span is
 * a leaf, so it gets a fresh 16-hex span id and parents onto the currently
 * active OTel span (the enclosing agent/workflow span). Falls back to the
 * session's anchored root when no live OTel context is available, or `""`
 * for unscoped throwaway sessions. Never throws.
 */
export function manualSpanIds(session: TPSession): {
  trace_id: string;
  span_id: string;
  parent_span_id: string;
} {
  const span_id = randomHex16();
  try {
    const active = trace.getActiveSpan();
    if (active) {
      const ctx = active.spanContext();
      if (
        ctx &&
        ctx.traceId &&
        ctx.traceId !== "00000000000000000000000000000000" &&
        ctx.spanId
      ) {
        return { trace_id: ctx.traceId, span_id, parent_span_id: ctx.spanId };
      }
    }
  } catch {
    // Fail-open — fall through to session ids.
  }
  return {
    trace_id: session.traceId,
    span_id,
    parent_span_id: sessionParentSpanId(session),
  };
}

/**
 * Represents a discrete session of LLM interactions tracked by TokenPolice.
 */
export class TPSession {
  userId: string;
  paidPlan: string;
  /**
   * @internal Provenance of `paidPlan` — wire metadata only, never matching.
   * `"app"`: the resolved value was ultimately supplied by application code
   * (this scope, or inherited from an ancestor scope that supplied it).
   * `"default"`: the SDK synthesized it (the `"free"` fallback fired with no
   * app input anywhere in the chain). Resolves along EXACTLY the same path the
   * value does, so an inherited value carries the parent's source. Sent as
   * `user.plan_source`; the collector maps anything missing/other to
   * `"unknown"`. Not user-settable — derived, never an option.
   */
  planSource: "app" | "default";
  workflowName: string;
  sessionId: string;
  metadata: Record<string, unknown>;

  // ── Span hierarchy ──
  traceId: string;
  rootSpanId: string;
  /**
   * @internal True only after `_runWithStructuralSpan` binds real non-zero
   * OTel ids onto this object. Throwaway sessions from `getCurrentSession()`
   * stay false so manual/Mode-A parents do not point at a never-logged
   * `rootSpanId`. Lives on the session object (not "is ALS set now") so
   * post-scope stream finalize holding this ref still parents correctly.
   */
  _anchored: boolean = false;
  private _spanCounter: number = 0;

  /**
   * @internal
   * Stores prompt/response composition keyed by `traceId:spanOrder`.
   * `provider` is an optional override the enforcer stashes when the physical
   * SDK (e.g. `gen_ai.system="openai"`) does not reflect the real provider —
   * notably the OpenAI SDK pointed at OpenRouter.
   */
  _pendingCompositions: Record<
    string,
    {
      prompt?: unknown[];
      response?: unknown[];
      provider?: string;
      /** Model name override stashed by the enforcer when the instrumentor
       * span's `gen_ai.request.model` is missing or mangled. */
      model?: string;
      /** Serving endpoint (host+path) stashed by the enforcer so the telemetry
       * onEnd can forward it as `model_extras.api_base` for provider identity. */
      api_base?: string;
      /** Vendor head of a gateway-routed model slug ("openai/gpt-4.1-nano" ->
       * "openai") stashed by the enforcer on the detected-gateway path; onEnd
       * forwards it as `model_extras.original_provider` when non-empty. Absent
       * on non-gateway spans. */
      original_provider?: string;
      /** Token usage stashed by the enforcer's streaming tap for providers whose
       * instrumentor span carries none (OpenAI-compatible streaming via
       * Traceloop), OR by the LangChain streamIterator wrapper from concat'd
       * AIMessage.usage_metadata (tagged usage_source:"langchain_message").
       * onEnd prefers LC message stash on under-bill; non-LC path still fills
       * only when span usage attrs are both zero. */
      usage?: {
        input_tokens?: number;
        output_tokens?: number;
        cached_tokens?: number;
        /** Anthropic-style cache WRITE tokens from LC usage_metadata.input_token_details.cache_creation. */
        cache_creation_tokens?: number;
        /** Tag set by the LangChain stream path so telemetry skips OpenAI N1
         * add-cached-back (LC Gemini input is cache-inclusive). */
        usage_source?: string;
        /** The provider's verbatim chunk usage (cache-INCLUSIVE prompt_tokens +
         * prompt_tokens_details), stashed alongside the netted positional counts
         * above. onEnd emits it as `usage.raw` on the Mode-A stream fallback so
         * the OpenAI-family shape mappers don't double-subtract cached. */
        raw?: Record<string, unknown>;
      };
      /** Reasoning-token count from LangChain's
       * `usage_metadata.output_token_details.reasoning` (stream concat +
       * generate path), stashed by the LC wrappers because the LC instrumentor
       * never emits `gen_ai.usage.reasoning_tokens`. Slot-level (survives the
       * per-chunk `usage` REPLACE above); onEnd uses it only as a fallback for
       * the zero span attr, clamped to the final output count. */
      reasoning_tokens?: number;
      /** Canonical service tier the provider reported on the response
       * ("batch" | "flex" | "priority"), stashed by the enforcer post-hook /
       * stream tap. onEnd forwards it as `usage.tier` so tier-specific
       * pricing applies. Absent for the provider's default tier. */
      service_tier?: string;
      /** The provider's verbatim usage object for a NON-streaming openai-wire
       * call, stashed by the enforcer post-hook only when it carries token
       * detail sub-objects (reasoning/cached) the Traceloop OpenAI attrs drop.
       * Distinct from `usage.raw` above (stream-tap paired with stashed
       * positional counts); onEnd forwards it as `usage.raw` after an
       * exact-count consistency check against the span attrs. */
      nonStreamVerbatimRawUsage?: Record<string, unknown>;
      /** Client-side latency primitives (is_streaming/ttft/total) stashed by
       * the enforcer's streaming tap; onEnd forwards them verbatim so
       * streamed instrumented calls carry TTFT like the manual paths. */
      latency?: {
        is_streaming: boolean;
        ttft_ms: number | null;
        total_ms: number;
        generation_ms: number | null;
        output_tokens: number | null;
        clock: string;
      };
    }
  > = {};

  /**
   * @internal
   * Ordered (id, name) tool-call ids from the most recently captured LLM
   * response. The enforcer REPLACES this on every response capture (so stale ids
   * from a prior agent-loop iteration can't leak), and the manual toolSpan() /
   * tool() path POPS the first name-match when it fires — auto-correlating a tool
   * execution to the model's tool-call id without the app passing it explicitly.
   * Best-effort + fail-open: any miss leaves callId "". Holds id + name only,
   * never args/results (privacy).
   */
  _pendingToolCalls: Array<{ id: string; name: string }> = [];

  /** @internal REPLACE the pending tool-call ids captured from the latest LLM response. */
  setPendingToolCalls(pairs: Array<{ id: string; name: string }>): void {
    this._pendingToolCalls = Array.isArray(pairs) ? pairs : [];
  }

  /**
   * @internal APPEND one (id, name) pair captured incrementally mid-stream.
   * Unlike setPendingToolCalls (which REPLACES with the whole growing
   * accumulator and would resurrect an already-consumed same-named id), append
   * adds exactly this one call — so FIFO pop-by-name stays correct when a single
   * streamed step emits two calls with the SAME tool name. No-op on empty id.
   * Fail-open: any error leaves the stash unchanged.
   */
  appendPendingToolCall(id: string, name: string): void {
    try {
      const cid = String(id ?? "");
      if (!cid) return;
      if (!Array.isArray(this._pendingToolCalls)) this._pendingToolCalls = [];
      this._pendingToolCalls.push({ id: cid, name: String(name ?? "") });
    } catch {
      /* fail-open */
    }
  }

  /** @internal FIFO pop-on-match: id of the first pending entry whose name matches, else "". */
  popPendingToolCallId(name: string): string {
    try {
      const list = this._pendingToolCalls;
      if (!Array.isArray(list) || list.length === 0) return "";
      const idx = list.findIndex((e) => e?.name === name);
      if (idx === -1) return "";
      const [entry] = list.splice(idx, 1);
      return entry?.id ?? "";
    } catch {
      return "";
    }
  }

  /**
   * Depth counter — > 0 while executing inside a LangChain-instrumented call.
   * LangChain calls the underlying provider SDK (openai/anthropic/...) which is
   * ALSO patched by the enforcer. The LangChain wrapper enters/exits this guard
   * so the nested provider wrapper passes straight through (one check, one
   * composition capture). A counter (not a bool) keeps nested LangChain calls
   * correct; concurrent LangChain calls in one session is an accepted edge case.
   */
  private _inLangchainDepth: number = 0;

  /** @internal Enter a LangChain-instrumented call scope. */
  enterLangchain(): void {
    this._inLangchainDepth++;
  }

  /** @internal Exit a LangChain-instrumented call scope. */
  exitLangchain(): void {
    if (this._inLangchainDepth > 0) this._inLangchainDepth--;
  }

  /** @internal True while executing inside a LangChain-instrumented call. */
  get inLangchain(): boolean {
    return this._inLangchainDepth > 0;
  }

  /**
   * Depth counter — > 0 while executing inside a LlamaIndex-instrumented call.
   * Same idea as inLangchain: LlamaIndex's provider LLM classes
   * (@llamaindex/openai, @llamaindex/anthropic, @llamaindex/google) internally
   * call the underlying provider SDK which is ALSO patched. The LlamaIndex
   * wrapper enters/exits this guard so the nested provider wrapper short-
   * circuits to a pure pass-through.
   */
  private _inLlamaIndexDepth: number = 0;

  /** @internal Enter a LlamaIndex-instrumented call scope. */
  enterLlamaIndex(): void {
    this._inLlamaIndexDepth++;
  }

  /** @internal Exit a LlamaIndex-instrumented call scope. */
  exitLlamaIndex(): void {
    if (this._inLlamaIndexDepth > 0) this._inLlamaIndexDepth--;
  }

  /** @internal True while executing inside a LlamaIndex-instrumented call. */
  get inLlamaIndex(): boolean {
    return this._inLlamaIndexDepth > 0;
  }

  /**
   * @internal
   * Usage stash for streamed provider calls made INSIDE a LlamaIndex scope.
   * @llamaindex/anthropic forwards only content deltas (never usage events),
   * so the LlamaIndex stream wrapper would log 0/0 tokens. The pass-through
   * tap on the inner provider stream merges message_start/message_delta usage
   * here; _logLlamaIndex consumes it as a fallback when the chunk-level
   * extraction yields zero. Cleared at the start of every LlamaIndex call.
   */
  _pendingLlamaIndexUsage: {
    model: string;
    usage: Record<string, unknown>;
  } | null = null;

  constructor(opts?: {
    userId?: string;
    paidPlan?: string;
    /**
     * @internal Explicit provenance for an INHERITED `paidPlan` (nested
     * scopes pass the parent's source alongside the parent's value). Omit for
     * a root scope — provenance is then derived from `paidPlan` below.
     */
    planSource?: "app" | "default";
    workflowName?: string;
    sessionId?: string;
    metadata?: Record<string, unknown>;
    traceId?: string;
    rootSpanId?: string;
  }) {
    this.userId = opts?.userId ?? "anonymous";
    this.paidPlan = opts?.paidPlan ?? "free";
    // Mirror the line above exactly: the `??` fallback fires only for a
    // nullish `paidPlan`, so nullish ⇒ SDK-synthesized ("default") and any
    // supplied value (including "") ⇒ app-supplied ("app"). An explicitly
    // passed planSource (the nested-scope inheritance path) wins; anything
    // else is ignored so the field can only ever hold the two wire values.
    this.planSource =
      opts?.planSource === "app" || opts?.planSource === "default"
        ? opts.planSource
        : opts?.paidPlan != null
          ? "app"
          : "default";
    this.workflowName = opts?.workflowName ?? "default_workflow";
    this.sessionId = opts?.sessionId ?? randomUUID();
    this.metadata = opts?.metadata ?? {};
    // W3C-format ids. When a session opens a real OTel agent span these are
    // overwritten with the span's native trace/span ids (see session()).
    this.traceId = opts?.traceId ?? randomHex32();
    this.rootSpanId = opts?.rootSpanId ?? randomHex16();
  }

  /** @internal Returns the next child span order index. */
  nextSpanOrder(): number {
    return this._spanCounter++;
  }

  /** @internal Returns current counter value without incrementing. */
  get spanCounter(): number {
    return this._spanCounter;
  }
}

/**
 * Internal AsyncLocalStorage instance for session propagation.
 * Thread-safe, async-safe — works across async/await, Promises, and callbacks.
 */
const _sessionStorage = new AsyncLocalStorage<TPSession>();
const _pendingSpanName = new AsyncLocalStorage<{ name: string | null }>();

/**
 * Reservation for the next instrumented LLM span's order. The instrumented
 * wrapper allocates the span order up front (before the provider call) and runs
 * the call inside this scope; telemetry `onStart` consumes the reservation so
 * the span it opens carries exactly that order instead of peeking the mutable
 * counter. Reserving the order before the provider call is what lets concurrent
 * same-session calls avoid cross-attributing their pre/post-call composition
 * stashes (see tests/concurrentAttribution.test.ts).
 */
const _reservedSpanOrder = new AsyncLocalStorage<{
  order: number;
  consumed: boolean;
}>();

/**
 * @internal Run `fn` with a reserved span order active, so the next telemetry
 * `onStart` consumes `reservation.order` rather than allocating a fresh one.
 * Uses AsyncLocalStorage `.run()` scoping — the reservation is automatically
 * unset when `fn` (and its async continuation) settle, so it can never leak
 * onto a later, unrelated span.
 */
export function runWithReservedSpanOrder<T>(
  reservation: { order: number; consumed: boolean },
  fn: () => T,
): T {
  return _reservedSpanOrder.run(reservation, fn);
}

/**
 * Marks the async-context window during which a manual Anthropic `.stream()`
 * wrapper is constructing its MessageStream. The OTel anthropic instrumentor is
 * wrapped INNER of our `.stream()` patch (it is registered first, at
 * setupOpenTelemetry, then our manual patch wraps outer at autoInstrument), so
 * its duplicate span is started synchronously WITHIN this window. Telemetry
 * `onStart` reads this flag to tag exactly that span for suppression in `onEnd`.
 *
 * Because AsyncLocalStorage scopes this per `.stream()` call, concurrent
 * same-session streams each tag their own duplicate span — no call can consume
 * another call's suppression, which the former session-wide one-shot boolean
 * allowed (it could eat the wrong span → lose that row → let the intended
 * duplicate double-count). A stub (run⇒fn(), active⇒false) keeps this a no-op
 * on runtimes without AsyncLocalStorage: the duplicate span is then simply not
 * suppressed (a harmless double-count of the instrumentor's own row) — never a
 * lost manual row, never a throw.
 */
const _anthropicStreamOtelSuppress: {
  run<T>(fn: () => T): T;
  active(): boolean;
} = (() => {
  try {
    const als = new AsyncLocalStorage<boolean>();
    return {
      run<T>(fn: () => T): T {
        return als.run(true, fn);
      },
      active(): boolean {
        return als.getStore() === true;
      },
    };
  } catch {
    return {
      run<T>(fn: () => T): T {
        return fn();
      },
      active(): boolean {
        return false;
      },
    };
  }
})();

/**
 * @internal Run `fn` with the Anthropic `.stream()` OTel-suppression window
 * active, so the (inner) instrumentor span the wrapped `.stream()` opens during
 * `fn` is tagged by telemetry `onStart` for `onEnd` to drop. Auto-unwinds when
 * `fn` (and its sync continuation) return, so the window can never leak onto a
 * later, unrelated span.
 */
export function runWithAnthropicStreamOtelSuppress<T>(fn: () => T): T {
  return _anthropicStreamOtelSuppress.run(fn);
}

/**
 * @internal True while executing inside a `runWithAnthropicStreamOtelSuppress`
 * window (a manual Anthropic `.stream()` construction). Never throws.
 */
export function anthropicStreamOtelSuppressActive(): boolean {
  try {
    return _anthropicStreamOtelSuppress.active();
  } catch {
    return false;
  }
}

/**
 * Marks the async-context window during which the LlamaIndex wrapper
 * (_setLlamaIndexWrapper / _guardedLlamaIndexStream) is executing the
 * underlying provider call. Within that window the LlamaIndex wrapper is the
 * SOLE billable emitter for the call in flight — it logs the manual row via
 * `_logLlamaIndex` (or `_emitCallFailureLog` on failure) and the inner
 * provider ENFORCER wrapper passes through on the `inLlamaIndex` guard. But
 * when a Traceloop OTel provider instrumentor (@traceloop/instrumentation-
 * anthropic / -openai) ALSO wraps the underlying provider SDK, it starts its
 * own span for the same physical call — a duplicate that
 * `TokenPoliceSpanProcessor.onEnd` would log as a SECOND full-cost llm row
 * (run-23 F-23-1: exact 2× billing). Telemetry `onStart` reads this flag to
 * tag exactly that span for suppression in `onEnd`.
 *
 * AsyncLocalStorage scopes this per call, NOT per session: a concurrent
 * same-session DIRECT provider call (no LlamaIndex in its chain) keeps its
 * instrumentor span billable — the session-scoped `inLlamaIndex` flag would
 * eat that span too (under-billing, the one failure mode worse than the bug).
 * A stub (run⇒fn(), active⇒false) keeps this a no-op on runtimes without
 * AsyncLocalStorage: the duplicate span then simply survives (the pre-fix
 * over-count status quo) — never a lost manual row, never a throw.
 */
const _llamaIndexOtelSuppress: {
  run<T>(fn: () => T): T;
  active(): boolean;
} = (() => {
  try {
    const als = new AsyncLocalStorage<boolean>();
    return {
      run<T>(fn: () => T): T {
        return als.run(true, fn);
      },
      active(): boolean {
        return als.getStore() === true;
      },
    };
  } catch {
    return {
      run<T>(fn: () => T): T {
        return fn();
      },
      active(): boolean {
        return false;
      },
    };
  }
})();

/**
 * @internal Run `fn` with the LlamaIndex OTel-suppression window active, so
 * any provider-instrumentor span that starts while the LlamaIndex wrapper is
 * driving the underlying call during `fn` is tagged by telemetry `onStart`
 * for `onEnd` to drop. Auto-unwinds when `fn` (and async work initiated
 * inside it) settle, so the window can never leak onto a later, unrelated
 * span.
 */
export function runWithLlamaIndexOtelSuppress<T>(fn: () => T): T {
  return _llamaIndexOtelSuppress.run(fn);
}

/**
 * @internal True while executing inside a `runWithLlamaIndexOtelSuppress`
 * window (a LlamaIndex wrapper driving the underlying provider call). Never
 * throws.
 */
export function llamaIndexOtelSuppressActive(): boolean {
  try {
    return _llamaIndexOtelSuppress.active();
  } catch {
    return false;
  }
}

/**
 * @internal Consume the active span-order reservation, if one is present and
 * not yet consumed. Marks it consumed (so a nested span opened within the same
 * scope allocates a fresh order) and returns the reserved order, or `null` when
 * there is nothing to consume. Never throws.
 */
export function consumeReservedSpanOrder(): number | null {
  try {
    const res = _reservedSpanOrder.getStore();
    if (res && !res.consumed) {
      res.consumed = true;
      return res.order;
    }
  } catch {
    // fall through to null — onStart falls back to fresh allocation
  }
  return null;
}

/**
 * Returns the active session, or a default one if none is set.
 */
export function getCurrentSession(): TPSession {
  return _sessionStorage.getStore() ?? new TPSession();
}

/**
 * Set a name for the next LLM call's span.
 * Consumed once — after the next LLM call, the name is cleared.
 *
 * Only takes effect **inside** a session/agent/chain/workflow scope: the name is
 * held in scope-local storage that these helpers establish. Called with no
 * enclosing scope it is a silent no-op (there is nowhere to store the name), so
 * always call it within one of those blocks.
 *
 * @example
 * ```typescript
 * await tp.session({ name: "rag_pipeline" }, async () => {
 * tp.setSpanName("route_query");
 * await client.chat.completions.create({ ... }); // This span is named "route_query"
 * // The next call has no name set (defaults to the model name)
 * await client.chat.completions.create({ ... });
 * });
 * ```
 */
export function setSpanName(name: string): void {
  const store = _pendingSpanName.getStore();
  if (store) {
    store.name = name;
  }
}

/** @internal Consumes and returns the pending span name. */
export function consumePendingSpanName(): string | null {
  const store = _pendingSpanName.getStore();
  if (store?.name) {
    const name = store.name;
    store.name = null;
    return name;
  }
  return null;
}

// ── session() — Callback-based context scope ────────────────────

export interface SessionOptions {
  /** Workflow/feature name for grouping. */
  name?: string;
  /** User identifier for budget attribution. */
  userId?: string;
  /** Billing plan tier (e.g., "free", "pro", "enterprise"). */
  paidPlan?: string;
  /** Arbitrary metadata to attach to all LLM calls in this scope. */
  metadata?: Record<string, unknown>;
  /**
   * Stable conversation/session id. Pass the SAME value across separate top-level
   * sessions (e.g. each turn of a chat) to thread them into one conversation in
   * the dashboard. Omit to get a per-run UUID (the run is its own singleton
   * session). Sanitized + capped at 200 chars; never affects the customer's app.
   */
  sessionId?: string;
  /**
   * Span kind of the root/grouping span. `"agent"` (default) — a dynamic,
   * LLM-driven loop. `"chain"` — a static/linear, developer-defined sequence
   * (glue code / pipeline). Use `chain()` / `agent()` for the explicit sugar.
   */
  kind?: "agent" | "chain";
}

/**
 * Runs a callback within a TokenPolice session context.
 * All instrumented LLM calls inside the callback will inherit this session.
 *
 * Supports nested sessions — inner sessions inherit the outer session's
 * session_id (for grouping) and merge metadata.
 *
 * @example
 * ```typescript
 * import * as tp from 'token-police';
 *
 * await tp.session({ name: "rag_pipeline", userId: "user_42" }, async () => {
 * const response = await openai.chat.completions.create({ ... });
 * });
 * ```
 */
export function session<T>(
  options: SessionOptions,
  fn: (session: TPSession) => T,
): T {
  return _sessionImpl(options, "agent", fn);
}

/**
 * Explicit **agent** span — a dynamic, LLM-driven execution loop where the model
 * decides the path on the fly. Same as `session()` (which defaults to agent);
 * use it when you want the intent to read explicitly in the code.
 *
 * @example
 * ```typescript
 * await tp.agent({ name: "support_agent", userId: "u1" }, async () => { ... });
 * ```
 */
export function agent<T>(
  options: SessionOptions,
  fn: (session: TPSession) => T,
): T {
  return _sessionImpl(options, "agent", fn);
}

/**
 * Explicit **chain** span — a static/linear, developer-defined sequence of steps
 * (a pipeline or glue code linking non-agentic steps, e.g. retriever → LLM). Use
 * a chain as the root entry point or to group sequential work; a purely agentic
 * trace (agent + tool + llm) may not need one at all.
 *
 * @example
 * ```typescript
 * await tp.chain({ name: "rag_pipeline", userId: "u1" }, async () => {
 * const docs = await retrieve(q);
 * return await openai.chat.completions.create({ ... });
 * });
 * ```
 */
export function chain<T>(
  options: SessionOptions,
  fn: (session: TPSession) => T,
): T {
  return _sessionImpl(options, "chain", fn);
}

/**
 * Shared implementation for session()/agent()/chain(). Builds the TPSession
 * (inheriting from an enclosing session when nested) and opens a structural
 * anchor span of the resolved kind (`options.kind` overrides `defaultKind`).
 */
function _sessionImpl<T>(
  options: SessionOptions,
  defaultKind: "agent" | "chain",
  fn: (session: TPSession) => T,
): T {
  const existing = _sessionStorage.getStore();
  const kind = options.kind ?? defaultKind;
  // Opt-in conversation id. Empty → fall back to the constructor's per-run UUID
  // (top-level) or the parent's id (nested).
  const explicitSessionId = sanitizeSessionId(options.sessionId);

  let s: TPSession;

  if (existing) {
    // Nested session → a CHILD anchor of the enclosing one. Inherit session_id
    // (grouping) and trace_id (same run), but NOT root_span_id — the nested
    // span gets its own id so it nests under its parent instead of collapsing
    // into the root (the old flat model). When an OTel span opens below, both
    // trace_id and root_span_id are refreshed from it.
    const mergedMeta = { ...existing.metadata, ...(options.metadata ?? {}) };

    // Provided-ness, not value, decides inheritance: any truthy provided value
    // (including the literal defaults "default_workflow"/"anonymous"/"free")
    // overrides the parent; an omitted/empty field inherits. Empty string keeps
    // behaving as "unset". Comparing against the default literals instead would
    // make exactly those three strings impossible to set in a nested scope.
    // Covered by tests/nestedScopeDefaultLiterals.test.ts.
    s = new TPSession({
      sessionId: explicitSessionId || existing.sessionId,
      traceId: existing.traceId,
      workflowName: options.name || existing.workflowName,
      userId: options.userId || existing.userId,
      paidPlan: options.paidPlan || existing.paidPlan,
      // Provenance follows the value on the SAME `||`: a truthy provided plan
      // is app-supplied here; an omitted/empty one inherits the parent's VALUE
      // and therefore the parent's SOURCE (so "app" set three scopes up still
      // reads as "app" here, and a never-set chain stays "default").
      planSource: options.paidPlan ? "app" : existing.planSource,
      metadata: mergedMeta,
    });
  } else {
    s = new TPSession({
      // Empty → TPSession constructor mints a per-run UUID (singleton session).
      sessionId: explicitSessionId || undefined,
      workflowName: options.name,
      userId: options.userId,
      paidPlan: options.paidPlan,
      metadata: options.metadata,
    });
  }

  return _sessionStorage.run(s, () => {
    return _pendingSpanName.run({ name: null }, () =>
      _runWithStructuralSpan(s, kind, () => fn(s)),
    );
  });
}

/**
 * Opens a real OTel structural span (`agent` or `chain`) around the session
 * body and makes it the active span, so that (a) OpenLLMetry LLM spans created
 * inside auto-nest under it, and (b) manual-telemetry spans can parent onto it
 * via `trace.getActiveSpan()`. The span's native (W3C-format) trace/span ids
 * become the session's `traceId` / `rootSpanId`. The span is tagged
 * `tp.kind=<kind>` so the SpanProcessor emits it as a structural anchor row
 * (no usage / cost) and does not treat it as an LLM call.
 *
 * Fail-open: if telemetry isn't set up (no tracer), just runs the body — the
 * session keeps its constructor-default hex ids. The same degradation applies
 * to ANY failure while setting the span up (attribute building over exotic
 * metadata, or a tracer whose span-open call itself throws): opening a
 * session must never throw customer-side — a failed span setup degrades to
 * running the body untraced, exactly once, with the body's own return/throw
 * forwarded unchanged. Covered by tests/structuralSpan.test.ts.
 */
function _runWithStructuralSpan<T>(
  s: TPSession,
  kind: "agent" | "chain",
  body: () => T,
): T {
  let tracer: Tracer | undefined;
  try {
    tracer = _agentTracerFactory?.() ?? undefined;
  } catch {
    tracer = undefined;
  }
  if (!tracer) return body();

  // Once-only body guard. `bodyRan` flips synchronously BEFORE body() is
  // invoked, so the recovery path below can never run the body a second time;
  // the body's genuine outcome (value or error) is recorded so it can be
  // replayed if the tracer misbehaves around it.
  let bodyRan = false;
  let bodyThrew = false;
  let bodyError: unknown;
  let bodyResult: T | undefined;
  const runBodyOnce = (): T => {
    if (bodyRan) {
      // A misbehaving tracer invoked its callback more than once — replay the
      // first outcome instead of re-running customer code.
      if (bodyThrew) throw bodyError;
      return bodyResult as T;
    }
    bodyRan = true;
    try {
      bodyResult = body();
      return bodyResult;
    } catch (e) {
      bodyThrew = true;
      bodyError = e;
      throw e;
    }
  };

  try {
    const attributes: Record<string, string> = {
      "tp.kind": kind,
      "tp.workflow_name": s.workflowName,
      "tp.user_id": s.userId,
      "tp.paid_plan": s.paidPlan,
      // Provenance rides with the plan so onEnd (which rebuilds the row from
      // span attributes, not the live session) can forward it.
      "tp.plan_source": s.planSource,
      "tp.session_id": s.sessionId,
    };
    for (const [k, v] of Object.entries(s.metadata)) {
      if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
        attributes[`tp.meta.${k}`] = String(v);
      } else if (v != null) {
        // JSON-serialize object-valued metadata (e.g. `_tp_routing`)
        // instead of dropping it, so reroute provenance survives downstream.
        try {
          attributes[`tp.meta.${k}`] = JSON.stringify(v);
        } catch {
          /* circular / un-serializable → skip this key */
        }
      }
    }

    return tracer.startActiveSpan(s.workflowName, { attributes }, (span: OtelSpan): T => {
      try {
        const ctx = span.spanContext();
        if (
          ctx?.traceId &&
          ctx.traceId !== "00000000000000000000000000000000" &&
          ctx.spanId &&
          ctx.spanId !== "0000000000000000"
        ) {
          s.traceId = ctx.traceId;
          s.rootSpanId = ctx.spanId;
          // Real structural ids bound → children may parent onto root.
          // Fail-open (zero/invalid context) leaves _anchored false so
          // parents become "" rather than a never-logged constructor id.
          s._anchored = true;
        }
      } catch {
        // Keep the constructor-default ids on any failure; _anchored stays false.
      }
      const endSpan = () => {
        try {
          span.end();
        } catch {
          // ignore
        }
      };
      // Stamp the anchor span ERROR when the decorated body throws/rejects, so
      // the emitted structural row reports status='failed' (Node OTel does NOT
      // auto-set status on exception, unlike Python). Annotation only — the
      // original error object is re-thrown byte-identical below, so the golden
      // rule holds. Wrapped in try/catch: a tracer misbehaving on setStatus must
      // never outrank the customer's error.
      const markErrored = (e: unknown) => {
        try {
          span.setStatus({
            code: SpanStatusCode.ERROR,
            message: String((e as any)?.message ?? e ?? ""),
          });
        } catch {
          // ignore — status annotation must never affect the customer's error
        }
      };
      let result: T;
      try {
        result = runBodyOnce();
      } catch (err) {
        markErrored(err);
        endSpan();
        throw err;
      }
      if (result && typeof (result as any).then === "function") {
        return (result as any).then(
          (v: any) => {
            endSpan();
            return v;
          },
          (e: any) => {
            markErrored(e);
            endSpan();
            throw e;
          },
        ) as T;
      }
      endSpan();
      return result;
    });
  } catch (err) {
    // Span setup failed (attribute build or the tracer's span-open call)
    // before the body could run → degrade to running the body untraced. The
    // session keeps its constructor-default ids (the documented no-tracer
    // behavior).
    if (!bodyRan) return runBodyOnce();
    // The body already ran inside the tracer's callback. Forward its genuine
    // outcome unchanged; anything the tracer threw on its own way out is
    // swallowed (telemetry must never outrank the customer's result).
    if (bodyThrew) throw bodyError;
    return bodyResult as T;
  }
}

// ── workflow() — Higher-order function (decorator pattern) ──────

export interface WorkflowOptions {
  /** Workflow name (e.g., "rag_pipeline"). */
  name?: string;
  /** Static user_id. Can be overridden dynamically. */
  userId?: string;
  /** Static paid_plan. Can be overridden dynamically. */
  paidPlan?: string;
  /** Static metadata. Merged with dynamic metadata. */
  metadata?: Record<string, unknown>;
  /**
   * Stable conversation/session id (see SessionOptions.sessionId). Can also be
   * supplied dynamically via the wrapped function's first-arg object
   * (`sessionId` / `session_id`).
   */
  sessionId?: string;
  /**
   * Span kind of the root span. Defaults to **`"chain"`** — a workflow is a
   * developer-defined, predefined sequence of steps. Pass `"agent"` for an
   * autonomous, LLM-driven loop.
   */
  kind?: "agent" | "chain";
  /**
   * If true (default), the decorator attempts to extract `userId`, `paidPlan`,
   * `sessionId`, and `metadata` from the wrapped function's first argument if it
   * is an object.
   */
  bindArgs?: boolean;
}

/**
 * Wraps a function as a TokenPolice **chain** by default — a workflow is a
 * predefined, linear sequence of steps. Pass `{ kind: "agent" }` for an
 * autonomous, LLM-driven loop. All nested LLM calls share the same session_id
 * and workflow_name.
 *
 * DYNAMIC BINDING: If bindArgs is true (default) and the wrapped function's
 * first argument is an object with `userId`, `paidPlan`, `sessionId`, or
 * `metadata` properties, they override the static values.
 *
 * @example
 * ```typescript
 * import * as tp from 'token-police';
 *
 * const runPipeline = tp.workflow(
 * { name: "rag_pipeline", paidPlan: "enterprise" },
 * async (userId: string, query: string) => {
 * return await openai.chat.completions.create({ ... });
 * }
 * );
 *
 * await runPipeline("user_123", "How do I reset my password?");
 * ```
 *
 * @example Dynamic binding with object argument (incl. conversation threading):
 * ```typescript
 * const processTurn = tp.workflow(
 * { name: "api_handler" },
 * async (opts: { userId: string; sessionId: string; query: string }) => {
 * return await openai.chat.completions.create({ ... });
 * }
 * );
 *
 * await processTurn({ userId: "user_456", sessionId: "conv_42", query: "Hello!" });
 * ```
 */
export function workflow<TArgs extends any[], TReturn>(
  options: WorkflowOptions,
  fn: (...args: TArgs) => TReturn,
): (...args: TArgs) => TReturn {
  const bindArgs = options.bindArgs ?? true;

  return (...args: TArgs): TReturn => {
    let resolvedUserId = options.userId;
    let resolvedPaidPlan = options.paidPlan;
    let resolvedSessionId = options.sessionId;
    let resolvedMetadata = { ...(options.metadata ?? {}) };

    // Dynamic binding: extract from first argument if it's an object
    if (bindArgs && args.length > 0) {
      const firstArg = args[0];

      if (typeof firstArg === "object" && firstArg !== null && !Array.isArray(firstArg)) {
        const argObj = firstArg as Record<string, unknown>;
        if (typeof argObj.userId === "string") {
          resolvedUserId = argObj.userId;
        }
        if (typeof argObj.user_id === "string") {
          resolvedUserId = argObj.user_id;
        }
        if (typeof argObj.paidPlan === "string") {
          resolvedPaidPlan = argObj.paidPlan;
        }
        if (typeof argObj.paid_plan === "string") {
          resolvedPaidPlan = argObj.paid_plan;
        }
        if (typeof argObj.sessionId === "string") {
          resolvedSessionId = argObj.sessionId;
        }
        if (typeof argObj.session_id === "string") {
          resolvedSessionId = argObj.session_id;
        }
        if (
          typeof argObj.metadata === "object" &&
          argObj.metadata !== null
        ) {
          resolvedMetadata = {
            ...resolvedMetadata,
            ...(argObj.metadata as Record<string, unknown>),
          };
        }
      }

      // A plain-string first arg is NOT auto-bound to userId: unlike Python
      // (which reads parameter names via inspect.signature), JS can't reliably
      // introspect positional parameter names, so dynamic binding is object-only.
    }

    // A workflow is a chain by default; `kind` overrides.
    return _sessionImpl(
      {
        name: options.name,
        userId: resolvedUserId,
        paidPlan: resolvedPaidPlan,
        sessionId: resolvedSessionId,
        kind: options.kind,
        metadata: resolvedMetadata,
      },
      "chain",
      () => fn(...args),
    );
  };
}

// ── Manual tool capture (frameworks that emit no tool span) ──────
// CrewAI, hand-rolled function tools, and MCP calls don't produce an OTel
// tool span, so the SpanProcessor never sees them. toolSpan() / tool() let a
// user capture those explicitly. Both fail open: a capture error never throws
// into customer code, and the wrapped tool always runs.

export interface ToolSpanOptions {
  /** Tool name (shown in the trace). */
  name: string;
  /** Tool type: "function" | "retriever" | "mcp" | "extension". */
  type?: string;
  /** Provider tool_call id, if known. */
  callId?: string;
  /** Optional args to record as a (hash, length) — raw content is discarded. */
  args?: unknown;
}

// `hashLen` (sha1-16 + code-point length for a tool arg/result) is the single
// shared copy in the composition leaf — imported above (no cycle: composition
// imports nothing from context).

function _emitToolRow(
  opts: ToolSpanOptions,
  startISO: string,
  endISO: string,
  durationMs: number,
  status: string,
  errKind: string,
  errMsg: string,
  result: unknown,
): void {
  try {
    const tp = getClient();
    if (!tp) return;
    const session = getCurrentSession();
    // Auto-correlate to the model's tool-call id when the caller didn't supply
    // one (the toolSpan()/tool() path). Matched by name (FIFO) against ids
    // stashed from the preceding LLM response; "" on any miss.
    const callId = opts.callId || session.popPendingToolCallId(opts.name);
    const ids = manualSpanIds(session);
    const [paramHash, paramLen] = hashLen(opts.args);
    const [resultHash, resultLen] = hashLen(result);

    const callOutcome: Record<string, unknown> = {
      status,
      // Positive sub-ms → 1; true-zero/negative → 0.
      duration_ms: toDurationMs(durationMs),
    };
    if (status === "failed") {
      // Route the raw value through the scrub helper (no pre-stringify);
      // default 'redacted' ships a hash, not the raw error string.
      Object.assign(callOutcome, scrubErrorMessage(errMsg, resolveErrorDetail()));
      if (errKind) callOutcome.error_kind = errKind;
    }

    const metadata: Record<string, unknown> = { workflow_name: session.workflowName };
    if (session.sessionId) metadata.session_id = session.sessionId;
    // B4: a `tool` row executes no model call, so it must never inherit the
    // applied-reroute marker a sibling LLM call left on the session metadata
    // (161 tool rows carried one live). Copy without it; never re-add.
    if (session.metadata) {
      copySessionMetadata(metadata, session.metadata);
    }

    tp.log(
      session.userId,
      session.paidPlan,
      session.workflowName,
      session.sessionId,
      "",
      "",
      0,
      0,
      0,
      metadata,
      {
        trace_id: ids.trace_id,
        span_id: ids.span_id,
        parent_span_id: ids.parent_span_id,
        span_kind: "tool",
        span_name: opts.name,
        span_order: 0,
        start_time: startISO,
        end_time: endISO,
      },
      [],
      [],
      {
        tool: {
          name: opts.name,
          type: opts.type ?? "function",
          call_id: callId,
          param_hash: paramHash,
          param_length: paramLen,
          result_hash: resultHash,
          result_length: resultLen,
        },
        call_outcome: callOutcome,
        planSource: session.planSource,
      },
    );
  } catch {
    // Fail-open — tool capture must never affect the customer's app.
  }
}

/**
 * Capture a tool / function execution as a `tool` span in the current trace.
 * For tools that no instrumentor sees (CrewAI, raw functions, MCP). Handles
 * sync and async `fn`. On a thrown error the span is recorded as failed and
 * the error is re-thrown unchanged.
 *
 * @example
 * ```typescript
 * const result = await tp.toolSpan({ name: "web_search" }, () => runSearch(q));
 * ```
 */
export function toolSpan<T>(opts: ToolSpanOptions, fn: () => T): T {
  // Wall clock for ISO start/end stamps (ms precision on the wire); mono clock
  // for duration so sub-ms tools don't collapse to duration_ms=0.
  const startISO = new Date().toISOString();
  const startMono = safeMonoNow();
  // End-stamp + mono delta computed inside a try so a hostile Date/clock
  // cannot throw past the customer (belt-and-suspenders; _emitToolRow is also
  // fail-open and safeMonoNow never throws).
  const done = (status: string, errKind: string, errMsg: string, result: unknown) => {
    try {
      _emitToolRow(
        opts,
        startISO,
        new Date().toISOString(),
        safeMonoNow() - startMono,
        status,
        errKind,
        errMsg,
        result,
      );
    } catch {
      /* capture must never affect the customer's tool path */
    }
  };

  let result: T;
  try {
    result = fn();
  } catch (err) {
    done(
      "failed",
      (err as any)?.constructor?.name ?? "",
      err instanceof Error ? err.message : String(err),
      undefined,
    );
    throw err;
  }

  if (result && typeof (result as any).then === "function") {
    return (result as any).then(
      (v: any) => {
        done("success", "", "", v);
        return v;
      },
      (e: any) => {
        done(
          "failed",
          e?.constructor?.name ?? "",
          e instanceof Error ? e.message : String(e),
          undefined,
        );
        throw e;
      },
    ) as T;
  }
  done("success", "", "", result);
  return result;
}

/**
 * Wraps a function so every call is captured as a `tool` span.
 *
 * @example
 * ```typescript
 * const search = tp.tool({ name: "web_search" }, (q: string) => runSearch(q));
 * ```
 */
export function tool<TArgs extends any[], TReturn>(
  opts: ToolSpanOptions,
  fn: (...args: TArgs) => TReturn,
): (...args: TArgs) => TReturn {
  return (...args: TArgs): TReturn =>
    toolSpan({ ...opts, args: opts.args ?? args }, () => fn(...args));
}

// ── serverless() — Auto-flush decorator ─────────────────────────

/**
 * Wraps a serverless handler (AWS Lambda, Vercel, etc.) to ensure
 * all pending telemetry is flushed before the function returns.
 *
 * @example
 * ```typescript
 * import * as tp from 'token-police';
 *
 * export const handler = tp.serverless(async (event, context) => {
 * // LLM calls here...
 * return { statusCode: 200, body: 'ok' };
 * });
 * ```
 */
export function serverless<TArgs extends any[], TReturn>(
  fn: (...args: TArgs) => TReturn,
): (...args: TArgs) => TReturn extends Promise<any> ? Promise<Awaited<TReturn>> : TReturn {
  return ((...args: TArgs) => {
    const result = fn(...args);

    // Duck-type on `.then` rather than `instanceof Promise` so foreign-Promise
    // thenables (Bluebird, a different realm/VM context, a custom PromiseLike)
    // still take the async flush path instead of falling through to the sync one.
    if (result && typeof (result as any).then === "function") {
      const p = result as any;
      const flush = async () => {
        const client = getClient();
        if (client) {
          await client.flush();
        }
      };
      // Prefer `.finally` (native/Bluebird promises); fall back to `.then` for a
      // bare thenable that only implements `.then`, so this never throws into the
      // customer's handler while still flushing on both settle paths.
      if (typeof p.finally === "function") {
        return p.finally(flush) as any;
      }
      return p.then(
        async (v: any) => {
          await flush();
          return v;
        },
        async (e: any) => {
          await flush();
          throw e;
        },
      ) as any;
    }

    // Sync path — Node cannot block on in-flight /log POSTs without a forbidden
    // sync primitive, so there is no real drain here. Durability for a sync
    // handler comes from the `keepalive: true` on each /log POST (the runtime
    // finishes it after the handler returns); flushSync() only emits a
    // pending-count diagnostic.
    const client = getClient();
    if (client) {
      client.flushSync();
    }
    return result as any;
  }) as any;
}

/**
 * Internal: Get the raw AsyncLocalStorage for use by telemetry module.
 * @internal
 */
export function _getSessionStorage(): AsyncLocalStorage<TPSession> {
  return _sessionStorage;
}
