/**
 * Token Police Node.js SDK.
 * Core client providing manual check/log methods and global configuration.
 *
 * Behavior is kept in parity with the TokenPolice Python SDK.
 */

import { setClient, getClient, assignClientId } from "./state";
import { TokenPoliceBlockedError } from "./exceptions";
import {
  setupOpenTelemetry,
  unsetupOpenTelemetry,
  forceFlushTokenPoliceSpans,
  type InstrumentModules,
} from "./telemetry";
import { autoInstrument, getUnwrappedResolvableProviders } from "./enforcer";
import { resolveDeployment } from "./runtime";
import { StreamClient } from "./stream";
import { spanKindFor } from "./spanKind";
import { randomHex16 } from "./context";

export const SDK_VERSION = "1.0.0";
export const SDK_SCHEMA_VERSION = "1";

/**
 * Logger scoped to TokenPolice SDK internals.
 *
 * `debug` is silent unless the active client was created with `logErrors: true`
 * (mirrors the same gate in telemetry.ts / enforcer.ts — the three loggers are
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
  warning: (msg: string) => console.warn(msg),
};

export type FirewallMode = "enforce" | "dry_run" | "off";
import type { ErrorDetailMode } from "./_classify";
export type { ErrorDetailMode };

export interface TokenPoliceOptions {
  /** Your TokenPolice API key (tp_sk_...). */
  apiKey: string;
  /**
   * TokenPolice API URL.
   * Defaults to TOKENPOLICE_BASE_URL env var or https://collect.tokenpolice.ai.
   */
  baseUrl?: string;
  /**
   * Max time for any remote call, in **SECONDS** (default: 2.0 = 2000ms).
   * A non-positive or non-finite value is treated as a misconfiguration
   * (an explicit instant-abort is never intended): the SDK warns and falls
   * back to the 2.0s default.
   */
  timeout?: number;
  /**
   * Firewall mode (default 'dry_run'):
   * - 'enforce': act on rules — block/reroute per each rule's mode.
   * - 'dry_run': evaluate and emit WOULD_* telemetry, but never act/throw.
   * - 'off': telemetry only — usage is logged for every provider, but no
   * pre-flight /check, no enforcement, and no SSE stream.
   */
  firewall?: FirewallMode;
  /**
   * @deprecated Use `firewall` instead. Legacy boolean: `true` → 'enforce',
   * `false` → 'off'. Ignored when `firewall` is set.
   */
  enforce?: boolean;
  /** If true, SDK errors are printed at WARNING level. */
  logErrors?: boolean;
  /**
   * Deployment mode (default 'auto').
   * - daemon: long-lived process. Opens SSE + caches Decision Pack.
   * - serverless: AWS Lambda / Vercel / GCP. Inline /check only.
   * - edge: Cloudflare / Vercel Edge / Deno. Inline /check only.
   */
  deployment?: "auto" | "daemon" | "serverless" | "edge";
  /** Cap on SSE reconnect backoff in seconds (default 300). */
  sseReconnectMaxIntervalSeconds?: number;
  /**
   * Grace period, in seconds, after the rule stream drops before a locally
   * ALLOWED call whose decision depended on a streamed entity list is
   * re-verified with an inline `/check` (default 60, clamped to [0, 3600];
   * 0 = verify from the first missed moment). Budget/anomaly blocks reach the
   * SDK only over the stream, so without this a long disconnect would let
   * already-blocked entities keep passing. Only entity-matching calls pay the
   * round-trip, and it stays fail-open.
   */
  streamStaleGraceSeconds?: number;
  /**
   * When an OpenAI-compatible chat stream is opened WITHOUT
   * `stream_options.include_usage`, the SDK injects it so the provider emits
   * the final usage chunk (otherwise the streamed call has no token usage and
   * its cost is silently lost). The synthetic usage-only terminal chunk is
   * stripped from your iterator, so your code sees exactly the chunks it asked
   * for. Set false to disable the injection (default: true; env override
   * TP_CAPTURE_STREAM_USAGE=0).
   */
  captureStreamUsage?: boolean;
  /**
   * Explicit module references to instrument for token usage extraction.
   * Pass your already-imported LLM SDK modules to solve ESM import ordering.
   *
   * When provided, the SDK calls manuallyInstrument() on each module,
   * which works regardless of import timing (ESM, CJS, tsx, Bun, etc.).
   *
   * When omitted, the SDK auto-discovers installed @traceloop/instrumentation-*
   * packages from your node_modules. This works well in CJS but may not
   * instrument modules that were imported before init() in ESM.
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
  instrumentModules?: InstrumentModules;
  /**
   * How much error detail leaves the process on a failed call
   * (default 'redacted'):
   * - 'none': ship only the safe classifier enums (error_kind, http_status).
   * - 'redacted': also ship a SHA-256 hash of the error message + the exception
   * class name — but never the raw string.
   * - 'raw': opt-in — restore the legacy verbatim (truncated) error string.
   * Provider 400s / tool errors can echo prompt content, so this
   * re-enables a plaintext path; use only when you trust the sink.
   *
   * An unrecognized value falls back to 'redacted' (never 'raw').
   */
  errorDetail?: ErrorDetailMode;
}

export interface RerouteDirective {
  mode: "dry_run" | "enforce";
  provider: string;
  model: string;
  rule_id: string;
  rule_name?: string;
  original?: { provider?: string; model?: string };
}

export interface CheckResult {
  status: string;
  reason?: string;
  /** On loop_detected blocks: the detector that fired (HASH_CYCLE, SKELETON, GROWTH, CAP, SPAN_NAME_CYCLE). */
  detail?: string;
  /** On loop_detected blocks: the trace_id that was blocked. */
  traceId?: string;
  ruleId?: string;
  /**
   * True when this result is a fail-open fallback (a remote error was swallowed
   * and the call was allowed through). Camel-cased to match `traceId`/`ruleId`.
   */
  failOpen?: boolean;
  /**
   * @deprecated Use {@link failOpen} instead. Retained (and still populated) for
   * backward compatibility; both fields are set together.
   */
  fail_open?: boolean;
  /** Present when a REROUTE rule matched. The SDK acts only when mode === "enforce". */
  reroute?: RerouteDirective;
  [key: string]: unknown;
}

/**
 * Token Police Node.js SDK Client.
 * Provides manual check() and log() methods, plus global configuration.
 */
export class TokenPolice {
  readonly apiKey: string;
  readonly baseUrl: string;
  readonly timeout: number;
  readonly firewall: FirewallMode;
  readonly logErrors: boolean;
  readonly deployment: "daemon" | "serverless" | "edge";
  readonly sseReconnectMaxIntervalSeconds: number;
  readonly streamStaleGraceSeconds: number;
  readonly captureStreamUsage: boolean;
  readonly errorDetail: ErrorDetailMode;
  readonly clientId: string;

  private _pendingPromises: Set<Promise<void>> = new Set();
  private _closed = false;
  // Exposed so init() can wire up the SSE stream after construction.
  _streamClient: StreamClient | null = null;

  constructor(options: TokenPoliceOptions) {
    if (!options.apiKey) {
      throw new Error("Token Police SDK: apiKey is required.");
    }

    if (!/^tp_sk_[a-zA-Z0-9_]+$/.test(options.apiKey)) {
      logger.warning(
        "TokenPolice: apiKey format invalid. Expected 'tp_sk_' prefix followed by alphanumerics.",
      );
    }

    this.apiKey = options.apiKey;
    this.baseUrl = (
      options.baseUrl ||
      process.env.TOKENPOLICE_BASE_URL ||
      "https://collect.tokenpolice.ai"
    ).replace(/\/+$/, "");
    // `timeout` is in SECONDS. A non-positive or non-finite value is a
    // misconfiguration (an explicit instant-abort is never intended), so warn
    // and fall back to the 2.0s default. An omitted value simply uses the
    // default with no warning.
    const t = options.timeout;
    if (t === undefined || t === null) {
      this.timeout = 2.0;
    } else if (typeof t === "number" && Number.isFinite(t) && t > 0) {
      this.timeout = t;
    } else {
      logger.warning(
        `TokenPolice: invalid timeout ${JSON.stringify(t)} (must be > 0 seconds); falling back to 2.0.`,
      );
      this.timeout = 2.0;
    }
    // Three-state firewall mode, default 'dry_run'. Legacy `enforce` boolean
    // is a deprecated alias (true → 'enforce', false → 'off'); `firewall`
    // wins when both are set.
    const resolvedFirewall: string =
      options.firewall ??
      (options.enforce === true
        ? "enforce"
        : options.enforce === false
          ? "off"
          : "dry_run");
    // Validate the resolved mode against the exact canonical set, byte-for-byte
    // (case-sensitive, no trim/lowercase). An unrecognized value falls back to
    // the safe default 'dry_run' — never 'enforce' — so a typo (e.g. "dryrun",
    // "Enforce", " off ") can never silently enable live blocking. Pure local
    // membership logic: it cannot throw or do I/O. The warning fires
    // unconditionally (not gated behind logErrors) — an unrecognized firewall
    // mode is a serious misconfiguration the developer must always see,
    // mirroring the apiKey-format warning above. console.warn is total, so it
    // cannot raise. Covered by firewallModeValidation.test.ts.
    if (resolvedFirewall === "enforce" || resolvedFirewall === "dry_run" || resolvedFirewall === "off") {
      this.firewall = resolvedFirewall;
    } else {
      logger.warning(
        `TokenPolice: unknown firewall mode ${JSON.stringify(resolvedFirewall)}; falling back to 'dry_run'.`,
      );
      this.firewall = "dry_run";
    }
    this.logErrors = options.logErrors ?? false;
    this.deployment = resolveDeployment(options.deployment);
    this.sseReconnectMaxIntervalSeconds = options.sseReconnectMaxIntervalSeconds ?? 300;
    // Stale-stream grace, clamped to [0, 3600] seconds. A non-numeric / NaN /
    // Infinity value is a misconfiguration and falls back to the 60s default
    // rather than silently disabling (or unbounding) the re-verify gate. Pure
    // local arithmetic — cannot throw. Parity: Python clamps identically.
    const g = options.streamStaleGraceSeconds;
    this.streamStaleGraceSeconds =
      typeof g === "number" && Number.isFinite(g) ? Math.min(3600, Math.max(0, g)) : 60;
    this.captureStreamUsage =
      options.captureStreamUsage ?? process.env.TP_CAPTURE_STREAM_USAGE !== "0";
    // Error-detail mode. Coerce any unrecognized/typo value to the safe default
    // 'redacted' (never 'raw') at this constructor choke point, so a mistake
    // can never silently re-open the raw-error plaintext path. Pure local
    // membership logic: it cannot throw or do I/O. Covered by
    // errorMessageScrub.test.ts.
    const ed = options.errorDetail;
    this.errorDetail = ed === "none" || ed === "redacted" || ed === "raw" ? ed : "redacted";
    this.clientId = assignClientId();
  }

  // ── Internal fetch helper ───────────────────────────────────────

  private async _fetch(
    path: string,
    body: Record<string, unknown>,
  ): Promise<Response> {
    const controller = new AbortController();
    const timeoutId = setTimeout(
      () => controller.abort(),
      this.timeout * 1000,
    );

    // On freeze-prone deployments (serverless/edge) the fire-and-forget `/log`
    // POST would be abandoned the instant the handler returns (the container
    // freezes), since Node cannot truly block on a Promise in `flushSync()`
    // without a forbidden sync-blocking primitive. Hand the in-flight request
    // to the runtime via `keepalive: true` so it can outlive the return.
    // Narrowly gated — pure local string comparisons, no I/O, no throw — so
    // `/check` (always awaited) and daemon `/log` (long-lived hot path) stay
    // byte-for-byte unchanged. `keepalive` is a client-side RequestInit hint:
    // the wire request the server sees is identical. Covered by
    // flushSyncKeepalive.test.ts.
    const useKeepalive =
      path === "/v1/guard/log" &&
      (this.deployment === "serverless" || this.deployment === "edge");

    try {
      const response = await fetch(`${this.baseUrl}${path}`, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${this.apiKey}`,
          "Content-Type": "application/json",
          // SDK identification headers — sent with every request so telemetry
          // can be attributed to the SDK version/deployment fleet that
          // produced it.
          "X-TP-Sdk-Version": SDK_VERSION,
          "X-TP-Sdk-Schema-Version": SDK_SCHEMA_VERSION,
          "X-TP-Client-Id": this.clientId,
          "X-TP-Deployment-Mode": this.deployment,
          "X-TP-Firewall-Mode": this.firewall,
        },
        body: JSON.stringify(body),
        signal: controller.signal,
        ...(useKeepalive ? { keepalive: true } : {}),
      });
      return response;
    } finally {
      clearTimeout(timeoutId);
    }
  }

  // ── Synchronous-style API (uses await internally) ─────────────

  /**
   * Pre-flight budget check. Fails open on any error.
   * Returns immediately with a CheckResult.
   */
  async check(
    userId = "anonymous",
    paidPlan = "free",
    workflowName = "default",
    sessionId = "",
    metadata: Record<string, unknown> = {},
    traceId?: string,
    model?: string,
    provider?: string,
    intent?: Record<string, unknown>,
    planSource?: string,
  ): Promise<CheckResult> {
    const payload: Record<string, unknown> = {
      user: {
        id: userId,
        paid_plan: paidPlan,
        // Provenance of `paid_plan` — see `log()`. Omitted when the caller did
        // not thread it (the server reads omission as "unknown"); never guessed.
        ...(planSource === "app" || planSource === "default"
          ? { plan_source: planSource }
          : {}),
      },
      metadata,
    };
    // Surface the positional workflowName into metadata.workflow_name so
    // workflow-scoped rules/budgets and dashboard grouping match this manual
    // call. Fires only for a non-default name the caller has not already placed
    // in metadata; writes to a shallow copy so the caller's object is never
    // mutated, and is guarded so hostile/non-object metadata is a no-op.
    if (workflowName && workflowName !== "default") {
      const md = payload.metadata;
      if (typeof md === "object" && md) {
        try {
          if (!("workflow_name" in md)) {
            payload.metadata = { ...md, workflow_name: workflowName };
          }
        } catch {
          /* fail-safe: leave metadata as-is */
        }
      }
    }
    if (traceId) payload.trace_id = traceId;
    // Send session_id on the pre-flight check so per-session rules can match
    // and group this call. Guarded — never throws.
    if (sessionId) payload.session_id = sessionId;
    if (model || provider) {
      payload.model = { name: model || "", provider: provider || "" };
    }
    if (intent && Object.keys(intent).length > 0) {
      payload.intent = intent;
      const kind = (intent as { kind?: unknown }).kind;
      if (typeof kind === "string" && kind) payload.modality = kind;
    }

    try {
      const response = await this._fetch("/v1/guard/check", payload);
      if (response.status === 429) {
        return (await response.json()) as CheckResult;
      }
      // 200 may carry a `reroute` directive — forward the body so the
      // enforcer can act on it. Fall back to allowed if body is empty/invalid.
      try {
        const body = (await response.json()) as CheckResult;
        if (body && typeof body === "object") {
          return body;
        }
      } catch {
        // empty body or non-JSON — treat as plain allowed
      }
      return { status: "allowed" };
    } catch (e) {
      if (this.logErrors) {
        logger.warning(
          `TokenPolice check failed (fail-open): ${e instanceof Error ? e.message : String(e)}`,
        );
      }
      // Populate both the camelCase `failOpen` and the deprecated snake_case
      // `fail_open` so callers on either field see the fail-open fallback.
      return { status: "allowed", failOpen: true, fail_open: true };
    }
  }

  /**
   * Logs an LLM request as fire-and-forget.
   * The promise is tracked internally so flush() can await it.
   */
  log(
    userId = "anonymous",
    paidPlan = "free",
    workflowName = "default",
    sessionId = "",
    model = "unknown",
    provider = "",
    inputTokens = 0,
    outputTokens = 0,
    cachedTokens = 0,
    metadata: Record<string, unknown> = {},
    span?: Record<string, unknown>,
    promptComposition?: unknown[],
    responseComposition?: unknown[],
    extras?: {
      local_decision?: Record<string, unknown>;
      observations?: unknown[];
      call_outcome?: Record<string, unknown>;
      /**
       * Verbatim provider `usage` object plus a shape identifier and
       * SDK-derived counts/durations. `usage.shape` tells TokenPolice how to
       * interpret `raw` for cost calculation.
       */
      usage?: {
        shape: string;
        raw: unknown;
        tier?: string;
        duration?: Record<string, number>;
        items?: Record<string, number>;
      };
      /**
       * Additional model attribution fields used for accurate pricing:
       * original_provider, endpoint (raw URL path — disambiguates batch vs.
       * realtime APIs), deployment ("byok" | "routed"), framework hint.
       */
      model_extras?: {
        original_provider?: string;
        endpoint?: string;
        deployment?: string;
        framework?: string;
        api_base?: string;
      };
      /**
       * Operation type: "chat" | "embedding" | "image_gen" | "audio_tts" | ...
       * Drives dashboard segmentation and operation-aware rule matching (e.g.
       * embedding traffic such as bulk RAG ingest is treated differently from
       * chat). Default "chat".
       */
      operation?: string;
      /**
       * Tool span metadata (span_kind="tool"). Carries no usage/cost — just
       * name/type/call_id + hashed arg/result sizes. Raw tool args/results
       * never leave the process.
       */
      tool?: {
        name: string;
        type?: string;
        call_id?: string;
        param_hash?: string;
        param_length?: number;
        result_hash?: string;
        result_length?: number;
      };
      /**
       * Client-side latency primitives (TTFT + streaming throughput) for
       * streaming calls; throughput metrics (tokens/sec, time-per-output-token)
       * are derived from these. Omit for non-streaming calls.
       */
      latency?: {
        is_streaming: boolean;
        ttft_ms: number | null;
        total_ms: number;
        generation_ms: number | null;
        output_tokens: number | null;
        clock: string;
      };
      /**
       * Provenance of `paidPlan`: `"app"` (supplied by application code, here
       * or inherited from an ancestor scope) or `"default"` (the SDK
       * synthesized the `"free"` fallback — nobody set a plan). Anything else,
       * including omitting it, drops the field from the payload; the server
       * reads an absent `plan_source` as "unknown". Wire metadata for
       * visibility only — it never affects rule matching or enforcement.
       *
       * Lives in `extras` rather than a 15th positional parameter on purpose:
       * `extras` must remain the LAST argument of `log()` (call spies across
       * the test suite read it as `call.at(-1)`).
       */
      planSource?: string;
    },
  ): void {
    // Silent no-op after close() (fail-open) — mirrors close()'s semantics: once
    // the client is closed, background /log POSTs are no longer issued or tracked
    // (a late log() would otherwise register an untracked promise that no flush
    // will ever await).
    if (this._closed) return;

    const modelBlock: Record<string, unknown> = {
      name: model,
      provider,
    };
    if (extras?.model_extras) {
      if (extras.model_extras.original_provider) modelBlock.original_provider = extras.model_extras.original_provider;
      if (extras.model_extras.endpoint) modelBlock.endpoint = extras.model_extras.endpoint;
      if (extras.model_extras.deployment) modelBlock.deployment = extras.model_extras.deployment;
      if (extras.model_extras.framework) modelBlock.framework = extras.model_extras.framework;
      // Raw serving endpoint (host+path, no query string) — used server-side to
      // identify the actual serving provider (e.g. api.minimax.io → minimax).
      // Advisory only.
      if (extras.model_extras.api_base) modelBlock.api_base = extras.model_extras.api_base;
    }

    const payload: Record<string, unknown> = {
      user: {
        id: userId,
        paid_plan: paidPlan,
        // Only the two canonical values reach the wire; undefined/anything else
        // is omitted so the server can distinguish "not reported" from a claim.
        // Read explicitly off `extras` (which is never spread into the payload,
        // metadata or headers), so `planSource` can only ever surface here.
        ...(extras?.planSource === "app" || extras?.planSource === "default"
          ? { plan_source: extras.planSource }
          : {}),
      },
      model: modelBlock,
      metadata,
      operation: extras?.operation || "chat",
    };

    // Surface the positional workflowName into metadata.workflow_name so
    // workflow-scoped rules/budgets and dashboard grouping match this manual
    // call. Fires only for a non-default name the caller has not already placed
    // in metadata; writes to a shallow copy so the caller's object is never
    // mutated, and is guarded so hostile/non-object metadata is a no-op.
    if (workflowName && workflowName !== "default") {
      const md = payload.metadata;
      if (typeof md === "object" && md) {
        try {
          if (!("workflow_name" in md)) {
            payload.metadata = { ...md, workflow_name: workflowName };
          }
        } catch {
          /* fail-safe: leave metadata as-is */
        }
      }
    }

    // Explicit conversation/session id channel (metadata.session_id is also
    // honored as a fallback). Pure assignment — never throws.
    if (sessionId) payload.session_id = sessionId;

    // Forward the verbatim provider usage object when the caller has it. When
    // absent, synthesize a minimal openai_compatible_chat shape from the
    // positional token counts so the call can still be priced.
    if (extras?.usage) {
      payload.usage = extras.usage;
    } else {
      payload.usage = {
        shape: "openai_compatible_chat",
        raw: {
          prompt_tokens: inputTokens,
          completion_tokens: outputTokens,
          prompt_tokens_details: cachedTokens ? { cached_tokens: cachedTokens } : undefined,
        },
      };
    }

    // Always emit a span block carrying a non-empty span_id. The collector keys
    // BOTH its per-span idempotency guard and the generations row's span_id off
    // this value: with none, a replayed body double-counts budget counters and
    // audit rows, and the row lands with an empty span_id that collapses the
    // trace tree. A caller-supplied span_id — including the deliberately
    // deterministic Anthropic-batch ids that *want* server-side dedup — is
    // forwarded untouched. Never mutates the caller's object (spread copy).
    // Wholly guarded (golden rule): a hostile `span` (throwing proxy/getter)
    // must never surface as an exception in the customer's call.
    try {
      const spanOut: Record<string, unknown> = span ? { ...span } : {};
      if (typeof spanOut.span_id !== "string" || spanOut.span_id === "") {
        spanOut.span_id = randomHex16();
      }
      // Stamp the modality-aware span_kind from `operation` (advisory;
      // recomputed authoritatively server-side). Structural kinds the caller
      // set (agent/tool/chain) are preserved.
      try {
        spanOut.span_kind = spanKindFor(extras?.operation, spanOut.span_kind);
      } catch {
        /* fail-safe: leave whatever the caller set */
      }
      payload.span = spanOut;
    } catch {
      // fail-safe: forward the caller's span verbatim (pre-C-12 behavior); the
      // collector synthesizes a span_id for persistence when it lacks one.
      if (span) payload.span = span;
    }
    if (promptComposition && promptComposition.length > 0) {
      payload.prompt_composition = promptComposition;
    }
    if (responseComposition && responseComposition.length > 0) {
      payload.response_composition = responseComposition;
    }
    if (extras?.local_decision) payload.local_decision = extras.local_decision;
    if (extras?.observations && extras.observations.length > 0) payload.observations = extras.observations;
    if (extras?.call_outcome) payload.call_outcome = extras.call_outcome;
    if (extras?.tool) payload.tool = extras.tool;
    if (extras?.latency) payload.latency = extras.latency;

    const promise = this._fetch("/v1/guard/log", payload)
      .catch((e) => {
        if (this.logErrors) {
          logger.warning(
            `TokenPolice log failed (swallowed): ${e instanceof Error ? e.message : String(e)}`,
          );
        }
      })
      .then(() => {}) as Promise<void>;

    this._pendingPromises.add(promise);
    promise.finally(() => {
      this._pendingPromises.delete(promise);
    });
  }

  /**
   * Asynchronously waits for all pending background log calls to complete.
   * Essential for serverless environments before the container sleeps.
   *
   * Also drains nextTick-deferred auto-instrumented LLM span logs so they
   * can call log() and register POSTs before this returns.
   */
  async flush(): Promise<void> {
    try {
      await forceFlushTokenPoliceSpans();
    } catch {
      /* fail-open */
    }
    const pending = [...this._pendingPromises];
    if (pending.length > 0) {
      await Promise.allSettled(pending);
    }
  }

  /**
   * Synchronous-context hook (diagnostic only — does NOT drain).
   *
   * Node cannot truly block on a Promise synchronously without a forbidden
   * blocking primitive (`deasync`/`execSync`/`Atomics.wait`/sync-XHR), so —
   * unlike the Python SDK's genuinely synchronous `flush_sync` — this cannot
   * await in-flight `/log` POSTs. Instead, serverless/edge telemetry
   * durability is provided at POST-issue time: `_fetch()` attaches
   * `keepalive: true` to the `/log` request so the runtime can complete it
   * after the handler returns / the container freezes. This method only logs
   * a pending-count diagnostic; for a real drain in an async context use
   * `await flush()` or `await close()`/`shutdown()`.
   */
  flushSync(): void {
    if (this._pendingPromises.size > 0) {
      logger.debug(
        `TokenPolice: ${this._pendingPromises.size} pending log(s) at flushSync. Use flush() in async contexts.`,
      );
    }
  }

  /**
   * Closes the client: stops the SSE stream + waits for pending /log POSTs.
   *
   * In Node, the long-lived `fetch` ReadableStream that powers `_streamClient`
   * pins the event loop, so `process.on('beforeExit')` never fires on its own
   * and short-lived scripts hang at exit. Test apps, batch jobs, and CLIs
   * should `await tp.shutdown()` (or `await getClient()?.close()`) before
   * returning to give the process a deterministic exit point.
   *
   * Long-running daemons can ignore this — they want the SSE stream open for
   * the process lifetime, and `beforeExit` cleanup at the bottom of init()
   * already fires when the loop drains.
   */
  async close(): Promise<void> {
    if (this._closed) return;
    // 1) Drain nextTick-deferred LLM logs WHILE still open so log() can
    // register POSTs. Sealing first would make deferred log() a silent no-op.
    try {
      await forceFlushTokenPoliceSpans();
    } catch {
      /* fail-open */
    }
    // 2) Seal: no new untracked POSTs after this point
    this._closed = true;
    try {
      this._streamClient?.stop();
    } catch {
      /* fail-safe */
    }
    // 3) Drain POSTs already registered (do NOT call forceFlush again after
    // seal — deferred log() would no-op)
    const pending = [...this._pendingPromises];
    if (pending.length > 0) {
      await Promise.allSettled(pending);
    }
  }

  /**
   * Synchronous close for cleanup hooks.
   */
  closeSync(): void {
    if (this._closed) return;
    this._closed = true;
    try { this._streamClient?.stop(); } catch { /* fail-safe */ }
    this.flushSync();
  }

  /**
   * Alias for {@link close} — symmetrically named with `tp.shutdown()`. The
   * top-level `shutdown()` export delegates here so customer code reads
   * naturally as "shut down the SDK" rather than "close the SDK".
   */
  async shutdown(): Promise<void> {
    await this.close();
  }
}

// One process-level `beforeExit` cleanup listener for the whole module, however
// many times init() runs — registered lazily on first init (not at import time,
// so a non-Node runtime never pays for it). Re-init must not stack listeners
// (MaxListenersExceededWarning) or retain replaced clients via per-init closures.
let _beforeExitRegistered = false;

// Providers already named in a zero-wrap warning — one warning per provider
// per process, however many times init() runs (see the audit in init()).
const _zeroWrapWarned = new Set<string>();

// ── Global init() function ──────────────────────────────────────────

/**
 * Global configuration.
 * Initializes OpenTelemetry and applies pre-flight wrappers automatically.
 *
 * @example
 * ```typescript
 * import * as tp from 'token-police';
 *
 * tp.init({
 * apiKey: 'tp_sk_your_api_key',
 * baseUrl: 'http://localhost:3001',
 * firewall: 'enforce',
 * });
 * ```
 */
export function init(options: TokenPoliceOptions): TokenPolice {
  let apiKey = options.apiKey;

  if (!apiKey) {
    apiKey = process.env.TOKENPOLICE_API_KEY || "";
  }

  if (!apiKey) {
    throw new Error("TokenPolice SDK: apiKey is required.");
  }

  if (!apiKey.startsWith("tp_sk_")) {
    logger.warning(
      "TokenPolice: apiKey does not start with 'tp_sk_'. It may be invalid.",
    );
  }

  // Warn (once) when init() replaces an already-installed client. The
  // teardown-and-replace in setClient() is intentional (test isolation /
  // hot-reload), but a silent swap hides a double-init bug. Detect the prior
  // instance before constructing/swapping, and wrap the read+warn so a broken
  // logger can never throw out of init() (fail-open).
  try {
    if (getClient()) {
      logger.warning(
        "TokenPolice: init() called again — replacing the previously initialized client.",
      );
    }
  } catch {
    // Silent fail-open — a warning must never crash the customer's init().
  }

  const client = new TokenPolice({ ...options, apiKey });
  setClient(client);

  // 1. Initialize local OpenTelemetry for token extraction.
  // Guarded (defense-in-depth): a provider-construction / OTel global-state
  // failure must degrade to reduced telemetry, never throw into init().
  try {
    setupOpenTelemetry(options.instrumentModules);
  } catch {
    // Silent fail-open — customer app keeps running with reduced telemetry.
  }

  // 2. Install provider taps for ALL firewall modes, including 'off'. The taps
  // are the same in every mode; behavior is decided at CALL time inside the
  // enforcer choke point (`_runAsyncCheck`), which short-circuits to log-only
  // when `firewall === "off"` (no /check, no block, no reroute). So 'off'
  // installs the taps to deliver full telemetry (log-only) for manual-tap
  // providers, while 'enforce'/'dry_run' additionally evaluate.
  // Guarded (defense-in-depth): any setup throw originating outside the
  // per-entry/per-target loops must not escape into the customer's init().
  try {
    autoInstrument(options.instrumentModules);
  } catch {
    // Silent fail-open — enforcement degrades; the app is never crashed.
  }

  // 2b. Zero-wrap audit — LOUD, once per provider per process. A provider SDK
  // that resolves from the APP's node_modules but got ZERO enforcement wraps
  // means enforcement silently degraded: the pre-flight /check never runs for
  // it, calls may be logged under the wrong provider, and (for manual-
  // telemetry providers) cost data is missing entirely. Warned in EVERY
  // firewall mode — attribution/cost corruption matters even with the
  // firewall "off". Providers supplied via instrumentModules wrap normally
  // and stay silent. Fully guarded: the audit itself must never throw into
  // the customer's init() (golden rule).
  try {
    const unwrapped = getUnwrappedResolvableProviders().filter(
      (m) => !_zeroWrapWarned.has(m),
    );
    if (unwrapped.length > 0) {
      for (const m of unwrapped) _zeroWrapWarned.add(m);
      const plural = unwrapped.length > 1;
      console.warn(
        `[TokenPolice Warning] TokenPolice found ${unwrapped.join(", ")} installed in this app ` +
          `but could not attach enforcement to ${plural ? "these packages" : "it"}. ` +
          `Firewall pre-flight checks will not run for ${plural ? "their" : "its"} calls, ` +
          `calls may be logged under the wrong provider, and cost data may be missing. ` +
          `Fix: import the provider SDK and pass the imported module to init() via instrumentModules ` +
          `(e.g. import OpenAI from "openai"; tp.init({ ..., instrumentModules: { openAI: OpenAI } })).`,
      );
    }
  } catch {
    // Silent fail-open — a warning must never break the customer's init().
  }

  // 3. Daemon + wired → start SSE stream. Serverless/edge use inline /check.
  if (client.firewall !== "off" && client.deployment === "daemon") {
    try {
      client._streamClient = new StreamClient({
        baseUrl: client.baseUrl,
        apiKey,
        sdkVersion: SDK_VERSION,
        deployment: client.deployment,
        clientId: client.clientId,
        firewall: client.firewall,
        reconnectCapSeconds: client.sseReconnectMaxIntervalSeconds,
      });
      client._streamClient.start();
    } catch {
      // Silent fail — enforcer falls back to inline /check.
    }
  }

  // 4. Register cleanup on process exit — ONE process-level listener, however
  // many times init() runs, acting on the CURRENT client (read via getClient()
  // at exit time) rather than the init that happened to register it. Without
  // the once-guard, every init() stacked a fresh listener that closed over its
  // own (possibly replaced) client — leaking listeners and retaining dead
  // clients. Registered lazily here (not at import time) and wrapped so it
  // never throws out of init() in a non-Node runtime.
  if (!_beforeExitRegistered) {
    try {
      process.on("beforeExit", () => {
        const c = getClient();
        if (!c) return;
        try { c._streamClient?.stop(); } catch { /* ignored */ }
        c.flush().catch(() => {});
      });
      _beforeExitRegistered = true;
    } catch {
      // May fail in non-Node environments
    }
  }

  return client;
}

// ── Top-level flush helpers ─────────────────────────────────────────

/**
 * Asynchronously wait for all pending telemetry logs to complete.
 */
export async function flush(): Promise<void> {
  const client = getClient();
  if (client) {
    await client.flush();
  }
}

/**
 * Synchronous-context hook (diagnostic only — does NOT drain).
 *
 * Node cannot block on a Promise synchronously, so this does not await
 * in-flight `/log` POSTs. On serverless/edge, durability comes from
 * `keepalive: true` attached to the `/log` request at issue time in `_fetch`,
 * letting the runtime finish it after the handler returns. For a real drain
 * in an async context use `await flush()` or `await close()`/`shutdown()`.
 */
export function flushSync(): void {
  const client = getClient();
  if (client) {
    client.flushSync();
  }
}

/**
 * Stop the SSE stream and wait for pending /log POSTs to drain. Call this
 * in short-lived Node scripts and test apps before returning — without it,
 * the long-lived fetch stream powering the daemon-mode cache pins the
 * event loop and the process hangs at exit. Daemons can ignore.
 *
 * Usage:
 *
 * ```ts
 * await tp.shutdown();
 * ```
 *
 * Equivalent to `await getClient()?.shutdown()` (which itself calls
 * `close()`).
 */
export async function shutdown(): Promise<void> {
  const client = getClient();
  if (client) {
    await client.shutdown();
  }
}
