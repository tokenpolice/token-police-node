/**
 * Framework tool-span capture for agent frameworks that run tools OFF the
 * OpenTelemetry path (so TokenPolice's SpanProcessor never sees them):
 *
 * - OpenAI Agents JS (`@openai/agents`) — records tool calls in its own
 * tracing system as `FunctionSpanData` spans. We register a `TracingProcessor`
 * and emit a TokenPolice `tool` row on each function span.
 * - LlamaIndex JS (`llamaindex`) — executes tools in `@llamaindex/core/agent`
 * `callTool` and surfaces them via `Settings.callbackManager` events
 * (`llm-tool-call` / `llm-tool-result`). We subscribe and emit a row.
 *
 * Both registrations are idempotent, gated on the framework being resolvable
 * from the app, and fully fail-open — a failure must never break the agent run.
 * Tool rows correlate to the current TokenPolice session (same trace as the LLM
 * rows). Kept in parity with the equivalent OpenAI Agents integration in the
 * Python SDK.
 */
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";
import { getClient } from "./state";
import { scrubErrorMessage, resolveErrorDetail, toDurationMs, safeMonoNow } from "./_classify";
import { getCurrentSession, manualSpanIds } from "./context";
import { hashLen } from "./composition";
import { copySessionMetadata } from "./routingMarkerStore";

// OpenAI Agents registration state — strategy in `maybeRegisterOpenAIAgentsTracing`.
let _agentsFallbackKicked = false;
/**
 * `@openai/agents-core` >= 0.4.0 holds its TraceProvider on `globalThis` under
 * this symbol, shared by the CJS and ESM builds (older releases keep one
 * module-level singleton PER build). Registering per module handle therefore
 * attached one processor per build to the same provider → every function span
 * emitted two tool rows (verification runs #24/#25, G1-24-1).
 */
const AGENTS_PROVIDER_SYMBOL = Symbol.for("openai.agents.core.traceProvider");
// Exactly one TokenPolice processor per TraceProvider INSTANCE.
const _agentsProcessorByProvider = new WeakMap<object, AgentsProcessor>();
// Pre-0.4 fallback only (handle exposes no provider): de-dupe by fn identity.
const _registeredAgentsFns = new Set<unknown>();
const _agentsProcessors: unknown[] = []; // keep refs so they aren't GC'd

interface AgentsProcessor {
  evicted: boolean;
  start(): void;
  onTraceStart(): Promise<void>;
  onTraceEnd(): Promise<void>;
  onSpanStart(): Promise<void>;
  onSpanEnd(span: any): Promise<void>;
  shutdown(): Promise<void>;
  forceFlush(): Promise<void>;
}

function makeAgentsProcessor(): AgentsProcessor {
  const processor: AgentsProcessor = {
    evicted: false,
    start() {},
    async onTraceStart() {},
    async onTraceEnd() {},
    async onSpanStart() {},
    async onSpanEnd(span: any) {
      handleAgentsSpanEnd(span);
    },
    // `setTraceProcessors()` — the app's own call, or the umbrella
    // `@openai/agents` package's module init — REPLACES the provider's list and
    // shuts down whatever it dropped. Remember the eviction so the next
    // per-call check re-registers instead of going silent for the process.
    async shutdown() {
      processor.evicted = true;
    },
    async forceFlush() {},
  };
  return processor;
}

/** The TraceProvider a module handle feeds, if it exposes one. Fail-open. */
function _resolveAgentsProvider(mod: any): object | undefined {
  try {
    const p = mod?.getGlobalTraceProvider?.();
    return p && typeof p === "object" ? p : undefined;
  } catch {
    return undefined;
  }
}

/** One-line debug, gated on the client's `logErrors` flag (off by default). */
function _debug(msg: string): void {
  try {
    if ((getClient() as any)?.logErrors) {
      // eslint-disable-next-line no-console
      console.log(`[TokenPolice Debug] ${msg}`);
    }
  } catch {
    /* ignore */
  }
}

/** CommonJS require rooted at the app's node_modules, falling back to the SDK's. */
function makeAppRequire(): NodeRequire {
  try {
    return createRequire(join(process.cwd(), "node_modules", "_"));
  } catch {
    return require;
  }
}

function cjsRequire(pkg: string, req: NodeRequire = makeAppRequire()): any | undefined {
  try {
    return req(pkg);
  } catch {
    return undefined;
  }
}

/**
 * A require that resolves `@openai/agents-core` even when it is not hoisted to
 * the app root (pnpm): try the app root, else root at the umbrella
 * `@openai/agents` package's own directory. `resolve` only locates the umbrella
 * — it never executes it (see `maybeRegisterOpenAIAgentsTracing` for why that
 * matters). Fail-open to the app require.
 */
function agentsCoreRequire(): NodeRequire {
  const appReq = makeAppRequire();
  try {
    appReq.resolve("@openai/agents-core");
    return appReq;
  } catch {
    /* not resolvable from the app root — try via the umbrella's location */
  }
  try {
    return createRequire(appReq.resolve("@openai/agents"));
  } catch {
    return appReq;
  }
}

/**
 * Pick the ESM entry (relative path) from a package.json `exports["."]` node:
 * prefer the `import` condition, then `default` (the ESM fallback for dual
 * packages), recursing through nested condition objects. Exported for tests.
 */
export function pickEsmEntry(node: any): string | undefined {
  if (!node) return undefined;
  if (typeof node === "string") return node;
  return pickEsmEntry(node.import) ?? pickEsmEntry(node.default);
}

/**
 * Resolve a package's ESM-entry **absolute file URL** from the APP's
 * node_modules. We must NOT use a bare `import(pkg)` from this module: the SDK
 * is typically symlinked into the app (`file:` dep), Node realpaths it, and the
 * framework packages live only under the APP's node_modules — so a bare import
 * resolved from the SDK's real dir fails. Resolving the absolute ESM URL and
 * importing that loads the SAME module instance the app loaded (cache-shared
 * under `tsx`), i.e. the instance whose trace-provider / Settings singleton the
 * running agent actually uses. Fail-open → undefined.
 */
function appEsmEntryUrl(pkg: string, req?: NodeRequire): string | undefined {
  try {
    const appReq = req ?? makeAppRequire();
    // `<pkg>/package.json` isn't always an exported subpath — walk up from the
    // resolved (CJS) entry to the package root whose package.json name === pkg.
    let dir = dirname(appReq.resolve(pkg));
    let pkgJsonPath: string | undefined;
    for (let i = 0; i < 10; i++) {
      const cand = join(dir, "package.json");
      try {
        if (JSON.parse(readFileSync(cand, "utf8")).name === pkg) {
          pkgJsonPath = cand;
          break;
        }
      } catch {
        /* not here / unreadable — keep walking */
      }
      const up = dirname(dir);
      if (up === dir) break;
      dir = up;
    }
    if (!pkgJsonPath) return undefined;
    const pkgDir = dirname(pkgJsonPath);
    const pj = JSON.parse(readFileSync(pkgJsonPath, "utf8"));
    const rel = pickEsmEntry(pj.exports?.["."]) ?? pj.module ?? pj.main;
    if (!rel || typeof rel !== "string") return undefined;
    return pathToFileURL(join(pkgDir, rel)).href;
  } catch {
    return undefined;
  }
}

/** Import a package's app-resolved ESM build (absolute URL). Fail-open. */
function esmImport(pkg: string, req?: NodeRequire): Promise<any | undefined> {
  const url = appEsmEntryUrl(pkg, req);
  if (!url) return Promise.resolve(undefined);
  return import(url).catch(() => undefined);
}

// `hashLen` (sha1-16 + code-point length for a tool arg/result) is the single
// shared copy in composition.ts — imported above.

export interface ToolRowInput {
  name: string;
  callId?: string;
  input?: unknown;
  output?: unknown;
  startedAtMs?: number;
  endedAtMs?: number;
  /**
   * Preferred measured duration in ms (may be fractional). When set, used for
   * call_outcome.duration_ms instead of endedAtMs-startedAtMs so sub-ms tools
   * measured with a mono clock are not truncated to 0.
   */
  durationMs?: number;
  failed?: boolean;
  errorMessage?: string;
}

/**
 * Emit a zero-usage `tool` row into the active session's trace. Values-based
 * analog of `telemetry._logToolSpan` (which reads from an OTel span). Fail-open.
 */
export function emitToolRow(t: ToolRowInput): void {
  try {
    const tp = getClient();
    if (!tp) return;

    const session = getCurrentSession();
    const ids = manualSpanIds(session);
    const name = t.name || "tool";
    // Prefer an explicit callId (agents/app-supplied). Otherwise FIFO pop from
    // the pending stash filled by the preceding LLM response capture (covers
    // OpenAI Agents + LlamaIndex wrap with no app code change). Never invent.
    let callId = t.callId ?? "";
    if (!callId) {
      try {
        callId = session.popPendingToolCallId(name) || "";
      } catch {
        callId = "";
      }
    }
    const [paramHash, paramLen] = hashLen(t.input);
    const [resultHash, resultLen] = hashLen(t.output);

    const metadata: Record<string, unknown> = {
      workflow_name: session.workflowName,
    };
    if (session.sessionId) metadata.session_id = session.sessionId;
    // B4: framework `tool` rows execute no model call — the reroute marker on
    // the session belongs to whichever LLM call was rerouted, never to this
    // row. Copy without it; never re-add.
    if (session.metadata) {
      copySessionMetadata(metadata, session.metadata);
    }

    const end = t.endedAtMs ?? Date.now();
    const start = t.startedAtMs ?? end;
    const hasExplicitDuration = typeof t.durationMs === "number" && Number.isFinite(t.durationMs);
    // Prefer an explicit mono/fractional duration when the wrapper measured one
    // (LlamaIndex wrap); otherwise fall back to wall start/end delta (Agents
    // ISO stamps are ms-truncated — same-ms collisions are floored to 1, see N6).
    const rawDuration = hasExplicitDuration ? (t.durationMs as number) : end - start;
    // N6: two genuinely observed but ms-truncated stamps landing on the SAME ms
    // mean the tool ran in <1ms — "ran but fast" (report 1, the convention)
    // rather than "no duration recorded" (0). Narrow gate: only when no explicit
    // duration was measured AND both stamps were actually supplied. A missing
    // start (start defaults to end) still reports 0, a missing end keeps the
    // historical emit-time wall delta, and a negative delta still falls
    // through to toDurationMs → 0.
    const sameMsStamps =
      !hasExplicitDuration &&
      typeof t.startedAtMs === "number" &&
      Number.isFinite(t.startedAtMs) &&
      typeof t.endedAtMs === "number" &&
      Number.isFinite(t.endedAtMs) &&
      t.endedAtMs - t.startedAtMs === 0;
    const callOutcome: Record<string, unknown> = {
      status: t.failed ? "failed" : "success",
      // Positive sub-ms → 1 via toDurationMs.
      duration_ms: sameMsStamps ? 1 : toDurationMs(rawDuration),
    };
    if (t.failed && t.errorMessage) {
      // Route the raw value through the central scrub helper (no pre-
      // stringify); default 'redacted' ships a hash, not the raw string.
      Object.assign(callOutcome, scrubErrorMessage(t.errorMessage, resolveErrorDetail()));
    }

    const spanObj = {
      trace_id: ids.trace_id,
      span_id: ids.span_id,
      parent_span_id: ids.parent_span_id,
      span_kind: "tool" as const,
      span_name: name,
      span_order: 0,
      start_time: new Date(start).toISOString(),
      end_time: new Date(end).toISOString(),
    };

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
      spanObj,
      [],
      [],
      {
        tool: {
          name,
          type: "function",
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
    // fail-open: a tool row must never break the agent run
  }
}

/**
 * Handle one OpenAI Agents JS span end — emit a tool row only for function
 * (tool) spans; ignore LLM (`response`/`generation`) spans (already captured
 * via the Responses wrapper → no double-count). Exported for unit testing.
 */
export function handleAgentsSpanEnd(span: any): void {
  try {
    const sd = span?.spanData;
    if (!sd || sd.type !== "function") return; // only tool/function spans
    // FunctionSpanData has no callId today; read if a future SDK adds it.
    // Otherwise emitToolRow pops from the pending stash by tool name.
    const callId = sd.callId ?? sd.call_id ?? sd.toolCallId ?? undefined;
    emitToolRow({
      name: String(sd.name ?? "tool"),
      callId: callId != null && callId !== "" ? String(callId) : undefined,
      input: sd.input,
      output: sd.output,
      startedAtMs: span?.startedAt ? Date.parse(span.startedAt) : undefined,
      endedAtMs: span?.endedAt ? Date.parse(span.endedAt) : undefined,
      failed: !!span?.error,
      errorMessage: span?.error?.message,
    });
  } catch {
    /* fail-open */
  }
}

/**
 * Register our processor on one resolved `@openai/agents-core` module handle.
 * De-duped by the TraceProvider INSTANCE the handle feeds
 * (`mod.getGlobalTraceProvider()`): on agents-core >= 0.4 the CJS and ESM
 * builds are two `addTraceProcessor` identities over ONE globalThis provider,
 * so identity de-dupe registered twice; on older releases each build has its
 * own provider and both legitimately get a processor. A handle that exposes no
 * provider falls back to `addTraceProcessor` identity. Exported for unit testing.
 */
export function registerAgentsOn(mod: any): boolean {
  const fn = mod?.addTraceProcessor;
  if (typeof fn !== "function") return false;
  const provider = _resolveAgentsProvider(mod);
  if (provider) {
    const existing = _agentsProcessorByProvider.get(provider);
    if (existing && !existing.evicted) return false;
  } else if (_registeredAgentsFns.has(fn)) {
    return false;
  }
  const processor = makeAgentsProcessor();
  // Record before attaching so a handle that throws is never retried.
  if (provider) _agentsProcessorByProvider.set(provider, processor);
  else _registeredAgentsFns.add(fn);
  try {
    fn.call(mod, processor); // appends — leaves the user's exporter intact
    _agentsProcessors.push(processor);
    _debug("registered OpenAI Agents tool-span processor");
    return true;
  } catch {
    return false;
  }
}

/** Register directly on a TraceProvider instance (no module handle). */
function registerOnAgentsProvider(provider: any): boolean {
  if (!provider || typeof provider.registerProcessor !== "function") return false;
  const existing = _agentsProcessorByProvider.get(provider);
  if (existing && !existing.evicted) return false;
  const processor = makeAgentsProcessor();
  _agentsProcessorByProvider.set(provider, processor);
  try {
    provider.registerProcessor(processor); // appends — leaves the user's exporter intact
    _agentsProcessors.push(processor);
    _debug("registered OpenAI Agents tool-span processor on the global trace provider");
    return true;
  } catch {
    return false;
  }
}

/**
 * Register a TokenPolice `TracingProcessor` with the OpenAI Agents JS SDK so
 * function/tool spans become tool rows. Idempotent and cheap enough for the
 * per-call hot path; no-op if `@openai/agents` isn't in use.
 *
 * 1. agents-core >= 0.4: the provider the running agent emits through lives on
 *    `globalThis[Symbol.for("openai.agents.core.traceProvider")]`, shared by
 *    the CJS and ESM builds. Register there directly — no module loading, so
 *    this can neither double up across builds nor pull a second build into the
 *    app. Re-checked per call (one symbol lookup + WeakMap get) so a processor
 *    evicted by `setTraceProcessors()` is re-attached on the next LLM call.
 * 2. No global provider (agents-core < 0.4 keeps one singleton PER build, or the
 *    app has not created it yet): one-shot fallback that registers on the CJS
 *    `require` and the ESM `import()` of `@openai/agents-core` — the build the
 *    running agent uses is one of them, and `registerAgentsOn` de-dupes by
 *    provider instance if they turn out to share one.
 *
 * Only `@openai/agents-core` is ever loaded, NEVER the umbrella `@openai/agents`
 * (its `addTraceProcessor` is core's re-export anyway). The umbrella's module
 * init runs `setDefaultOpenAITracingExporter()` + `setDefaultModelProvider()`;
 * requiring its CJS build into an ESM app executes that init a second time on
 * the SHARED provider, which replaces every registered processor (the
 * customer's own included) and swaps the default model provider. Mirrors the
 * Python SDK, which registers only when `agents` is already in `sys.modules`.
 */
export function maybeRegisterOpenAIAgentsTracing(): void {
  try {
    const globalProvider = (globalThis as any)[AGENTS_PROVIDER_SYMBOL];
    if (globalProvider && typeof globalProvider.registerProcessor === "function") {
      registerOnAgentsProvider(globalProvider);
      return;
    }
    if (_agentsFallbackKicked) return;
    _agentsFallbackKicked = true;
    const req = agentsCoreRequire();
    registerAgentsOn(cjsRequire("@openai/agents-core", req));
    // `.catch` is defense-in-depth: esmImport already swallows its own errors,
    // but this guarantees a rejected import promise can never surface as a
    // process-killing unhandled rejection if that internal contract changes.
    void esmImport("@openai/agents-core", req).then(registerAgentsOn).catch(() => {});
  } catch {
    /* fail-open */
  }
}

/**
 * Wrap LlamaIndex tool objects' `.call` so each execution emits a TokenPolice
 * tool row.
 *
 * Apps commonly run a **manual tool loop** — `llm.chat({ messages, tools })`,
 * then they invoke `await tool.call(input)` themselves — which never goes
 * through LlamaIndex's `agent()`/`AgentWorkflow`, so the `Settings.callbackManager`
 * `llm-tool-*` events are never dispatched. The reliable hook is the tool object
 * itself: TokenPolice already intercepts `llm.chat(...)`, whose params carry the
 * `tools` array, so we wrap each tool's own `.call` (an own per-instance closure,
 * not on the prototype). De-duped per tool object via a WeakSet. This also covers
 * `AgentWorkflow` (its `callTool` invokes the same `tool.call(input)`).
 *
 * Transparent: returns the original result / re-raises unchanged, and — crucially
 * — PRESERVES synchronicity. A SYNC `tool.call` must keep returning its value
 * directly (not a Promise), or a customer's `const v = tool.call(x)` silently
 * breaks. The wrapper is therefore a plain (non-async) function that detects a
 * thenable result and only defers the row when the original itself is async.
 * Fail-open.
 */
const _wrappedTools = new WeakSet<object>();

/** Best-effort row emit that can never throw into the customer's tool path. */
function safeEmit(row: ToolRowInput): void {
  try {
    emitToolRow(row);
  } catch {
    /* emitToolRow is already fail-open; this guard is a second layer so a row
       emit can never corrupt a resolved value or mask the customer's error */
  }
}

export function wrapLlamaIndexTools(tools: any): void {
  try {
    if (!Array.isArray(tools)) return;
    for (const t of tools) {
      // Per-tool isolation: a frozen/sealed tool (assignment throws) must NOT
      // abort wrapping the remaining tools. Each tool wraps in its own scope.
      try {
        if (!t || typeof t !== "object" || typeof t.call !== "function") continue;
        if (_wrappedTools.has(t)) continue;
        const original = t.call;
        const name = String(t?.metadata?.name ?? t?.name ?? "tool");
        t.call = function (this: any, input: any, ...rest: any[]) {
          // Wall for ISO stamps; mono for duration so sub-ms tools report >=1.
          const startedAtMs = Date.now();
          const startMono = safeMonoNow();
          const elapsed = () => safeMonoNow() - startMono;
          let result: any;
          try {
            result = original.apply(this, [input, ...rest]);
          } catch (e: any) {
            // Sync throw: emit a failed row (guarded), then rethrow unchanged.
            safeEmit({
              name,
              input,
              startedAtMs,
              durationMs: elapsed(),
              failed: true,
              errorMessage: String(e?.message ?? e),
            });
            throw e;
          }
          if (result && typeof result.then === "function") {
            // Async path: identical behavior to the previous async wrapper,
            // including rejection propagation and row content.
            return result.then(
              (ok: any) => {
                safeEmit({
                  name,
                  input,
                  output: ok,
                  startedAtMs,
                  durationMs: elapsed(),
                  failed: false,
                  errorMessage: "",
                });
                return ok;
              },
              (err: any) => {
                safeEmit({
                  name,
                  input,
                  startedAtMs,
                  durationMs: elapsed(),
                  failed: true,
                  errorMessage: String(err?.message ?? err),
                });
                throw err;
              },
            );
          }
          // Sync return: emit synchronously (guarded) and hand back the value AS-IS.
          safeEmit({
            name,
            input,
            output: result,
            startedAtMs,
            durationMs: elapsed(),
            failed: false,
            errorMessage: "",
          });
          return result;
        };
        // Mark wrapped only AFTER the assignment succeeds, so a frozen tool that
        // threw above is not falsely recorded as wrapped.
        _wrappedTools.add(t);
      } catch {
        /* skip this tool; the rest of the array still wraps */
      }
    }
    _debug("wrapped LlamaIndex tool.call");
  } catch {
    /* fail-open: a cosmetic tool row must never break the agent */
  }
}
