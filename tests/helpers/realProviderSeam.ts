/**
 * Shared harness for the `realProviderSeams.*.test.ts` family — the only Node
 * tests that load the REAL provider packages instead of a hand-built fake.
 *
 * WHY THESE TESTS EXIST
 * ---------------------
 * Both SDKs deliberately pin no provider version: the customer owns the
 * installed `openai` / `@anthropic-ai/sdk` / `@google/genai` / `ai` copy and the
 * enforcer instruments whatever shape it finds at runtime. The cost of that
 * policy is that a provider release can rename an export, move a method off a
 * prototype, or change a usage field and our instrumentation silently stops
 * producing rows — with no signal on our side.
 *
 * `provider-drift-suite/` (tier 1) exists to raise that alarm nightly, but it
 * can only see what the SDK's own unit tests can see. Every Node provider test
 * before this family drove a hand-built fake of the export shape, and a fake
 * cannot drift — so the Node half of the matrix was marked `fake-only` and
 * skipped. These files close that gap: they load the real package, walk the
 * enforcer's OWN patch specs against it, and run one non-stream + one stream
 * call end to end with a stubbed transport.
 *
 * Layered on purpose, because the two layers fail for different reasons:
 *   1. SHAPE      — the export path the enforcer patches still exists and is a
 *                   function. Derived from `__test__._TARGET_METHODS` (and the
 *                   real `_instrument*` helpers) so the assertion cannot drift
 *                   away from the code it guards.
 *   2. ROUND TRIP — a real client + a canned wire response carrying known token
 *                   counts produces a /log payload with exactly those counts,
 *                   the right model, and the right operation. Shape can be
 *                   right while the usage field the extractor reads has been
 *                   renamed; only the round trip catches that.
 *
 * HARD RULES
 * ----------
 * - NO NETWORK. Every client is constructed with a stubbed transport, and
 *   `installNoNetworkGuard()` replaces `globalThis.fetch` plus
 *   `http(s).request` with throwers so an un-stubbed call fails loudly here
 *   rather than reaching a provider.
 * - NO API KEYS. Every credential in these files is a literal placeholder.
 * - GOLDEN RULE. Nothing here may exercise a path where a TokenPolice failure
 *   surfaces to the customer as anything but a block on `firewall: "enforce"`.
 *   The harness client therefore runs `firewall: "dry_run"` (pre-flight really
 *   executes, never blocks) unless a test is specifically about enforcement.
 */
import { createRequire } from "module";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { setClient } from "../../src/state";
import { __test__ } from "../../src/enforcer";
import { resolveInstalledVersion } from "../../src/telemetry";

export const requireCjs = createRequire(import.meta.url);

// ───────────────────────────── log capture ─────────────────────────────

/**
 * One captured `tp.log(...)` call, named. The positional signature is
 * enforcer.ts `_logManual` / telemetry.ts `onEnd`:
 *   log(userId, paidPlan, workflowName, sessionId, model, provider,
 *       inputTokens, outputTokens, cachedTokens, metadata, span,
 *       promptComposition, responseComposition, extra)
 * Naming it here keeps every assertion in the seam files readable and means a
 * future signature change breaks in ONE place instead of twelve.
 */
export interface CapturedLog {
  userId: unknown;
  paidPlan: unknown;
  workflowName: unknown;
  sessionId: unknown;
  model: string;
  provider: string;
  inputTokens: number;
  outputTokens: number;
  cachedTokens: number;
  metadata: any;
  span: any;
  promptComposition: any;
  responseComposition: any;
  extra: any;
  /** The raw positional array, for assertions the named view doesn't cover. */
  args: any[];
}

function toCaptured(args: any[]): CapturedLog {
  return {
    userId: args[0],
    paidPlan: args[1],
    workflowName: args[2],
    sessionId: args[3],
    model: args[4],
    provider: args[5],
    inputTokens: args[6],
    outputTokens: args[7],
    cachedTokens: args[8],
    metadata: args[9],
    span: args[10],
    promptComposition: args[11],
    responseComposition: args[12],
    extra: args[13],
    args,
  };
}

export interface SeamHarness {
  /** Every captured log, in emission order. */
  logs: CapturedLog[];
  /** Every `/check` payload the pre-flight sent. */
  checks: any[][];
  /** Install a capturing client. Call in `beforeEach`. */
  install(overrides?: Record<string, any>): void;
  /** Drop the client and clear captures. Call in `afterEach`. */
  reset(): void;
  /** The most recent log; throws with context when nothing was logged. */
  last(): CapturedLog;
  /** The single log matching `pred`; throws when 0 or >1 match. */
  only(pred: (l: CapturedLog) => boolean, what: string): CapturedLog;
}

/**
 * A capturing TokenPolice client wired into the real `src/state` singleton.
 *
 * Deliberately NOT a `vi.mock("../src/state")`: these files exercise the real
 * instrumentation end to end, and a partially-mocked state module would let a
 * genuine regression in observation/session plumbing pass unnoticed.
 */
export function createSeamHarness(): SeamHarness {
  const logs: CapturedLog[] = [];
  const checks: any[][] = [];
  return {
    logs,
    checks,
    install(overrides: Record<string, any> = {}): void {
      logs.length = 0;
      checks.length = 0;
      setClient({
        enforce: false,
        // dry_run: the pre-flight /check really runs (so the seam under test is
        // the same code path a customer hits) but can never block.
        firewall: "dry_run",
        logErrors: false,
        check: async (...args: any[]) => {
          checks.push(args);
          return { status: "allowed" };
        },
        log: (...args: any[]) => {
          logs.push(toCaptured(args));
        },
        ...overrides,
      } as any);
    },
    reset(): void {
      logs.length = 0;
      checks.length = 0;
      setClient(undefined as any);
    },
    last(): CapturedLog {
      if (logs.length === 0) {
        throw new Error(
          "expected at least one tp.log(...) call — the enforcer produced NO " +
            "telemetry for this call, which is the exact symptom of a seam that " +
            "stopped attaching",
        );
      }
      return logs[logs.length - 1];
    },
    only(pred: (l: CapturedLog) => boolean, what: string): CapturedLog {
      const hits = logs.filter(pred);
      if (hits.length !== 1) {
        throw new Error(
          `expected exactly one ${what} row, got ${hits.length}. All rows: ` +
            JSON.stringify(
              logs.map((l) => ({
                model: l.model,
                provider: l.provider,
                op: l.extra?.operation,
                shape: l.extra?.usage?.shape,
              })),
            ),
        );
      }
      return hits[0];
    },
  };
}

/** Settle the enforcer's deferred log paths (microtasks + a macrotask turn). */
export async function flushLogs(ms = 20): Promise<void> {
  await new Promise((r) => setTimeout(r, ms));
}

// ─────────────────────────── stubbed transports ───────────────────────────

/** Minimal fetch-alike: always answers with `body` as JSON. */
export function jsonResponder(
  body: unknown,
  init: { status?: number; headers?: Record<string, string> } = {},
): (...args: any[]) => Promise<Response> {
  return async () =>
    new Response(JSON.stringify(body), {
      status: init.status ?? 200,
      headers: { "content-type": "application/json", ...(init.headers ?? {}) },
    });
}

/**
 * Minimal SSE fetch-alike. `events` are emitted as `data: <json>` frames in
 * order; `done` appends OpenAI's `data: [DONE]` sentinel (Anthropic's wire
 * format has no such sentinel, hence the flag).
 */
export function sseResponder(
  events: unknown[],
  opts: { done?: boolean; headers?: Record<string, string>; eventNames?: boolean } = {},
): (...args: any[]) => Promise<Response> {
  const frames = events.map((e) => {
    const json = typeof e === "string" ? e : JSON.stringify(e);
    // Anthropic's stream requires the `event:` line; OpenAI's tolerates it.
    if (opts.eventNames && typeof e === "object" && e && "type" in (e as any)) {
      return `event: ${(e as any).type}\ndata: ${json}\n\n`;
    }
    return `data: ${json}\n\n`;
  });
  if (opts.done !== false && !opts.eventNames) frames.push("data: [DONE]\n\n");
  const payload = frames.join("");
  return async () =>
    new Response(payload, {
      status: 200,
      headers: { "content-type": "text/event-stream", ...(opts.headers ?? {}) },
    });
}

/**
 * A `globalThis.fetch` stub for SDKs that offer no constructor-level transport
 * injection (`@google/genai`, `cohere-ai`, `@mistralai/mistralai`,
 * `@huggingface/inference`, `@openrouter/sdk`, `voyageai`).
 *
 * Routes are matched by substring against the request URL, in declaration
 * order. An unmatched URL THROWS — a silent fallthrough to the real network is
 * exactly what this harness must never allow.
 */
export function routedGlobalFetch(
  routes: Array<{ match: string; respond: (...a: any[]) => Promise<Response> }>,
): { urls: string[]; restore: () => void } {
  const urls: string[] = [];
  const real = globalThis.fetch;
  globalThis.fetch = (async (input: any, init?: any): Promise<Response> => {
    const url = String(input?.url ?? input ?? "");
    urls.push(url);
    const hit = routes.find((r) => url.includes(r.match));
    if (!hit) {
      throw new Error(
        `[no-network guard] un-stubbed fetch to ${url} — add a route or fix ` +
          `the stub; these tests must never reach a provider`,
      );
    }
    return hit.respond(input, init);
  }) as any;
  return { urls, restore: () => { globalThis.fetch = real; } };
}

/**
 * Hard no-network guard. Replaces `globalThis.fetch` AND `http/https.request`
 * (node-fetch v2, which `voyageai` bundles, bypasses global fetch entirely) so
 * an un-stubbed provider call fails loudly inside the test instead of leaving
 * the process. Install once per file in `beforeAll`; restore in `afterAll`.
 */
export function installNoNetworkGuard(): () => void {
  const realFetch = globalThis.fetch;
  const boom = (where: string) => (...args: any[]): never => {
    const target = String(args?.[0]?.href ?? args?.[0]?.hostname ?? args?.[0] ?? "");
    throw new Error(
      `[no-network guard] ${where} attempted a real request to "${target}" — ` +
        `realProviderSeams tests must stub every transport`,
    );
  };
  globalThis.fetch = boom("fetch") as any;

  // The CJS module objects, not the ESM namespaces — an `import * as http`
  // namespace is frozen, so assigning to it throws. Guarded per module: a
  // runtime that refuses the patch must not take the whole guard down with it,
  // and `fetch` alone already covers every SDK here except node-fetch v2.
  const restores: Array<() => void> = [() => { globalThis.fetch = realFetch; }];
  for (const mod of ["node:http", "node:https"]) {
    try {
      const m: any = requireCjs(mod);
      const original = m.request;
      m.request = boom(`${mod}.request`);
      restores.push(() => { m.request = original; });
    } catch {
      /* not patchable in this runtime — fetch guard still stands */
    }
  }
  return () => {
    for (const r of restores) {
      try {
        r();
      } catch {
        /* best effort */
      }
    }
  };
}

// ─────────────────────── enforcer spec-table access ───────────────────────

export interface TargetSpec {
  moduleName: string;
  objectPath: string[];
  method: string;
  isAsync: boolean;
  manualTelemetry?: boolean;
  streaming?: boolean;
  provider?: string;
  modality?: string;
  shape?: string;
  operation?: string;
}

/**
 * The enforcer's OWN registry, read live from `__test__._TARGET_METHODS`.
 *
 * Asserting against this rather than a copied list is the whole point: a row
 * added to `src/enforcer.ts` is automatically covered by the shape layer, and a
 * row deleted there stops being asserted — the test can never claim to guard a
 * seam the enforcer no longer patches.
 */
export function targetsFor(moduleName: string): TargetSpec[] {
  const all = (__test__ as any)._TARGET_METHODS as TargetSpec[];
  return all.filter((t) => t.moduleName === moduleName);
}

/** Human label for a target, matching the enforcer's own warning text. */
export function targetLabel(t: TargetSpec): string {
  return `${t.moduleName} ${t.objectPath.join(".")}.${t.method}`;
}

/** `_resolvePath`'s walk, duplicated here so the test can report WHERE it broke. */
export function walkPath(root: any, path: string[]): { obj: any; brokeAt: string | null } {
  let obj = root;
  for (const key of path) {
    if (obj == null) return { obj: null, brokeAt: key };
    obj = obj[key];
  }
  return { obj, brokeAt: obj == null ? path[path.length - 1] : null };
}

/** Every function-valued own property on `obj`, for failure messages. */
export function methodNames(obj: any): string[] {
  try {
    return Object.getOwnPropertyNames(obj).filter((n) => {
      try {
        return typeof obj[n] === "function";
      } catch {
        return false;
      }
    });
  } catch {
    return [];
  }
}

/**
 * True when `pkg` is a declared devDependency of token-police-node.
 *
 * The seam files gate on "did the package load?", and a bare gate turns a real
 * LOAD FAILURE into a silent skip — or, with an `if (!available) return`, into a
 * vacuous PASS. The same gate is legitimate when the package genuinely is not
 * installed. This tells the two apart: every provider these files exercise is a
 * declared devDependency here, so "declared but did not load" is a hard failure
 * and "not declared" is the only case in which skipping is honest.
 */
export function declaredDevDependency(pkg: string): boolean {
  try {
    const here = dirname(fileURLToPath(import.meta.url));
    const manifest = JSON.parse(
      readFileSync(join(here, "..", "..", "package.json"), "utf8"),
    );
    return Boolean(manifest?.devDependencies?.[pkg]);
  } catch {
    return false;
  }
}

/**
 * Installed version of a package, or undefined when it isn't present.
 *
 * Uses the SDK's own `resolveInstalledVersion`, not a bare
 * `require("<pkg>/package.json")`: several providers (openai among them) omit
 * "./package.json" from their exports map, so the direct require throws
 * ERR_PACKAGE_PATH_NOT_EXPORTED and the naive read reports every one of them as
 * absent.
 */
export function installedVersion(pkg: string): string | undefined {
  return resolveInstalledVersion(requireCjs, pkg);
}

/**
 * Snapshot the current function identity at each of `targets`, for a
 * before/after comparison around `autoInstrument()`. Identity change is the
 * only honest proof the wrapper landed: `_wrapMethod` fails SILENTLY (a
 * `logErrors`-gated console.warn) when a path or method has moved, which is
 * precisely the drift these files exist to catch.
 */
export function snapshotMethods(
  root: any,
  targets: TargetSpec[],
): Map<string, unknown> {
  const snap = new Map<string, unknown>();
  for (const t of targets) {
    const { obj } = walkPath(root, t.objectPath);
    snap.set(targetLabel(t), obj?.[t.method]);
  }
  return snap;
}
