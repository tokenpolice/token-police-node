/**
 * REAL-PACKAGE seam contract — `@openai/agents`.
 *
 * THIN BY DESIGN (long-tail framework): the one export the integration hangs
 * on, plus proof that our processor actually registers and receives spans.
 *
 * OpenAI Agents JS runs tool calls through its OWN tracing stack, not OTel, so
 * the only way TokenPolice sees a tool span is the TraceProvider — reached via
 * `addTraceProcessor(...)` (a single named export on both `@openai/agents` and
 * `@openai/agents-core`, both builds) or, on agents-core >= 0.4, directly on
 * `globalThis[Symbol.for("openai.agents.core.traceProvider")]`, the ONE
 * provider both builds share. If that export is renamed or the provider stops
 * exposing `registerProcessor`, tool rows stop, silently.
 */
import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
} from "vitest";

import { registerAgentsOn, maybeRegisterOpenAIAgentsTracing } from "../src/frameworkTools";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import {
  declaredDevDependency,
  installNoNetworkGuard,
  installedVersion,
  requireCjs,
} from "./helpers/realProviderSeam";

const VERSION = installedVersion("@openai/agents");
// Loaded at MODULE scope, not in beforeAll — see the note in
// realProviderSeams.aiSdk.test.ts. A hook-based gate made a load failure report
// as a PASSING test, which is precisely the silence this family exists to break.
const loaded = await (async () => {
  try {
    return { mod: await import("@openai/agents"), error: null as unknown };
  } catch (err) {
    return { mod: null, error: err };
  }
})();
const agentsEsm: any = loaded.mod;
const agentsCoreCjs: any = (() => {
  try {
    return requireCjs("@openai/agents-core");
  } catch {
    return undefined;
  }
})();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] @openai/agents@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

const maybe = agentsEsm ? it : it.skip;

// A declared devDependency that will not import is a HARD failure, never a
// skip — otherwise a broken package silently empties this whole file.
(declaredDevDependency("@openai/agents") ? it : it.skip)(
  "@openai/agents imports",
  () => {
    expect(
      loaded.error,
      `@openai/agents is a declared devDependency but the import threw: ` +
        `${loaded.error instanceof Error ? loaded.error.message : String(loaded.error)}`,
    ).toBeNull();
  },
);

describe("@openai/agents — shape", () => {
  maybe("addTraceProcessor is exported from the ESM build", () => {
    expect(
      typeof agentsEsm.addTraceProcessor,
      "addTraceProcessor is gone from @openai/agents — registerAgentsOn can " +
        "no longer attach, so tool spans stop producing rows",
    ).toBe("function");
  });

  maybe("addTraceProcessor is exported from @openai/agents-core too", () => {
    if (!agentsCoreCjs) return; // not installed standalone on this release
    expect(typeof agentsCoreCjs.addTraceProcessor).toBe("function");
  });
});

describe("@openai/agents — registerAgentsOn attaches to the real module", () => {
  maybe("returns true on first attach and dedupes on the second", () => {
    // registerAgentsOn dedupes by the TraceProvider INSTANCE the handle feeds
    // (falling back to `addTraceProcessor` identity when none is exposed), so
    // the CJS and ESM builds over one shared provider register once while a
    // genuinely separate provider still gets its own. Asserted against the
    // real export.
    const first = registerAgentsOn(agentsEsm);
    const second = registerAgentsOn(agentsEsm);
    expect(first || second, "registerAgentsOn never attached at all").toBe(true);
    expect(second, "the identity dedupe stopped working — double registration").toBe(false);
  });

  maybe("a rejected addTraceProcessor never throws into init()", () => {
    // GOLDEN RULE: a broken/hostile agents build must degrade to "no tool
    // rows", never to an exception in the customer's startup path.
    const hostile = {
      addTraceProcessor() {
        throw new Error("upstream refused the processor");
      },
    };
    expect(() => registerAgentsOn(hostile)).not.toThrow();
    expect(registerAgentsOn(hostile)).toBe(false);
  });
});

const AGENTS_PROVIDER_SYMBOL = Symbol.for("openai.agents.core.traceProvider");

describe("@openai/agents — ONE tool row per function span on the shared provider (G1-24-1)", () => {
  // agents-core >= 0.4.0 keeps its TraceProvider on globalThis, shared by the
  // CJS and ESM builds. Verification runs #24/#25 saw every tool call land as
  // TWO tool rows because a processor was registered per build. Drive a real
  // function span through the real package and count rows.
  maybe("ESM umbrella + CJS core handles collapse to a single processor", async () => {
    if (!agentsCoreCjs) return;
    registerAgentsOn(agentsEsm);
    registerAgentsOn(agentsCoreCjs);
    maybeRegisterOpenAIAgentsTracing();
    const calls: any[] = [];
    setClient({ log: (...a: any[]) => calls.push(a) } as any);
    const session = new TPSession({
      userId: "u",
      paidPlan: "free",
      workflowName: "wf",
      traceId: "a".repeat(32),
      rootSpanId: "b".repeat(16),
    });
    // agents-core disables tracing outright under NODE_ENV=test (vitest sets
    // it): every span would be a NoopSpan and no processor would ever fire.
    agentsEsm.setTracingDisabled(false);
    try {
      await _getSessionStorage().run(session, () =>
        agentsEsm.withTrace("seam", () =>
          agentsEsm.withFunctionSpan(async () => "ok", {
            data: { name: "getCustomerInfo", input: "{}" },
          }),
        ),
      );
    } finally {
      agentsEsm.setTracingDisabled(true);
    }
    const toolRows = calls.filter((a) => a[13]?.tool?.name === "getCustomerInfo");
    expect(
      toolRows.length,
      `expected ONE tool row per function span, got ${toolRows.length} — a processor was registered per build`,
    ).toBe(1);
  });

  maybe("the module fallback never loads the umbrella @openai/agents CJS build", () => {
    // The umbrella's module init calls setDefaultOpenAITracingExporter(), which
    // REPLACES every processor on the shared provider (the customer's own
    // included) — so the SDK must only ever touch `@openai/agents-core`, whose
    // init has no side effects. Force the fallback by hiding the global provider.
    const saved = (globalThis as any)[AGENTS_PROVIDER_SYMBOL];
    delete (globalThis as any)[AGENTS_PROVIDER_SYMBOL];
    try {
      maybeRegisterOpenAIAgentsTracing();
    } finally {
      if (saved === undefined) delete (globalThis as any)[AGENTS_PROVIDER_SYMBOL];
      else (globalThis as any)[AGENTS_PROVIDER_SYMBOL] = saved;
    }
    const umbrella = Object.keys(requireCjs.cache).filter((k) =>
      /[\\/]@openai[\\/]agents[\\/]dist[\\/]index\.js$/.test(k),
    );
    expect(
      umbrella,
      "the SDK required @openai/agents (CJS) — its init resets the shared provider's processors",
    ).toEqual([]);
    const core = Object.keys(requireCjs.cache).filter((k) =>
      /[\\/]@openai[\\/]agents-core[\\/]dist[\\/]index\.js$/.test(k),
    );
    expect(core.length, "the fallback did not run against @openai/agents-core at all").toBeGreaterThan(0);
  });
});
