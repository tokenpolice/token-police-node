/**
 * REAL-PACKAGE seam contract — `@llamaindex/openai`.
 *
 * THIN BY DESIGN (long-tail framework): the class/prototype seam plus one
 * non-stream round trip through the real LlamaIndex-JS OpenAI LLM. The other
 * provider subpackages (`@llamaindex/anthropic`, `@llamaindex/google`) go
 * through the identical `_instrumentLlamaIndexProvider` code path with a
 * different class-name list, so one representative package is enough to catch
 * a structural break; add the others if a customer makes them load-bearing.
 *
 * `_instrumentLlamaIndexProvider(mod, ["OpenAI", "OpenAIResponses"])` patches
 * `.chat` / `.complete` on each named class's prototype and stamps
 * `__tp_li_wrapped` for idempotency. `OpenAIResponses` is a SIBLING class, not
 * a subclass — patching `OpenAI` does not reach it — so both names must keep
 * resolving from the package root.
 */
import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
  beforeEach,
  afterEach,
} from "vitest";

import { __test__, uninstrument } from "../src/enforcer";
import {
  createSeamHarness,
  flushLogs,
  installNoNetworkGuard,
  installedVersion,
  jsonResponder,
  methodNames,
  requireCjs,
  routedGlobalFetch,
} from "./helpers/realProviderSeam";

const PKG = "@llamaindex/openai";
const VERSION = installedVersion(PKG);
const liModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
const maybe = liModule ? it : it.skip;

/** The exact class-name list autoInstrument() passes for this package. */
const CLASS_NAMES = ["OpenAI", "OpenAIResponses"];

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] @llamaindex/openai@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(30);
  uninstrument();
  h.reset();
});

describe("@llamaindex/openai — shape", () => {
  maybe("both LLM classes are exported and carry chat + complete", () => {
    for (const name of CLASS_NAMES) {
      const Cls = liModule?.[name] ?? liModule?.default?.[name];
      expect(typeof Cls, `${name} is no longer exported from ${PKG}`).toBe("function");
      for (const m of ["chat", "complete"]) {
        expect(
          typeof Cls.prototype[m],
          `${name}.prototype.${m} is gone; prototype carries: ` +
            methodNames(Cls.prototype).join(", "),
        ).toBe("function");
      }
    }
  });

  maybe("OpenAIResponses is a SIBLING of OpenAI, not a subclass", () => {
    // The registry lists both names precisely because patching one does not
    // reach the other. If that ever changed, both patches would land on one
    // prototype and every Responses call would be wrapped twice.
    const A = liModule.OpenAI;
    const B = liModule.OpenAIResponses;
    expect(A.prototype).not.toBe(B.prototype);
  });

  maybe("_instrumentLlamaIndexProvider wraps and marks each prototype method", () => {
    const before = new Map<string, unknown>();
    for (const name of CLASS_NAMES) {
      for (const m of ["chat", "complete"]) {
        before.set(`${name}.${m}`, liModule[name].prototype[m]);
      }
    }
    (__test__ as any)._instrumentLlamaIndexProvider(liModule, CLASS_NAMES);
    (__test__ as any)._setInstrumented(true);
    for (const name of CLASS_NAMES) {
      for (const m of ["chat", "complete"]) {
        const now = liModule[name].prototype[m];
        expect(now, `${name}.prototype.${m} was NOT wrapped`).not.toBe(
          before.get(`${name}.${m}`),
        );
        expect((now as any).__tp_li_wrapped).toBe(true);
      }
    }
  });
});

describe("@llamaindex/openai — round trip: llm.chat (framework wrapper)", () => {
  maybe("reports the underlying provider's tokens through the framework seam", async () => {
    const net = routedGlobalFetch([
      {
        match: "api.openai.com",
        respond: jsonResponder({
          id: "chatcmpl-li",
          object: "chat.completion",
          created: 1,
          model: "gpt-4o-mini",
          choices: [
            { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 27, completion_tokens: 5, total_tokens: 32 },
        }),
      },
    ]);
    try {
      (__test__ as any)._instrumentLlamaIndexProvider(liModule, CLASS_NAMES);
      (__test__ as any)._setInstrumented(true);
      const llm = new liModule.OpenAI({
        apiKey: "sk-test-not-a-real-key",
        model: "gpt-4o-mini",
      });
      await llm.chat({ messages: [{ role: "user", content: "hello" }] });
      await flushLogs(50);

      // Exactly one row: the framework wrapper claims the call and suppresses
      // the inner provider wrapper, so a second row here would be a double-bill.
      const row = h.only((l) => l.inputTokens === 27, "llamaindex chat");
      expect(row.model).toBe("gpt-4o-mini");
      expect(row.outputTokens).toBe(5);
    } finally {
      net.restore();
    }
  });
});
