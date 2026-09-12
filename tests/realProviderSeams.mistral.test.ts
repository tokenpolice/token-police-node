/**
 * REAL-PACKAGE seam contract — `@mistralai/mistralai`.
 *
 * THIN BY DESIGN (long-tail provider): the probe walk, the four sub-client
 * prototypes `_instrumentMistral` reaches, and one non-stream round trip.
 *
 * `@mistralai/mistralai` was already a devDependency before this file, but
 * nothing loaded it: `instrumentModulesKeyContract.test.ts` builds a synthetic
 * Mistral fixture instead, so the cell was marked `fake-only` in
 * provider-drift-suite/catalog.json. The Speakeasy client exposes `chat`,
 * `embeddings`, `ocr` and `audio.transcriptions` as instance properties whose
 * classes are NOT root-exported, so — as with Cohere — the enforcer probes a
 * throw-away client to reach each prototype. Any of those property names moving
 * silently drops that operation.
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

const PKG = "@mistralai/mistralai";
const VERSION = installedVersion(PKG);
const mistralModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
const maybe = mistralModule ? it : it.skip;
const Mistral: any = mistralModule?.Mistral ?? mistralModule?.default;

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] @mistralai/mistralai@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(30);
  uninstrument();
  h.reset();
});

function probe(): any {
  return new Mistral({ apiKey: "tp-probe-not-a-real-key" });
}
function protoOf(instance: any): any {
  return instance && Object.getPrototypeOf(instance)?.constructor?.prototype;
}

describe("@mistralai/mistralai — shape: the probe walk _instrumentMistral depends on", () => {
  maybe("Mistral is exported at the package root", () => {
    expect(typeof Mistral).toBe("function");
  });

  maybe("chat.complete and chat.stream are on the Chat prototype", () => {
    // These are the ONLY two rows without a fallback: no OpenLLMetry-JS
    // instrumentor covers Mistral, so losing them means zero rows.
    const chatProto = protoOf(probe().chat);
    expect(chatProto, "Mistral no longer exposes a `chat` sub-client").toBeTruthy();
    for (const m of ["complete", "stream"]) {
      expect(
        typeof chatProto[m],
        `Chat.prototype.${m} is gone; prototype carries: ` +
          methodNames(chatProto).join(", "),
      ).toBe("function");
    }
  });

  maybe("the optional sub-clients keep at least one instrumented method each", () => {
    // Embeddings / OCR / transcriptions are patched opportunistically — the
    // enforcer skips whichever method is absent. What must NOT happen is a
    // sub-client losing ALL of them, which is a silent coverage loss.
    const p = probe();
    const cases: Array<[string, any, string[]]> = [
      ["embeddings", protoOf(p.embeddings), ["create"]],
      ["ocr", protoOf(p.ocr), ["process", "processAsync"]],
      ["audio.transcriptions", protoOf(p.audio?.transcriptions), ["complete", "completeAsync"]],
    ];
    for (const [name, proto, methods] of cases) {
      if (!proto) continue; // sub-client absent on this release — nothing to lose
      expect(
        methods.some((m) => typeof proto[m] === "function"),
        `${name} exposes none of [${methods.join(", ")}]; prototype carries: ` +
          methodNames(proto).join(", "),
      ).toBe(true);
    }
  });

  maybe("_instrumentMistral replaces Chat.prototype.complete and .stream", () => {
    const chatProto = protoOf(probe().chat);
    const before = { complete: chatProto.complete, stream: chatProto.stream };
    (__test__ as any)._instrumentMistral(mistralModule);
    (__test__ as any)._setInstrumented(true);
    expect(chatProto.complete, "Chat.prototype.complete was NOT wrapped").not.toBe(
      before.complete,
    );
    expect(chatProto.stream).not.toBe(before.stream);
  });
});

describe("@mistralai/mistralai — round trip: chat.complete (manual telemetry)", () => {
  maybe("reports the wire's prompt/completion tokens under the mistral shape", async () => {
    const net = routedGlobalFetch([
      {
        match: "api.mistral.ai",
        respond: jsonResponder({
          id: "m-1",
          object: "chat.completion",
          created: 1,
          model: "mistral-large-latest",
          choices: [
            { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 23, completion_tokens: 6, total_tokens: 29 },
        }),
      },
    ]);
    try {
      (__test__ as any)._instrumentMistral(mistralModule);
      (__test__ as any)._setInstrumented(true);
      const client = new Mistral({ apiKey: "tp-probe-not-a-real-key" });
      await client.chat.complete({
        model: "mistral-large-latest",
        messages: [{ role: "user", content: "hello" }],
      });
      await flushLogs();

      const row = h.only((l) => l.provider === "mistral", "mistral chat");
      expect(row.model).toBe("mistral-large-latest");
      expect(row.inputTokens).toBe(23);
      expect(row.outputTokens).toBe(6);
      expect(row.extra?.usage?.shape).toBe("mistral_chat");
    } finally {
      net.restore();
    }
  });
});
