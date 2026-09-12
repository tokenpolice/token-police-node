/**
 * REAL-PACKAGE seam contract — `@google/genai`.
 *
 * See tests/helpers/realProviderSeam.ts for why this family exists.
 *
 * `@google/genai` is the highest-risk Node provider for silent drift: there is
 * NO OpenLLMetry-JS instrumentor for it, so every token this SDK reports comes
 * from the enforcer's own manual-telemetry wrapper reading `usageMetadata`. If
 * the prototype method moves or the usage field is renamed, Google traffic goes
 * to zero rows with nothing else in the path to notice.
 *
 * The registry deliberately carries BOTH the `*Internal` delegate and the
 * public alias for several operations, because the public
 * `generateContent` / `embedContent` / `generateImages` / `generateVideos` are
 * instance ARROW FIELDS on some releases (un-patchable) and prototype methods
 * on others. Only the `*Internal` rows are load-bearing — the aliases are a
 * best-effort second chance and are allowed to be absent. That asymmetry is
 * encoded in OPTIONAL_ALIASES below rather than left implicit, so a missing
 * alias stays quiet while a missing delegate goes red.
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

import { autoInstrument, uninstrument } from "../src/enforcer";
import {
  createSeamHarness,
  flushLogs,
  installNoNetworkGuard,
  installedVersion,
  jsonResponder,
  methodNames,
  requireCjs,
  routedGlobalFetch,
  snapshotMethods,
  sseResponder,
  targetLabel,
  targetsFor,
  walkPath,
} from "./helpers/realProviderSeam";

const PKG = "@google/genai";
const VERSION = installedVersion(PKG);
const googleModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
const maybe = googleModule ? it : it.skip;

/**
 * Registry rows the enforcer treats as best-effort: each is the PUBLIC alias of
 * an `*Internal` delegate that is patched separately. `_wrapMethod` logs and
 * skips when the alias is absent, and the call still routes through the wrapped
 * delegate, so an absent alias is not a metering gap.
 *
 * Each entry names the delegate that MUST be present for that claim to hold —
 * asserted below, so this list can never quietly become an excuse for a real
 * break.
 */
const OPTIONAL_ALIASES: Record<string, string> = {
  generateImages: "generateImagesInternal",
  generateVideos: "generateVideosInternal",
  embedContent: "embedContentInternal",
};

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] @google/genai@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(50);
  uninstrument();
  h.reset();
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 1 — SHAPE
// ═══════════════════════════════════════════════════════════════════════
describe("@google/genai — shape: the enforcer's patch targets exist", () => {
  const targets = targetsFor(PKG);

  it("the registry still declares the generateContent delegates", () => {
    const names = targets.map((t) => t.method);
    expect(names).toContain("generateContentInternal");
    expect(names).toContain("generateContentStreamInternal");
  });

  maybe("Models is exported at the package root", () => {
    // The objectPath is ["Models", "prototype"] off the module root — NOT off a
    // client class — so a root-export rename breaks every google seam at once.
    expect(typeof googleModule.Models).toBe("function");
  });

  for (const t of targets) {
    const optionalDelegate = OPTIONAL_ALIASES[t.method];
    maybe(
      `${targetLabel(t)} ${optionalDelegate ? "(optional alias)" : "resolves and is a function"}`,
      () => {
        const { obj, brokeAt } = walkPath(googleModule, t.objectPath);
        expect(
          obj,
          `objectPath [${t.objectPath.join(", ")}] broke at "${brokeAt}"`,
        ).toBeTruthy();

        if (optionalDelegate) {
          // Absent alias is fine ONLY while its delegate is present — that is
          // the whole basis for treating it as optional.
          expect(
            typeof obj[optionalDelegate],
            `alias "${t.method}" is absent AND its delegate ` +
              `"${optionalDelegate}" is gone too — this operation now has NO ` +
              `instrumented entry point and reports zero tokens`,
          ).toBe("function");
          return;
        }
        expect(
          typeof obj[t.method],
          `method "${t.method}" is gone; object carries: ${methodNames(obj).join(", ")}`,
        ).toBe("function");
      },
    );
  }

  maybe("autoInstrument() replaces every non-alias google target", () => {
    const required = targets.filter((t) => !OPTIONAL_ALIASES[t.method]);
    const before = snapshotMethods(googleModule, required);
    autoInstrument();
    for (const t of required) {
      const { obj } = walkPath(googleModule, t.objectPath);
      expect(
        obj?.[t.method],
        `${targetLabel(t)} was NOT wrapped — google has no OpenLLMetry ` +
          `instrumentor, so this seam is the ONLY source of its tokens`,
      ).not.toBe(before.get(targetLabel(t)));
    }
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 2 — ROUND TRIP
//
// `@google/genai` offers no constructor-level transport injection, so the
// global fetch is routed instead. `routedGlobalFetch` THROWS on an unmatched
// URL, so a call that escapes the stub fails loudly here.
// ═══════════════════════════════════════════════════════════════════════
describe("@google/genai — round trip: models.generateContent (manual telemetry)", () => {
  const CONTENT = (usageMetadata: Record<string, unknown>) => ({
    candidates: [
      { content: { role: "model", parts: [{ text: "hi" }] }, finishReason: "STOP", index: 0 },
    ],
    modelVersion: "gemini-2.5-flash",
    usageMetadata,
  });

  maybe("non-stream reports usageMetadata with cached tokens split out", async () => {
    const net = routedGlobalFetch([
      { match: "generativelanguage.googleapis.com", respond: jsonResponder(CONTENT({
        promptTokenCount: 17,
        candidatesTokenCount: 6,
        totalTokenCount: 23,
        cachedContentTokenCount: 2,
      })) },
    ]);
    try {
      autoInstrument();
      const genai = new googleModule.GoogleGenAI({ apiKey: "g-test-not-a-real-key" });
      await genai.models.generateContent({ model: "gemini-2.5-flash", contents: "hello" });
      await flushLogs();

      const row = h.only((l) => l.provider === "google", "google chat");
      expect(row.model).toBe("gemini-2.5-flash");
      // 17 prompt tokens of which 2 were cached → 15 fresh + 2 cached.
      expect(row.inputTokens).toBe(15);
      expect(row.cachedTokens).toBe(2);
      expect(row.inputTokens + row.cachedTokens).toBe(17);
      expect(row.outputTokens).toBe(6);
      expect(row.extra?.usage?.shape).toBe("google_genai");
      expect(row.extra?.usage?.raw?.promptTokenCount).toBe(17);
      expect(net.urls.length).toBeGreaterThan(0);
    } finally {
      net.restore();
    }
  });

  maybe("stream reports usage from the final usageMetadata-bearing chunk", async () => {
    // The REST streaming endpoint answers with SSE frames of the same shape.
    const net = routedGlobalFetch([
      {
        match: "generativelanguage.googleapis.com",
        respond: sseResponder(
          [
            {
              candidates: [{ content: { role: "model", parts: [{ text: "hi" }] }, index: 0 }],
              modelVersion: "gemini-2.5-flash",
            },
            CONTENT({ promptTokenCount: 31, candidatesTokenCount: 4, totalTokenCount: 35 }),
          ],
          { done: false },
        ),
      },
    ]);
    try {
      autoInstrument();
      const genai = new googleModule.GoogleGenAI({ apiKey: "g-test-not-a-real-key" });
      const stream: any = await genai.models.generateContentStream({
        model: "gemini-2.5-flash",
        contents: "hello",
      });
      for await (const _ of stream) {
        /* drain */
      }
      await flushLogs(60);

      const row = h.only((l) => l.provider === "google", "google chat stream");
      expect(row.inputTokens).toBe(31);
      expect(row.outputTokens).toBe(4);
      expect(row.extra?.latency?.is_streaming).toBe(true);
    } finally {
      net.restore();
    }
  });

  maybe("embedContent logs operation=embedding with the google embedding shape", async () => {
    const net = routedGlobalFetch([
      {
        match: "generativelanguage.googleapis.com",
        respond: jsonResponder({
          embeddings: [{ values: [0.1, 0.2, 0.3] }],
          metadata: { billableCharacterCount: 5 },
        }),
      },
    ]);
    try {
      autoInstrument();
      const genai = new googleModule.GoogleGenAI({ apiKey: "g-test-not-a-real-key" });
      await genai.models.embedContent({
        model: "gemini-embedding-001",
        contents: "hello",
      });
      await flushLogs();

      const row = h.only((l) => l.extra?.operation === "embedding", "google embedding");
      expect(row.provider).toBe("google");
      expect(row.extra?.usage?.shape).toBe("google_genai_embeddings");
      // Google's embed response carries no token count; the enforcer
      // approximates from the request. The contract asserted here is that a row
      // IS produced with a positive count — a silent zero would mean unpriced
      // RAG ingest traffic.
      expect(row.inputTokens).toBeGreaterThan(0);
    } finally {
      net.restore();
    }
  });
});
