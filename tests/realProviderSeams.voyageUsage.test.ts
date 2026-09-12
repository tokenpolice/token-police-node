/**
 * REAL-PACKAGE seam contract — Voyage AI token usage.
 *
 * Split out of `realProviderSeams.voyage.test.ts` so a usage-shape regression
 * fails on its own, separately from the shape and attach alarms in the
 * sibling file.
 *
 * THE BREAK IT PINS (fixed 2026-09)
 * ---------------------------------
 * `_extractEmbeddingUsage`'s `voyage` branch used to read only
 * `result.usage.total_tokens` (snake_case). The published `voyageai` client is
 * Fern-generated and camelCases the parsed response, so the object the
 * enforcer sees is `{ usage: { totalTokens: 19 } }`. `Number(undefined) || 0`
 * → **every Voyage embedding call was logged with 0 input tokens**: the row
 * existed, so nothing looked broken, but the spend was silently un-billed.
 * The extractor now reads `total_tokens ?? totalTokens`.
 *
 * Voyage is the Anthropic-recommended embedding provider — the enforcer's own
 * comment notes "every Claude RAG customer is a Voyage customer" — so this is a
 * revenue-visible under-count, not a cosmetic one.
 *
 * Why nothing caught it: no Node test loaded the real package (the cell was
 * `fake-only`), and the hand-built fake was written in snake_case to match the
 * extractor.
 *
 * DO NOT change these numbers to observed zeros to make the run green — a zero
 * here means the client's usage shape moved again and billing is broken.
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
  requireCjs,
  routedGlobalFetch,
} from "./helpers/realProviderSeam";

const PKG = "voyageai";
const VERSION = installedVersion(PKG);
const voyageModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
const maybe = voyageModule ? it : it.skip;
const VoyageAIClient: any =
  voyageModule?.VoyageAIClient ?? voyageModule?.default?.VoyageAIClient ?? voyageModule?.default;

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] voyageai@${VERSION ?? "ABSENT"} (usage)`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(30);
  uninstrument();
  h.reset();
});

describe("voyageai — round trip: embed usage", () => {
  maybe("the wire's token count reaches the /log payload", async () => {
    const net = routedGlobalFetch([
      {
        match: "api.voyageai.com",
        respond: jsonResponder({
          object: "list",
          model: "voyage-3",
          data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
          usage: { total_tokens: 19 },
        }),
      },
    ]);
    try {
      (__test__ as any)._instrumentVoyage(voyageModule);
      (__test__ as any)._setInstrumented(true);
      const client = new VoyageAIClient({ apiKey: "vk-test-not-a-real-key" });
      await client.embed({ model: "voyage-3", input: ["hello"] });
      await flushLogs();

      const row = h.only((l) => l.extra?.operation === "embedding", "voyage embedding");
      expect(row.provider).toBe("voyage");
      expect(row.model).toBe("voyage-3");
      expect(row.extra?.usage?.shape).toBe("voyage_embed");
      expect(
        row.inputTokens,
        "Voyage embedding logged 0 tokens: the wire says total_tokens=19 but " +
          "the Fern client hands the enforcer `usage.totalTokens`, and the " +
          "extractor only reads `usage.total_tokens`. Silent under-billing.",
      ).toBe(19);
    } finally {
      net.restore();
    }
  });
});
