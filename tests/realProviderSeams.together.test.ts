/**
 * REAL-PACKAGE seam contract — `together-ai`.
 *
 * See tests/helpers/realProviderSeam.ts for why this family exists. Together is
 * the case that proves the point: it is a manual-telemetry provider (no
 * OpenLLMetry-JS instrumentor exists), and the registry's image row targets
 * `Together.Images.prototype.create`.
 *
 * THE IMAGE PAIR. together-ai renamed `Images.prototype.create` to `generate`
 * around 0.30 (0.50.0 has only `generate`). PR #601 registered BOTH names as
 * sibling rows: exactly one resolves on any given install, and `_wrapMethod`
 * skips the other via its method-not-found early return. So the image seam
 * cannot be asserted the way every other row is ("this path resolves") — the
 * contract is a PAIR contract, and both of its failure modes matter:
 *   - NEITHER resolves → images run unwrapped: no pre-flight /check and no row
 *     at all (together-ai has no OpenLLMetry-JS instrumentor). That is the bug
 *     #601 fixed, and the one this file's first run found.
 *   - BOTH resolve     → both get wrapped, so one image call produces two
 *     /check calls and two rows. A transitional release keeping `create` as a
 *     deprecated alias would do exactly this, and it would double-bill.
 * The assertions below pin the pair, the exactly-one resolution, and the
 * exactly-one wrap.
 *
 * `tests/togetherImagesGenerate.test.ts` (from #601) is the complement: it
 * drives hand-built fakes of BOTH vendor shapes, including the `create`-only
 * <0.30 shape that no installed package can provide any more. Neither file
 * replaces the other, and the drift cell runs both.
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
  snapshotMethods,
  sseResponder,
  targetLabel,
  targetsFor,
  walkPath,
} from "./helpers/realProviderSeam";

const PKG = "together-ai";
const VERSION = installedVersion(PKG);
const togetherModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
const maybe = togetherModule ? it : it.skip;
const Together: any = togetherModule?.Together ?? togetherModule?.default;

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] together-ai@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(50);
  uninstrument();
  h.reset();
});

function makeClient(fetchStub: (...a: any[]) => Promise<Response>): any {
  return new Together({ apiKey: "tk-test-not-a-real-key", fetch: fetchStub });
}

const MODEL = "meta-llama/Llama-3.3-70B-Instruct-Turbo";

// ═══════════════════════════════════════════════════════════════════════
// Layer 1 — SHAPE
// ═══════════════════════════════════════════════════════════════════════
describe("together-ai — shape: the enforcer's patch targets exist", () => {
  const allTargets = targetsFor(PKG);
  const imageTargets = allTargets.filter((t) => t.modality === "image_gen");
  const plainTargets = allTargets.filter((t) => t.modality !== "image_gen");

  it("the registry declares the chat and embedding rows", () => {
    expect(plainTargets.map(targetLabel).sort()).toEqual([
      "together-ai Together.Chat.Completions.prototype.create",
      "together-ai Together.Embeddings.prototype.create",
    ]);
  });

  it("the image rows are exactly the {create, generate} pair", () => {
    // Deliberately not a hardcoded total over all four rows: this states the
    // PAIR invariant #601 introduced. A third image row, or the loss of either
    // name, changes what "exactly one resolves / exactly one wraps" below can
    // even mean, so it should fail HERE rather than confuse those two.
    expect(imageTargets.map((t) => t.method).sort()).toEqual(["create", "generate"]);
    for (const t of imageTargets) {
      expect(t.objectPath).toEqual(["Together", "Images", "prototype"]);
      expect(t.shape).toBe("together_image");
    }
  });

  maybe("the package root exposes the Together class _pickClassExport resolves", () => {
    expect(typeof Together).toBe("function");
  });

  for (const t of plainTargets) {
    maybe(`${targetLabel(t)} resolves and is a function`, () => {
      const { obj, brokeAt } = walkPath(togetherModule, t.objectPath);
      expect(
        obj,
        `objectPath [${t.objectPath.join(", ")}] broke at "${brokeAt}"`,
      ).toBeTruthy();
      expect(
        typeof obj[t.method],
        `method "${t.method}" is gone from ${t.objectPath.join(".")}; the ` +
          `object still carries: ${methodNames(obj).join(", ")}. This target ` +
          `runs UNWRAPPED — no pre-flight /check and, because together-ai has ` +
          `no OpenLLMetry instrumentor, no telemetry at all.`,
      ).toBe("function");
    });
  }

  maybe("EXACTLY ONE of the image pair resolves on the installed client", () => {
    const proto = walkPath(togetherModule, ["Together", "Images", "prototype"]).obj;
    expect(proto, "Together.Images.prototype no longer resolves at all").toBeTruthy();
    const present = imageTargets.filter((t) => typeof proto[t.method] === "function");
    expect(
      present.map((t) => t.method),
      `Images.prototype carries [${methodNames(proto).join(", ")}]. NONE of the ` +
        `registered names present means images run unwrapped (no /check, no ` +
        `row); BOTH present means one call is wrapped twice — two /check calls ` +
        `and two together_image rows for one image.`,
    ).toHaveLength(1);
  });

  maybe("autoInstrument() replaces every plain target and EXACTLY ONE image method", () => {
    const before = snapshotMethods(togetherModule, allTargets);
    autoInstrument();
    for (const t of plainTargets) {
      const { obj } = walkPath(togetherModule, t.objectPath);
      expect(obj?.[t.method], `${targetLabel(t)} was NOT wrapped`).not.toBe(
        before.get(targetLabel(t)),
      );
    }
    const proto = walkPath(togetherModule, ["Together", "Images", "prototype"]).obj;
    const wrapped = imageTargets.filter((t) => {
      const now = proto?.[t.method];
      return typeof now === "function" && now !== before.get(targetLabel(t));
    });
    expect(
      wrapped.map((t) => t.method),
      "the image seam must end up wrapped on exactly one method — zero means " +
        "un-metered image spend, two means every image is billed twice",
    ).toHaveLength(1);
  });
});

// ═══════════════════════════════════════════════════════════════════════
// Layer 2 — ROUND TRIP
// ═══════════════════════════════════════════════════════════════════════
describe("together-ai — round trip: chat completions (manual telemetry)", () => {
  maybe("non-stream reports the OpenAI-shaped usage under the together shape", async () => {
    autoInstrument();
    await makeClient(
      jsonResponder({
        id: "tg-1",
        object: "chat.completion",
        created: 1,
        model: MODEL,
        choices: [
          { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
        ],
        usage: { prompt_tokens: 55, completion_tokens: 8, total_tokens: 63 },
      }),
    ).chat.completions.create({ model: MODEL, messages: [{ role: "user", content: "hello" }] });
    await flushLogs();

    const row = h.only((l) => l.provider === "together", "together chat");
    expect(row.model).toBe(MODEL);
    expect(row.inputTokens).toBe(55);
    expect(row.outputTokens).toBe(8);
    expect(row.extra?.usage?.shape).toBe("together_chat");
  });

  maybe("stream reports usage from the final usage-bearing chunk", async () => {
    autoInstrument();
    const chunk = (o: Record<string, unknown>) => ({
      id: "tg-1",
      object: "chat.completion.chunk",
      created: 1,
      model: MODEL,
      ...o,
    });
    const stream: any = await makeClient(
      sseResponder([
        chunk({ choices: [{ index: 0, delta: { role: "assistant", content: "hi" } }] }),
        chunk({
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 61, completion_tokens: 4, total_tokens: 65 },
        }),
      ]),
    ).chat.completions.create({
      model: MODEL,
      messages: [{ role: "user", content: "hello" }],
      stream: true,
    });
    for await (const _ of stream) {
      /* drain */
    }
    await flushLogs(60);

    const row = h.only((l) => l.provider === "together", "together chat stream");
    expect(row.inputTokens).toBe(61);
    expect(row.outputTokens).toBe(4);
    expect(row.extra?.latency?.is_streaming).toBe(true);
  });

  maybe("embeddings.create logs operation=embedding", async () => {
    autoInstrument();
    await makeClient(
      jsonResponder({
        object: "list",
        model: "togethercomputer/m2-bert-80M-8k-retrieval",
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2] }],
        usage: { prompt_tokens: 14, total_tokens: 14 },
      }),
    ).embeddings.create({
      model: "togethercomputer/m2-bert-80M-8k-retrieval",
      input: "hello",
    });
    await flushLogs();

    const row = h.only((l) => l.extra?.operation === "embedding", "together embedding");
    expect(row.provider).toBe("together");
    expect(row.inputTokens).toBe(14);
    expect(row.extra?.usage?.shape).toBe("together_embed");
  });
});

describe("together-ai — round trip: images (whichever half of the pair exists)", () => {
  maybe("an image call produces exactly one together_image row", async () => {
    autoInstrument();
    const client = makeClient(
      jsonResponder({
        id: "img-1",
        model: "black-forest-labs/FLUX.1-schnell",
        object: "list",
        data: [{ index: 0, type: "b64_json", b64_json: "aGk=" }],
      }),
    );

    // Call whichever surface this release exposes — the assertion is about the
    // ROW, not the method name, so it keeps its meaning across the rename.
    const images: any = client.images;
    const call = typeof images.generate === "function" ? images.generate : images.create;
    await call.call(images, {
      model: "black-forest-labs/FLUX.1-schnell",
      prompt: "a cat",
      width: 1024,
      height: 1024,
      n: 1,
    });
    await flushLogs();

    const rows = h.logs.filter((l) => l.extra?.usage?.shape === "together_image");
    expect(
      rows.length,
      `expected exactly one together_image row; got ${rows.length}. ` +
        `Images.prototype carries [${methodNames(Together.Images.prototype).join(", ")}]. ` +
        `Zero means image spend is un-metered and un-gated; two means the ` +
        `create/generate pair both resolved and every image is billed twice.`,
    ).toBe(1);
    expect(rows[0].extra?.operation).toBe("image_gen");
    expect(rows[0].extra?.usage?.items?.images_generated).toBe(1);
    expect(h.checks.length, "the image call ran with no pre-flight /check").toBeGreaterThan(0);
  });
});
