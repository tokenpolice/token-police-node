/**
 * REAL-PACKAGE seam contract — `cohere-ai` (Cohere v2).
 *
 * THIN BY DESIGN: this is one of the long-tail providers. It covers the seam's
 * structural preconditions and one non-stream round trip, not the full
 * operation matrix that openai / anthropic / google / ai-sdk get. Deepen it if
 * Cohere becomes load-bearing for a customer.
 *
 * The seam is unusually fragile and cannot be expressed in `_TARGET_METHODS` at
 * all: `CohereClientV2.chat` / `.chatStream` are instance-bound arrow fields set
 * in the constructor, so `_instrumentCohere` constructs a throw-away client to
 * reach the internal `V2Client` class — which the package does NOT export — and
 * patches ITS prototype. Three separate things can move without any of our
 * fakes noticing: the `CohereClientV2` export, the private `clientV2` property
 * name, and the `V2Client` prototype methods.
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

const PKG = "cohere-ai";
const VERSION = installedVersion(PKG);
const cohereModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
const maybe = cohereModule ? it : it.skip;
const CohereClientV2: any =
  cohereModule?.CohereClientV2 ?? cohereModule?.default?.CohereClientV2;

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] cohere-ai@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(30);
  uninstrument();
  h.reset();
});

/** Reproduces `_instrumentCohere`'s probe walk so the test reports WHERE it broke. */
function probeV2Client(): { probe: any; V2ClientClass: any } {
  const probe = new CohereClientV2({ token: "tp-probe-not-a-real-key" });
  const v2client = probe?.clientV2;
  return { probe, V2ClientClass: v2client && Object.getPrototypeOf(v2client)?.constructor };
}

describe("cohere-ai — shape: the probe walk _instrumentCohere depends on", () => {
  maybe("CohereClientV2 is exported at the package root", () => {
    expect(typeof CohereClientV2).toBe("function");
  });

  maybe("constructing a client is side-effect free and exposes `clientV2`", () => {
    // The Fern constructor only stores options — no network. If the internal
    // property is renamed, `_instrumentCohere` returns early and EVERY Cohere
    // v2 call runs unwrapped (no /check, and no telemetry: the Traceloop
    // cohere instrumentor only covers the legacy v1 client).
    const { probe, V2ClientClass } = probeV2Client();
    expect(probe?.clientV2, "CohereClientV2 no longer exposes `clientV2`").toBeTruthy();
    expect(typeof V2ClientClass).toBe("function");
  });

  maybe("V2Client.prototype still carries chat and chatStream", () => {
    const { V2ClientClass } = probeV2Client();
    for (const m of ["chat", "chatStream"]) {
      expect(
        typeof V2ClientClass.prototype[m],
        `V2Client.prototype.${m} is gone; prototype carries: ` +
          methodNames(V2ClientClass.prototype).join(", "),
      ).toBe("function");
    }
    // `embed` / `embedAsync` are patched opportunistically (the enforcer skips
    // whichever is absent), so at least one must exist for embedding coverage.
    expect(
      typeof V2ClientClass.prototype.embed === "function" ||
        typeof V2ClientClass.prototype.embedAsync === "function",
    ).toBe(true);
  });

  maybe("_instrumentCohere replaces the prototype methods it targets", () => {
    const { V2ClientClass } = probeV2Client();
    const before = {
      chat: V2ClientClass.prototype.chat,
      chatStream: V2ClientClass.prototype.chatStream,
    };
    (__test__ as any)._instrumentCohere(cohereModule);
    (__test__ as any)._setInstrumented(true);
    expect(V2ClientClass.prototype.chat, "V2Client.prototype.chat was NOT wrapped").not.toBe(
      before.chat,
    );
    expect(V2ClientClass.prototype.chatStream).not.toBe(before.chatStream);
  });
});

describe("cohere-ai — round trip: chat (manual telemetry)", () => {
  maybe("reports billed_units input/output tokens under the cohere shape", async () => {
    const net = routedGlobalFetch([
      {
        match: "api.cohere.com",
        respond: jsonResponder({
          id: "c-1",
          message: { role: "assistant", content: [{ type: "text", text: "hi" }] },
          finish_reason: "COMPLETE",
          usage: {
            billed_units: { input_tokens: 12, output_tokens: 4 },
            tokens: { input_tokens: 12, output_tokens: 4 },
          },
        }),
      },
    ]);
    try {
      (__test__ as any)._instrumentCohere(cohereModule);
      (__test__ as any)._setInstrumented(true);
      const client = new CohereClientV2({ token: "tp-probe-not-a-real-key" });
      await client.chat({
        model: "command-r-plus",
        messages: [{ role: "user", content: "hello" }],
      });
      await flushLogs();

      const row = h.only((l) => l.provider === "cohere", "cohere chat");
      expect(row.model).toBe("command-r-plus");
      expect(row.inputTokens).toBe(12);
      expect(row.outputTokens).toBe(4);
      expect(row.extra?.usage?.shape).toBe("cohere_chat");
    } finally {
      net.restore();
    }
  });
});
