/**
 * REAL-PACKAGE seam contract — `@openrouter/sdk`.
 *
 * THIN BY DESIGN (long-tail provider): the probe walk plus one non-stream round
 * trip. The package is ESM-only and does NOT export its `Chat` class, so
 * `_instrumentOpenRouter` constructs a throw-away `OpenRouter` to reach
 * `Chat.prototype.send`. There is no OpenLLMetry instrumentor for it, so a
 * broken probe means no rows AND no enforcement.
 *
 * Note the request envelope: OpenRouter's Speakeasy client takes
 * `{ chatRequest: { model, messages } }`, and the enforcer reads the model out
 * of that envelope (`reqBody.chatRequest.model`) as a matching-only hint so
 * model/provider rules still match. A flattened request shape would silently
 * bucket every OpenRouter call under model "unknown".
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
  declaredDevDependency,
  flushLogs,
  installNoNetworkGuard,
  installedVersion,
  jsonResponder,
  methodNames,
  routedGlobalFetch,
} from "./helpers/realProviderSeam";

const PKG = "@openrouter/sdk";
const VERSION = installedVersion(PKG);
// Loaded at MODULE scope, not in beforeAll: the package is ESM-only so the
// import must be dynamic, but a hook-based gate forced `if (!available) return`,
// which reports a load failure as a PASSING test. Top-level await lets the gate
// be a real `it.skip`, and the declared-devDependency check below turns a
// genuine load failure into a hard failure.
const loaded = await (async () => {
  try {
    return { mod: await import(PKG), error: null as unknown };
  } catch (err) {
    return { mod: null, error: err };
  }
})();
const orModule: any = loaded.mod;

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] @openrouter/sdk@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(30);
  uninstrument();
  h.reset();
});

const maybe = orModule ? it : it.skip;

// A declared devDependency that will not import is a HARD failure, never a
// skip — otherwise a broken package silently empties this whole file.
(declaredDevDependency(PKG) ? it : it.skip)(`${PKG} imports`, () => {
  expect(
    loaded.error,
    `${PKG} is a declared devDependency but the import threw: ` +
      `${loaded.error instanceof Error ? loaded.error.message : String(loaded.error)}`,
  ).toBeNull();
});

function probeChatClass(): any {
  const OpenRouterClass = orModule?.OpenRouter ?? orModule?.default ?? orModule;
  const probe = new OpenRouterClass({ apiKey: "tp-probe-not-a-real-key" });
  return probe?.chat && Object.getPrototypeOf(probe.chat)?.constructor;
}

describe("@openrouter/sdk — shape: the probe walk _instrumentOpenRouter depends on", () => {
  maybe("OpenRouter is exported and its lazy `chat` getter yields a Chat instance", () => {
    const OpenRouterClass = orModule?.OpenRouter ?? orModule?.default ?? orModule;
    expect(typeof OpenRouterClass).toBe("function");
    const ChatClass = probeChatClass();
    expect(ChatClass, "OpenRouter no longer exposes a `chat` sub-client").toBeTruthy();
    expect(
      typeof ChatClass.prototype.send,
      `Chat.prototype.send is gone; prototype carries: ` +
        methodNames(ChatClass.prototype).join(", "),
    ).toBe("function");
  });

  maybe("_instrumentOpenRouter replaces Chat.prototype.send", () => {
    const ChatClass = probeChatClass();
    const before = ChatClass.prototype.send;
    (__test__ as any)._instrumentOpenRouter(orModule);
    (__test__ as any)._setInstrumented(true);
    expect(ChatClass.prototype.send, "Chat.prototype.send was NOT wrapped").not.toBe(before);
  });
});

describe("@openrouter/sdk — round trip: chat.send (manual telemetry)", () => {
  maybe("reports the wire tokens and recovers the model from the chatRequest envelope", async () => {
    const net = routedGlobalFetch([
      {
        match: "openrouter.ai",
        respond: jsonResponder({
          id: "or-1",
          object: "chat.completion",
          created: 1,
          model: "openai/gpt-4o-mini",
          system_fingerprint: null,
          choices: [
            {
              index: 0,
              message: { role: "assistant", content: "hi" },
              finish_reason: "stop",
            },
          ],
          usage: { prompt_tokens: 33, completion_tokens: 7, total_tokens: 40 },
        }),
      },
    ]);
    try {
      (__test__ as any)._instrumentOpenRouter(orModule);
      (__test__ as any)._setInstrumented(true);
      const OpenRouterClass = orModule?.OpenRouter ?? orModule?.default ?? orModule;
      const client: any = new OpenRouterClass({ apiKey: "tp-probe-not-a-real-key" });
      await client.chat.send({
        chatRequest: {
          model: "openai/gpt-4o-mini",
          messages: [{ role: "user", content: "hello" }],
        },
      });
      await flushLogs();

      const row = h.only((l) => l.provider === "openrouter", "openrouter chat");
      expect(row.model).toBe("openai/gpt-4o-mini");
      expect(row.inputTokens).toBe(33);
      expect(row.outputTokens).toBe(7);
    } finally {
      net.restore();
    }
  });
});
