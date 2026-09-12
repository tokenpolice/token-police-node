/**
 * REAL-PACKAGE seam contract — `@huggingface/inference`.
 *
 * THIN BY DESIGN (long-tail provider): the require-cache seam plus one
 * non-stream chat round trip. `modality.test.ts` and `preflightCtxF7.test.ts`
 * already drive the real package for the image/TTS/ASR handlers, so this file
 * deliberately does not repeat them.
 *
 * The seam is the most implementation-coupled of the whole registry:
 * `InferenceClient`'s task methods are per-instance, non-writable fields copied
 * BY VALUE from a `tasks` index module at construction time, so there is no
 * prototype and no delegated class to patch. `_instrumentHuggingFace` instead
 * rewrites the exports of the SOURCE submodules in Node's require cache —
 * `tasks/nlp/chatCompletion.js` and friends — which every later client picks
 * up. A directory reshuffle upstream breaks that with no error anywhere.
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

const PKG = "@huggingface/inference";
const VERSION = installedVersion(PKG);
const hfModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
const maybe = hfModule ? it : it.skip;

/** The require-cache paths `_instrumentHuggingFace` looks for, verbatim. */
const TASK_SUFFIXES = [
  "/tasks/nlp/chatCompletion.js",
  "/tasks/nlp/chatCompletionStream.js",
  "/tasks/nlp/featureExtraction.js",
  "/tasks/cv/textToImage.js",
  "/tasks/audio/textToSpeech.js",
  "/tasks/audio/automaticSpeechRecognition.js",
];

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] @huggingface/inference@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(30);
  uninstrument();
  h.reset();
});

describe("@huggingface/inference — shape: the require-cache task submodules", () => {
  maybe("InferenceClient is exported at the package root", () => {
    expect(typeof hfModule.InferenceClient).toBe("function");
  });

  maybe("every task submodule the enforcer patches is in the require cache", () => {
    const cache = (__test__ as any)._requireCache();
    const keys = Object.keys(cache).filter((k) => k.includes(PKG));
    expect(keys.length, "the package is not in the CJS require cache at all").toBeGreaterThan(0);

    const missing = TASK_SUFFIXES.filter(
      (suffix) => !keys.some((k) => k.replace(/\\/g, "/").endsWith(suffix)),
    );
    expect(
      missing,
      `these task submodules moved upstream — the require-cache patch silently ` +
        `no-ops for each, so those operations run unwrapped: ${missing.join(", ")}`,
    ).toEqual([]);
  });

  maybe("_instrumentHuggingFace replaces the chatCompletion submodule export", () => {
    const cache = (__test__ as any)._requireCache();
    const key = Object.keys(cache).find(
      (k) => k.includes(PKG) && k.replace(/\\/g, "/").endsWith("/tasks/nlp/chatCompletion.js"),
    )!;
    const before = cache[key].exports.chatCompletion;
    (__test__ as any)._instrumentHuggingFace(hfModule);
    (__test__ as any)._setInstrumented(true);
    expect(
      cache[key].exports.chatCompletion,
      "chatCompletion was NOT wrapped — HuggingFace has no OpenLLMetry " +
        "instrumentor, so this seam is the only source of its tokens",
    ).not.toBe(before);
  });
});

describe("@huggingface/inference — round trip: chatCompletion (manual telemetry)", () => {
  maybe("reports the OpenAI-compatible usage under the huggingface shape", async () => {
    const net = routedGlobalFetch([
      {
        match: "huggingface.co",
        respond: jsonResponder({
          id: "hf-1",
          object: "chat.completion",
          created: 1,
          model: "meta-llama/Llama-3.1-8B-Instruct",
          choices: [
            { index: 0, message: { role: "assistant", content: "hi" }, finish_reason: "stop" },
          ],
          usage: { prompt_tokens: 14, completion_tokens: 3, total_tokens: 17 },
        }),
      },
    ]);
    try {
      (__test__ as any)._instrumentHuggingFace(hfModule);
      (__test__ as any)._setInstrumented(true);
      const client = new hfModule.InferenceClient("hf_test_not_a_real_key");
      await client.chatCompletion({
        model: "meta-llama/Llama-3.1-8B-Instruct",
        messages: [{ role: "user", content: "hello" }],
      });
      await flushLogs();

      const row = h.only((l) => l.provider === "huggingface", "huggingface chat");
      expect(row.model).toBe("meta-llama/Llama-3.1-8B-Instruct");
      expect(row.inputTokens).toBe(14);
      expect(row.outputTokens).toBe(3);
      expect(row.extra?.usage?.shape).toBe("huggingface_chat");
    } finally {
      net.restore();
    }
  });
});
