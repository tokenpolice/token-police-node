/**
 * Serving provider vs wire/parse key (streamed OpenAI-compatible gateways).
 *
 * After host remap feeds serving slugs (minimax, xai, …) into chunk-usage
 * and composition parsers. Those parsers must key on the **module** client
 * (OpenAI wire shape), not the billing vendor — otherwise:
 * - minimax → falls through to Google usageMetadata → usage never latched
 * → usage-only terminal chunk still stripped → 0 llm rows
 * - xai → _isAiSdkParse → AI-SDK extract expects camelCase; OpenAI snake_case
 * usage yields 0/0 → guard drops stash → 0 llm rows; composition empty
 *
 * Fix: Mode-A composition + stream tap use `_wireParseKey(moduleProvider)`.
 * Serving still drives /check, override stash, attempt context.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage } from "../src/context";
import { __test__ as enforcerTest, _extractUsage } from "../src/enforcer";

const {
  _wireParseKey,
  _chunkHasUsage,
  _isAiSdkParse,
  _newStreamAccumulator,
  _accumulateStreamChunk,
  _streamAccumulatorToResponse,
  _tapStreamUsageForOnEnd,
} = enforcerTest as any;

describe("Wire parse key", () => {
  it("_wireParseKey is the module provider (identity)", () => {
    expect(_wireParseKey("openai")).toBe("openai");
    expect(_wireParseKey("anthropic")).toBe("anthropic");
    expect(_wireParseKey("xai")).toBe("xai");
  });

  it("OpenAI-shaped usage chunk latches under wire=openai, not serving=minimax", () => {
    const usageChunk = {
      usage: { prompt_tokens: 12, completion_tokens: 7, total_tokens: 19 },
      choices: [],
    };
    // Module/wire key — what the fix passes into the tap
    expect(_chunkHasUsage("openai", usageChunk)).toBe(true);
    const u = _extractUsage("openai", usageChunk, [{ model: "MiniMax-M2" }]);
    expect(u.inputTokens).toBe(12);
    expect(u.outputTokens).toBe(7);

    // Serving slug (pre-fix bug): falls through to Google usageMetadata
    expect(_chunkHasUsage("minimax", usageChunk)).toBe(false);
  });

  it("OpenAI-shaped usage is not usable under serving=xai (AI-SDK parse key)", () => {
    expect(_isAiSdkParse("xai")).toBe(true); // true Vercel-AI / legacy key — unchanged
    expect(_isAiSdkParse("openai")).toBe(false);
    expect(_isAiSdkParse(_wireParseKey("openai"))).toBe(false);

    // OpenAI snake_case usage chunk (OpenAI SDK → api.x.ai)
    const openaiChunk = {
      usage: { prompt_tokens: 3, completion_tokens: 4 },
      choices: [{ delta: { content: "hi" } }],
    };
    expect(_chunkHasUsage("openai", openaiChunk)).toBe(true);
    const uOpen = _extractUsage("openai", openaiChunk, [{ model: "grok-3" }]);
    expect(uOpen.inputTokens).toBe(3);
    expect(uOpen.outputTokens).toBe(4);

    // Serving=xai selects AI-SDK extract (camelCase) → 0/0 → stash guard drops row
    const uXai = _extractUsage("xai", openaiChunk, [{ model: "grok-3" }]);
    expect(uXai.inputTokens).toBe(0);
    expect(uXai.outputTokens).toBe(0);

    // Composition: OpenAI deltas ignored by AI-SDK accumulator
    const accX = _newStreamAccumulator("xai");
    _accumulateStreamChunk("xai", accX, {
      choices: [{ delta: { content: "hi" } }],
    });
    expect(_streamAccumulatorToResponse("xai", accX)).toBeNull();
  });

  it("OpenAI stream accumulator builds composition under wire=openai", () => {
    const acc = _newStreamAccumulator("openai");
    expect(acc).not.toBeNull();
    _accumulateStreamChunk("openai", acc, {
      choices: [{ delta: { content: "Hello " } }],
    });
    _accumulateStreamChunk("openai", acc, {
      choices: [{ delta: { content: "world" } }],
    });
    const resp = _streamAccumulatorToResponse("openai", acc);
    expect(resp.choices[0].message.content).toBe("Hello world");

    // Serving slug has no openai-shaped accumulator branch
    expect(_newStreamAccumulator("minimax")).toBeNull();
  });
});

describe("Stream tap with wire key (integration)", () => {
  let session: TPSession;

  beforeEach(() => {
    setClient({
      log: () => {},
      captureStreamUsage: true,
    } as any);
    session = new TPSession({
      userId: "u1",
      paidPlan: "pro",
      workflowName: "wf",
      traceId: "c".repeat(32),
      rootSpanId: "d".repeat(16),
    });
  });

  afterEach(() => {
    setClient(null as any);
  });

  it("taps OpenAI usage + strips synthetic usage-only chunk when suppress=true", async () => {
    expect(typeof _tapStreamUsageForOnEnd).toBe("function");

    const chunks = [
      { choices: [{ delta: { content: "Hi" } }] },
      {
        usage: { prompt_tokens: 10, completion_tokens: 2, total_tokens: 12 },
        choices: [],
      },
    ];
    async function* gen() {
      for (const c of chunks) yield c;
    }
    const stream = gen();

    await _getSessionStorage().run(session, async () => {
      const tapped = _tapStreamUsageForOnEnd(
        _wireParseKey("openai"),
        stream,
        session,
        [{ model: "gpt-4o-mini", messages: [{ role: "user", content: "x" }] }],
        0,
        true, // suppress injected usage-only chunk
      );
      const seen: any[] = [];
      for await (const c of tapped) seen.push(c);
      // Usage-only terminal chunk stripped from customer iterator
      expect(seen).toHaveLength(1);
      expect(seen[0].choices[0].delta.content).toBe("Hi");

      const cd = (session as any)._pendingCompositions[`${session.traceId}:0`];
      expect(cd?.usage?.input_tokens).toBe(10);
      expect(cd?.usage?.output_tokens).toBe(2);
      expect(cd?.response?.length).toBeGreaterThan(0);
    });
  });

  it("customer-owned include_usage chunk is NOT stripped (suppress=false)", async () => {
    const chunks = [
      { choices: [{ delta: { content: "Hi" } }] },
      {
        usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
        choices: [],
      },
    ];
    async function* gen() {
      for (const c of chunks) yield c;
    }

    await _getSessionStorage().run(session, async () => {
      const tapped = _tapStreamUsageForOnEnd(
        _wireParseKey("openai"),
        gen(),
        session,
        [{ model: "gpt-4o-mini", messages: [{ role: "user", content: "x" }] }],
        0,
        false, // customer asked for include_usage — chunk is theirs
      );
      const seen: any[] = [];
      for await (const c of tapped) seen.push(c);
      expect(seen).toHaveLength(2);
      expect(seen[1].usage.prompt_tokens).toBe(5);

      const cd = (session as any)._pendingCompositions[`${session.traceId}:0`];
      expect(cd?.usage?.input_tokens).toBe(5);
    });
  });

  it("serving slug minimax would miss usage that wire=openai latches", async () => {
    // Documents the pre-fix bug: if tap keyed on serving, usage never latches.
    const usageChunk = {
      usage: { prompt_tokens: 9, completion_tokens: 1, total_tokens: 10 },
      choices: [],
    };
    expect(_chunkHasUsage("minimax", usageChunk)).toBe(false);
    expect(_chunkHasUsage(_wireParseKey("openai"), usageChunk)).toBe(true);
  });
});
