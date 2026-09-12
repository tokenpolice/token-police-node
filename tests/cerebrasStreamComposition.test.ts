/**
 * I5 — Cerebras streaming response_composition (OpenAI-shaped stream allowlist).
 *
 * Cerebras is OpenAI wire-compatible for stream deltas
 * (`choices[0].delta.{content,tool_calls}`). Usage extraction already handled
 * it; stream composition required membership in all three stages:
 * _newStreamAccumulator / _accumulateStreamChunk / _streamAccumulatorToResponse
 *
 * Without the allowlist, accumulator was null and finalize fell back to the
 * last usage-only chunk → Tier-3 `complete_response`. openrouter intentionally
 * stays out (Mode-A gate incomplete without more work).
 */
import { describe, it, expect } from "vitest";
import { __test__ } from "../src/enforcer";

const {
  _newStreamAccumulator,
  _accumulateStreamChunk,
  _streamAccumulatorToResponse,
} = __test__ as any;

describe("I5 cerebras stream composition allowlist", () => {
  it("creates an accumulator for cerebras (all three stages wired)", () => {
    const acc = _newStreamAccumulator("cerebras");
    expect(acc).not.toBeNull();
    expect(acc).toHaveProperty("textParts");
    expect(acc).toHaveProperty("toolCalls");
  });

  it("openrouter is still not accumulated (scope: Mode-A gate incomplete)", () => {
    expect(_newStreamAccumulator("openrouter")).toBeNull();
  });

  it("content deltas compose assistant text", () => {
    const acc = _newStreamAccumulator("cerebras");
    _accumulateStreamChunk("cerebras", acc, {
      choices: [{ index: 0, delta: { content: "Hello " } }],
    });
    _accumulateStreamChunk("cerebras", acc, {
      choices: [{ index: 0, delta: { content: "world" } }],
    });
    const composed = _streamAccumulatorToResponse("cerebras", acc);
    expect(composed).not.toBeNull();
    expect(composed.choices[0].message.content).toBe("Hello world");
    expect(composed.choices[0].message.role).toBe("assistant");
  });

  it("tool_call deltas merge by index", () => {
    const acc = _newStreamAccumulator("cerebras");
    _accumulateStreamChunk("cerebras", acc, {
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              {
                index: 0,
                id: "call_1",
                type: "function",
                function: { name: "lookup", arguments: '{"q":' },
              },
            ],
          },
        },
      ],
    });
    _accumulateStreamChunk("cerebras", acc, {
      choices: [
        {
          index: 0,
          delta: {
            tool_calls: [
              { index: 0, function: { arguments: '"x"}' } },
            ],
          },
        },
      ],
    });
    const composed = _streamAccumulatorToResponse("cerebras", acc);
    expect(composed).not.toBeNull();
    const tcs = composed.choices[0].message.tool_calls;
    expect(tcs).toHaveLength(1);
    expect(tcs[0].id).toBe("call_1");
    expect(tcs[0].function.name).toBe("lookup");
    expect(tcs[0].function.arguments).toBe('{"q":"x"}');
  });

  it("empty stream returns null (fail-open to last usage chunk)", () => {
    const acc = _newStreamAccumulator("cerebras");
    expect(_streamAccumulatorToResponse("cerebras", acc)).toBeNull();
    _accumulateStreamChunk("cerebras", acc, {
      choices: [{ index: 0, delta: {} }],
    });
    expect(_streamAccumulatorToResponse("cerebras", acc)).toBeNull();
  });

  it("peers still accumulate", () => {
    for (const p of ["openai", "groq", "together", "huggingface", "litellm", "mistral"]) {
      expect(_newStreamAccumulator(p)).not.toBeNull();
    }
  });
});
