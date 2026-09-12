/**
 * Node-8 — LangChain `_streamIterator` response-composition capture is O(n),
 * not O(n²).
 *
 * The streamIterator wrapper (_setLangchainWrapper, kind "streamIterator")
 * accumulates streamed AIMessageChunks into a single message so the SDK can
 * emit a real response composition (content + tool_calls) for the LLM span.
 * Previously it rebuilt the ENTIRE composition (hashing the whole accumulated
 * content) on EVERY chunk → O(n²) CPU in the customer's event loop on long
 * agent streams. The fix moves that single rebuild OUT of the per-chunk loop
 * into a `finally`, so it runs exactly ONCE per stream, on every exit path
 * (full drain, early break, mid-stream throw), with the SAME final composition.
 *
 * These tests drive the REAL wrapper (installed via
 * __test__._instrumentLangChainChatModels onto a fake BaseChatModel prototype)
 * inside a real session context, and count composition rebuilds by spying on
 * buildResponseComposition (wrapped to call through to the real implementation,
 * so the captured composition is genuine).
 */
import { describe, test, expect, beforeEach, afterEach, vi } from "vitest";

// Wrap buildResponseComposition so we can COUNT rebuilds while still producing
// real compositions. The mock applies to the enforcer's `./composition` import
// (same resolved module id) and to our direct import below.
vi.mock("../src/composition", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/composition")>();
  return {
    ...actual,
    buildResponseComposition: vi.fn(actual.buildResponseComposition),
  };
});

import { __test__ } from "../src/enforcer";
import { buildResponseComposition } from "../src/composition";
import { TPSession, _getSessionStorage } from "../src/context";

const buildResponseCompositionMock = vi.mocked(buildResponseComposition);

/** Count how many times a LangChain response composition was (re)built. */
function langchainBuildCount(): number {
  return buildResponseCompositionMock.mock.calls.filter(
    (c) => c[0] === "langchain",
  ).length;
}

/**
 * Minimal AIMessageChunk mock: a text-carrying assistant chunk with the merge
 * (`concat`) semantics the wrapper relies on. `_getType()` makes the langchain
 * composition parser classify it as an assistant message.
 */
function makeChunk(text: string): any {
  return {
    content: text,
    tool_calls: [],
    _getType() {
      return "ai";
    },
    concat(other: any) {
      return makeChunk(this.content + (other?.content ?? ""));
    },
  };
}

/** A fake @langchain/core BaseChatModel whose _streamIterator yields `chunks`. */
function makeFakeBaseChatModel(
  chunks: any[],
  opts: { throwAt?: number } = {},
): any {
  class BaseChatModel {
    model = "fake-chat-model";
    async *_streamIterator(_input: any, _options?: any): AsyncGenerator<any> {
      for (let i = 0; i < chunks.length; i++) {
        if (opts.throwAt === i) throw new Error("boom");
        yield chunks[i];
      }
    }
  }
  return BaseChatModel;
}

/** Install the real wrapper onto a fresh fake prototype. */
function instrument(BaseChatModel: any): void {
  __test__._instrumentLangChainChatModels({ BaseChatModel });
}

let session: TPSession;

beforeEach(() => {
  buildResponseCompositionMock.mockClear();
  session = new TPSession();
});

afterEach(() => {
  // Restore any prototype patches this test installed.
  for (const thunk of __test__._restoreThunks.splice(0)) {
    try {
      thunk();
    } catch {
      /* ignore */
    }
  }
});

/** compKey the streamIterator path writes to (reserved nextSpanOrder() → 0). */
function compKey(): string {
  return `${session.traceId}:0`;
}

/** Run `fn` inside the session's ALS context so getCurrentSession() is stable. */
function inSession<T>(fn: () => Promise<T>): Promise<T> {
  return _getSessionStorage().run(session, fn);
}

describe("LangChain _streamIterator composition capture (Node-8)", () => {
  test("rebuilds the composition exactly ONCE for a multi-chunk stream", async () => {
    const chunks = ["Hel", "lo ", "wor", "ld", "!"].map(makeChunk);
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const model = new BCM();

    const received: any[] = [];
    await inSession(async () => {
      for await (const c of model._streamIterator("hi")) received.push(c);
    });

    // Exactly one rebuild despite 5 chunks (was 5 pre-fix — O(n²)).
    expect(langchainBuildCount()).toBe(1);
    expect(received).toHaveLength(5);
  });

  test("final captured composition equals the fully-accumulated content", async () => {
    const parts = ["Hel", "lo ", "wor", "ld", "!"];
    const chunks = parts.map(makeChunk);
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const model = new BCM();

    await inSession(async () => {
      for await (const _ of model._streamIterator("hi")) {
        /* drain */
      }
    });

    // Independently computed expected composition from the full concatenation.
    const expected = buildResponseCompositionMock("langchain", {
      generations: [[{ message: makeChunk(parts.join("")) }]],
    });
    const captured = session._pendingCompositions[compKey()]?.response;
    expect(captured).toEqual(expected);
    // Sanity: it reflects the whole stream, not just the first chunk.
    expect((captured as any)[0].length).toBe(parts.join("").length);
  });

  test("early break captures the PARTIAL accumulation exactly once, no throw", async () => {
    const parts = ["one", "two", "three", "four", "five"];
    const chunks = parts.map(makeChunk);
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const model = new BCM();

    const received: any[] = [];
    await expect(
      inSession(async () => {
        let n = 0;
        for await (const c of model._streamIterator("hi")) {
          received.push(c);
          if (++n === 2) break;
        }
      }),
    ).resolves.toBeUndefined();

    expect(received).toHaveLength(2);
    // Captured once, with only the first two chunks accumulated.
    expect(langchainBuildCount()).toBe(1);
    const expected = buildResponseCompositionMock("langchain", {
      generations: [[{ message: makeChunk("onetwo") }]],
    });
    expect(session._pendingCompositions[compKey()]?.response).toEqual(expected);
  });

  test("mid-stream throw: error propagates unchanged, partial capture recorded, no unhandled rejection", async () => {
    const parts = ["aa", "bb", "cc", "dd"];
    const chunks = parts.map(makeChunk);
    // Throw when the underlying iterator is asked for the 3rd item (index 2),
    // after chunks 0 and 1 were yielded.
    const BCM = makeFakeBaseChatModel(chunks, { throwAt: 2 });
    instrument(BCM);
    const model = new BCM();

    const unhandled: unknown[] = [];
    const onUnhandled = (e: unknown) => unhandled.push(e);
    process.on("unhandledRejection", onUnhandled);

    const received: any[] = [];
    try {
      await expect(
        inSession(async () => {
          for await (const c of model._streamIterator("hi")) received.push(c);
        }),
      ).rejects.toThrow("boom");

      // Two chunks made it to the customer before the throw.
      expect(received).toHaveLength(2);
      // Partial composition captured exactly once.
      expect(langchainBuildCount()).toBe(1);
      const expected = buildResponseCompositionMock("langchain", {
        generations: [[{ message: makeChunk("aabb") }]],
      });
      expect(session._pendingCompositions[compKey()]?.response).toEqual(
        expected,
      );

      // Let any stray rejection surface.
      await new Promise((r) => setTimeout(r, 0));
      expect(unhandled).toEqual([]);
    } finally {
      process.off("unhandledRejection", onUnhandled);
    }
  });

  test("chunks reach the customer as the SAME objects, in order", async () => {
    const chunks = ["x", "y", "z"].map(makeChunk);
    const BCM = makeFakeBaseChatModel(chunks);
    instrument(BCM);
    const model = new BCM();

    const received: any[] = [];
    await inSession(async () => {
      for await (const c of model._streamIterator("hi")) received.push(c);
    });

    expect(received).toHaveLength(chunks.length);
    received.forEach((c, i) => expect(c).toBe(chunks[i]));
  });
});
