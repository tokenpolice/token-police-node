/**
 * Part 2 — Responses-API image_generation → child openai_images spans.
 *
 * Chat raw usage no longer injects image_output_count; each completed
 * image_generation_call becomes a child log with shape openai_images.
 */
import { beforeEach, describe, expect, it, vi } from "vitest";
import { deflateSync } from "node:zlib";

const h = vi.hoisted(() => {
  const logMock = vi.fn();
  const nextSpanOrder = vi.fn();
  return { logMock, nextSpanOrder };
});

vi.mock("../src/state", () => ({
  getClient: () => ({ log: h.logMock }),
  pushObservation: () => {},
  drainObservations: () => [],
  getPack: () => null,
  isCacheHealthy: () => false,
}));

vi.mock("../src/context", async () => {
  const actual = await vi.importActual<typeof import("../src/context")>("../src/context");
  return {
    ...actual,
    getCurrentSession: () => ({
      userId: "u",
      paidPlan: "free",
      workflowName: "wf",
      sessionId: "sid",
      metadata: {},
      traceId: "t".repeat(32),
      nextSpanOrder: h.nextSpanOrder,
      _pendingCompositions: {},
    }),
    manualSpanIds: () => ({
      trace_id: "t".repeat(32),
      span_id: "s".repeat(16),
      parent_span_id: "p".repeat(16),
    }),
  };
});

import { __test__ } from "../src/enforcer";

const {
  _extractRawUsage,
  _extractResponsesImageToolConfig,
  _listResponsesImageCalls,
  _logResponsesImageChildren,
} = __test__ as any;

function makePng(w: number, h: number): Buffer {
  const crcTable: number[] = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crcTable[n] = c >>> 0;
  }
  const crc = (buf: Buffer) => {
    let c = 0xffffffff;
    for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
    const out = Buffer.alloc(4);
    out.writeUInt32BE((c ^ 0xffffffff) >>> 0, 0);
    return out;
  };
  const chunk = (tag: string, data: Buffer) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length, 0);
    const type = Buffer.from(tag, "ascii");
    return Buffer.concat([len, type, data, crc(Buffer.concat([type, data]))]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  const raw = Buffer.alloc((1 + w * 3) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (1 + w * 3)] = 0;
  }
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw)),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const PNG_64X48_B64 = makePng(64, 48).toString("base64");

describe("Part 2: _extractRawUsage openai_responses", () => {
  it("does not inject image_output_count into chat usage", () => {
    const usage = { input_tokens: 50, output_tokens: 10 };
    const raw = _extractRawUsage("openai_responses", {
      usage,
      output: [
        { type: "image_generation_call", id: "ig_1", result: "x" },
        { type: "message", content: [{ type: "output_text", text: "hi" }] },
      ],
    });
    expect(raw).toBe(usage);
    expect((raw as any).image_output_count).toBeUndefined();
  });

  it("unwraps response.completed usage without image count", () => {
    const usage = { input_tokens: 80, output_tokens: 20 };
    const raw = _extractRawUsage("openai_responses", {
      type: "response.completed",
      response: {
        usage,
        output: [{ type: "image_generation_call", id: "ig" }],
      },
    });
    expect(raw).toBe(usage);
  });
});

describe("Part 2: tool config + list", () => {
  it("defaults model to gpt-image-1", () => {
    expect(_extractResponsesImageToolConfig([])).toEqual({
      model: "gpt-image-1",
      size: "",
      quality: "",
    });
  });

  it("reads tools[] config and clears auto", () => {
    expect(
      _extractResponsesImageToolConfig([
        {
          tools: [
            {
              type: "image_generation",
              model: "gpt-image-1-mini",
              size: "auto",
              quality: "high",
            },
          ],
        },
      ]),
    ).toEqual({ model: "gpt-image-1-mini", size: "", quality: "high" });
  });

  it("lists completed images and skips failed", () => {
    const items = _listResponsesImageCalls({
      output: [
        { type: "image_generation_call", status: "failed", id: "f" },
        {
          type: "image_generation_call",
          status: "completed",
          id: "ok",
          result: "abc",
        },
      ],
    });
    expect(items).toHaveLength(1);
    expect(items[0].id).toBe("ok");
  });

  it("list boom fail-open", () => {
    const boom = {
      get output() {
        throw new Error("boom");
      },
    };
    expect(_listResponsesImageCalls(boom)).toEqual([]);
  });
});

describe("Part 2: _logResponsesImageChildren", () => {
  beforeEach(() => {
    h.logMock.mockReset();
    h.nextSpanOrder.mockReset();
    h.nextSpanOrder.mockReturnValueOnce(10).mockReturnValueOnce(11);
  });

  it("emits one openai_images child per image", () => {
    _logResponsesImageChildren(
      [
        {
          tools: [
            {
              type: "image_generation",
              size: "1024x1024",
              quality: "medium",
            },
          ],
        },
      ],
      {
        output: [
          {
            type: "image_generation_call",
            status: "completed",
            result: PNG_64X48_B64,
          },
          {
            type: "image_generation_call",
            status: "completed",
            result: PNG_64X48_B64,
          },
        ],
      },
      new Date(),
    );
    expect(h.logMock).toHaveBeenCalledTimes(2);
    const args = h.logMock.mock.calls[0];
    expect(args[4]).toBe("gpt-image-1");
    expect(args[5]).toBe("openai");
    const extras = args[args.length - 1];
    expect(extras.operation).toBe("image_gen");
    expect(extras.usage.shape).toBe("openai_images");
    expect(extras.usage.items.images_generated).toBe(1);
    expect(extras.usage.items.image_quality).toBe("medium");
    expect(extras.usage.items.image_size).toBe("1024x1024");
  });

  it("uses binary dims when request size omitted", () => {
    _logResponsesImageChildren(
      [{ tools: [{ type: "image_generation" }] }],
      {
        output: [
          {
            type: "image_generation_call",
            status: "completed",
            result: PNG_64X48_B64,
          },
        ],
      },
      new Date(),
    );
    const call = h.logMock.mock.calls[0];
    const extras = call[call.length - 1];
    expect(extras.usage.items.image_size).toBe("64x48");
  });

  // Layer 3: the image_generation_call item ships only
  // {id,result,status,type} today, so these paths are dormant — but undeclared
  // wire fields survive JSON parsing, so the moment OpenAI echoes the
  // server-resolved dims we capture them and the server can price the exact
  // tier instead of estimating it.
  describe("output-item dims", () => {
    const logChildren = (item: Record<string, unknown>, tools: unknown[] = [{ type: "image_generation" }]) => {
      _logResponsesImageChildren(
        [{ tools }],
        { output: [{ type: "image_generation_call", status: "completed", result: PNG_64X48_B64, ...item }] },
        new Date(),
      );
      const call = h.logMock.mock.calls[0];
      return call[call.length - 1].usage.items;
    };

    it("reads quality off the output item when the request never set it", () => {
      expect(logChildren({ quality: "medium" }).image_quality).toBe("medium");
    });

    it("prefers the item's server-resolved values over the request tool config", () => {
      // The request asked for 'auto'/nothing; the item reports what was produced.
      const items = logChildren(
        { quality: "high", size: "1536x1024" },
        [{ type: "image_generation", quality: "auto", size: "1024x1024" }],
      );
      expect(items.image_quality).toBe("high");
      expect(items.image_size).toBe("1536x1024");
    });

    it("treats an item 'auto' as unobserved rather than a value", () => {
      // '' means "not observed" — do not invent a default quality/size
      // as approximated — passing 'auto' through would contradict every tier.
      const items = logChildren({ quality: "auto", size: "auto" });
      expect(items.image_quality).toBe("");
      expect(items.image_size).toBe("64x48"); // falls through to the b64 header
    });

    it("falls back to the request tool config when the item carries nothing", () => {
      const items = logChildren({}, [{ type: "image_generation", quality: "low", size: "1024x1536" }]);
      expect(items.image_quality).toBe("low");
      expect(items.image_size).toBe("1024x1536");
    });

    it("ignores a non-string item dim", () => {
      expect(logChildren({ quality: { bogus: 1 }, size: 1024 }).image_quality).toBe("");
    });
  });

  it("text-only emits nothing", () => {
    _logResponsesImageChildren(
      [{}],
      {
        output: [
          {
            type: "message",
            content: [{ type: "output_text", text: "hi" }],
          },
        ],
      },
      new Date(),
    );
    expect(h.logMock).not.toHaveBeenCalled();
  });
});
