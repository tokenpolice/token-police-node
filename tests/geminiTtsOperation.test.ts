/**
 * I3 — Gemini TTS reclass: operation=audio_tts + composition audio parts.
 *
 * Gemini TTS reuses generateContent (same surface as chat). Request-side
 * responseModalities=AUDIO and/or response candidatesTokensDetails AUDIO must
 * reclass to audio_tts on check+log. Stream path must not hardcode "chat".
 * Typed response inlineData audio must not fall to Tier-3 complete_response.
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import { buildResponseComposition } from "../src/composition";
import {
  _wantsGoogleAudioOut,
  _isGoogleAudioOutput,
  __test__,
} from "../src/enforcer";

const {
  _resolveGoogleLogOperation,
  _resolveFailOperation,
  _wrapManualStream,
  _stashAttemptContext,
} = __test__;

describe("_wantsGoogleAudioOut", () => {
  it("detects config.responseModalities AUDIO", () => {
    expect(
      _wantsGoogleAudioOut([
        {
          model: "gemini-2.5-flash-preview-tts",
          config: { responseModalities: ["AUDIO"] },
        },
      ]),
    ).toBe(true);
  });

  it("detects snake_case response_modalities", () => {
    expect(
      _wantsGoogleAudioOut([{ config: { response_modalities: ["AUDIO"] } }]),
    ).toBe(true);
  });

  it("detects speechConfig", () => {
    expect(
      _wantsGoogleAudioOut([{ config: { speechConfig: { voice: "Kore" } } }]),
    ).toBe(true);
  });

  it("text-only chat is false", () => {
    expect(
      _wantsGoogleAudioOut([
        { model: "gemini-2.5-flash", config: { maxOutputTokens: 80 } },
      ]),
    ).toBe(false);
    expect(_wantsGoogleAudioOut([])).toBe(false);
    expect(_wantsGoogleAudioOut([null as any])).toBe(false);
  });

  it("hostile config fail-open", () => {
    const hostile = {
      get config() {
        throw new Error("boom");
      },
    };
    expect(_wantsGoogleAudioOut([hostile])).toBe(false);
  });
});

describe("_isGoogleAudioOutput", () => {
  it("camelCase candidatesTokensDetails AUDIO", () => {
    expect(
      _isGoogleAudioOutput({
        usageMetadata: {
          candidatesTokensDetails: [{ modality: "AUDIO", tokenCount: 42 }],
        },
      }),
    ).toBe(true);
  });

  it("snake_case candidates_tokens_details", () => {
    expect(
      _isGoogleAudioOutput({
        usage_metadata: {
          candidates_tokens_details: [{ modality: "AUDIO", token_count: 10 }],
        },
      }),
    ).toBe(true);
  });

  it("zero audio tokens is false", () => {
    expect(
      _isGoogleAudioOutput({
        usageMetadata: {
          candidatesTokensDetails: [{ modality: "AUDIO", tokenCount: 0 }],
        },
      }),
    ).toBe(false);
  });

  it("text-only details is false", () => {
    expect(
      _isGoogleAudioOutput({
        usageMetadata: {
          candidatesTokensDetails: [{ modality: "TEXT", tokenCount: 20 }],
        },
      }),
    ).toBe(false);
  });

  it("inlineData audio part", () => {
    expect(
      _isGoogleAudioOutput({
        candidates: [
          {
            content: {
              parts: [{ inlineData: { mimeType: "audio/pcm", data: "x" } }],
            },
          },
        ],
      }),
    ).toBe(true);
  });

  it("hostile fail-open", () => {
    const boom = {
      get usageMetadata() {
        throw new Error("x");
      },
      get candidates() {
        throw new Error("y");
      },
    };
    expect(_isGoogleAudioOutput(boom)).toBe(false);
  });
});

describe("_resolveGoogleLogOperation", () => {
  it("reclass from response AUDIO details", () => {
    const op = _resolveGoogleLogOperation(
      "google",
      "chat",
      [{ model: "gemini-2.5-flash" }],
      {
        usageMetadata: {
          candidatesTokensDetails: [{ modality: "AUDIO", tokenCount: 5 }],
        },
      },
    );
    expect(op).toBe("audio_tts");
  });

  it("reclass from request modalities alone", () => {
    const op = _resolveGoogleLogOperation(
      "google",
      "chat",
      [{ config: { responseModalities: ["AUDIO"] } }],
      { usageMetadata: {} },
    );
    expect(op).toBe("audio_tts");
  });

  it("text-only stays chat", () => {
    const op = _resolveGoogleLogOperation(
      "google",
      "chat",
      [{ model: "gemini-2.5-flash", config: { maxOutputTokens: 10 } }],
      {
        usageMetadata: {
          candidatesTokensDetails: [{ modality: "TEXT", tokenCount: 8 }],
        },
      },
    );
    expect(op).toBe("chat");
  });

  it("non-google provider unchanged", () => {
    expect(
      _resolveGoogleLogOperation(
        "openai",
        "chat",
        [{ config: { responseModalities: ["AUDIO"] } }],
        {
          usageMetadata: {
            candidatesTokensDetails: [{ modality: "AUDIO", tokenCount: 5 }],
          },
        },
      ),
    ).toBe("chat");
  });

  it("embedding operation not overwritten", () => {
    expect(
      _resolveGoogleLogOperation(
        "google",
        "embedding",
        [{ config: { responseModalities: ["AUDIO"] } }],
        null,
      ),
    ).toBe("embedding");
  });
});

describe("composition — Google response inlineData audio", () => {
  it("object path audio inlineData → type audio", () => {
    const comp = buildResponseComposition("google", {
      candidates: [
        {
          content: {
            parts: [
              { inlineData: { mimeType: "audio/L16;codec=pcm", data: "AQID" } },
            ],
          },
        },
      ],
    });
    expect(comp).toEqual([{ role: "assistant", type: "audio" }]);
    expect(comp[0].type).not.toBe("complete_response");
  });

  it("snake_case inline_data", () => {
    const comp = buildResponseComposition("google", {
      candidates: [
        {
          content: {
            parts: [
              { inline_data: { mime_type: "audio/mpeg", data: "xx" } },
            ],
          },
        },
      ],
    });
    expect(comp[0].type).toBe("audio");
  });

  it("text chat still text", () => {
    const comp = buildResponseComposition("google", {
      candidates: [{ content: { parts: [{ text: "hello" }] } }],
    });
    expect(comp[0].type).toBe("text");
    expect(comp[0].role).toBe("assistant");
  });
});

describe("failure-path attempt operation", () => {
  it("stashes audio_tts when request wants audio (check/log mirror)", () => {
    const session: any = {};
    // Production helper used by the main manual wrapper.
    const failOperation = _resolveFailOperation(undefined, undefined, {
      kind: "audio_tts",
    });
    _stashAttemptContext(
      session,
      "google",
      [{ model: "gemini-2.5-flash-preview-tts", config: { responseModalities: ["AUDIO"] } }],
      failOperation,
    );
    expect(session._attempted_operation).toBe("audio_tts");
    expect(session._attempted_provider).toBe("google");
    expect(session._attempted_model).toBe("gemini-2.5-flash-preview-tts");
  });

  it("text-only leaves embedding operation intact", () => {
    const session: any = {};
    _stashAttemptContext(
      session,
      "google",
      [{ model: "text-embedding-004" }],
      "embedding",
    );
    expect(session._attempted_operation).toBe("embedding");
  });

  // Modality-only registry rows (no operation) must not fall through to "chat".
  it("Modality image_gen falls back when operation is absent", () => {
    const session: any = {};
    const failOperation = _resolveFailOperation(undefined, "image_gen");
    _stashAttemptContext(
      session,
      "google",
      [{ model: "imagen-4.0-generate-001" }],
      failOperation,
    );
    expect(session._attempted_operation).toBe("image_gen");
  });

  it("Modality audio_tts falls back when operation is absent", () => {
    const session: any = {};
    const failOperation = _resolveFailOperation(undefined, "audio_tts");
    _stashAttemptContext(session, "openai", [{ model: "tts-1" }], failOperation);
    expect(session._attempted_operation).toBe("audio_tts");
  });

  it("Explicit operation wins over modality", () => {
    const session: any = {};
    const failOperation = _resolveFailOperation("embedding", "image_gen");
    _stashAttemptContext(
      session,
      "google",
      [{ model: "text-embedding-004" }],
      failOperation,
    );
    expect(session._attempted_operation).toBe("embedding");
  });

  it("No operation and no modality leaves stash undefined (emit defaults chat)", () => {
    const session: any = {};
    const failOperation = _resolveFailOperation();
    _stashAttemptContext(session, "openai", [{ model: "gpt-4.1-mini" }], failOperation);
    expect(session._attempted_operation).toBeUndefined();
  });

  it("Pure resolver: image_gen / embedding / chat defaults", () => {
    // Direct production-helper assertions — a revert of the expression
    // in enforcer.ts must fail this suite (not a local mirror).
    expect(_resolveFailOperation(undefined, "image_gen")).toBe("image_gen");
    expect(_resolveFailOperation(undefined, "audio_tts")).toBe("audio_tts");
    expect(_resolveFailOperation("embedding", "image_gen")).toBe("embedding");
    expect(_resolveFailOperation(undefined, undefined)).toBeUndefined();
    expect(
      _resolveFailOperation(undefined, undefined, { kind: "audio_tts" }),
    ).toBe("audio_tts");
  });
});

describe("stream wrap reclass", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("_wrapManualStream logs audio_tts for Google AUDIO usage chunk", async () => {
    const { TokenPolice } = await import("../src/client");
    const { setClient } = await import("../src/state");
    const { session } = await import("../src/context");

    const client = new TokenPolice({
      apiKey: "tp_sk_test_i3",
      baseUrl: "http://localhost:59999",
      timeout: 0.1,
      firewall: "dry_run",
    } as any);
    setClient(client);
    const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});

    async function* chunks() {
      yield {
        usageMetadata: {
          candidatesTokensDetails: [{ modality: "AUDIO", tokenCount: 12 }],
          promptTokenCount: 3,
          candidatesTokenCount: 12,
        },
        modelVersion: "gemini-2.5-flash-preview-tts",
      };
    }

    await session({ name: "i3_tts_stream" }, async () => {
      const stream = await _wrapManualStream(
        chunks(),
        "google",
        [
          {
            model: "gemini-2.5-flash-preview-tts",
            config: { responseModalities: ["AUDIO"] },
          },
        ],
        0,
        "span",
        new Date(),
        performance.now(),
      );
      for await (const _ of stream) {
        /* drain */
      }
    });
    await new Promise((r) => setTimeout(r, 0));

    const googleRows = logSpy.mock.calls.filter((c: any[]) => c[5] === "google");
    expect(googleRows.length).toBeGreaterThan(0);
    const extras = googleRows[googleRows.length - 1][13];
    expect(extras?.operation).toBe("audio_tts");
  });
});
