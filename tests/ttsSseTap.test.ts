/**
 * OpenAI TTS SSE tap fallback.
 *
 * Invariant under test: the tap must never hand the customer an empty rebuilt
 * body when the provider actually sent bytes. If the per-frame audio field is
 * renamed/moved so extraction yields nothing, the tap falls back to the
 * original raw SSE bytes verbatim (original content-type preserved) while still
 * attaching the captured usage so metering keeps working.
 */
import { describe, it, expect } from "vitest";
import { __test__ } from "../src/enforcer";

const { _maybeTapOpenAITtsSse, _parseOpenAITtsSseBuffer, TP_CAPTURED_USAGE } =
  __test__ as any;

const DONE_USAGE = { input_tokens: 11, output_tokens: 0, total_tokens: 11 };
const TARGET = { modality: "audio_tts", shape: "openai_audio_tts" };
const ARGS = [{ stream_format: "sse", model: "gpt-4o-mini-tts", voice: "alloy" }];

function b64(s: string): string {
  return Buffer.from(s, "utf-8").toString("base64");
}

function frame(obj: any): string {
  return `data: ${JSON.stringify(obj)}\n\n`;
}

/** Duck-typed fetch Response whose bytes are consumed once via arrayBuffer(). */
function makeSseResponse(sse: string, contentType = "text/event-stream"): any {
  const bytes = Buffer.from(sse, "utf-8");
  const headers = new Headers();
  headers.set("content-type", contentType);
  headers.set("x-request-id", "req_123");
  return {
    status: 200,
    statusText: "OK",
    body: {}, // truthy — the tap bails when body is missing
    headers,
    arrayBuffer: async () =>
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  };
}

function renamedFieldSse(): string {
  // Provider renamed the per-frame audio field: `delta` instead of `audio`.
  return (
    frame({ type: "speech.audio.delta", delta: b64("AUDIO_CHUNK_1") }) +
    frame({ type: "speech.audio.delta", delta: b64("AUDIO_CHUNK_2") }) +
    frame({ type: "speech.audio.done", usage: DONE_USAGE }) +
    "data: [DONE]\n\n"
  );
}

function wellformedSse(): { sse: string; audio: Buffer } {
  const c1 = "AUDIO_CHUNK_1";
  const c2 = "AUDIO_CHUNK_2";
  const sse =
    frame({ type: "speech.audio.delta", audio: b64(c1) }) +
    frame({ type: "speech.audio.delta", audio: b64(c2) }) +
    frame({ type: "speech.audio.done", usage: DONE_USAGE }) +
    "data: [DONE]\n\n";
  return { sse, audio: Buffer.from(c1 + c2, "utf-8") };
}

describe("OpenAI TTS SSE tap", () => {
  it("pre-fix evidence: parser extracts zero audio when the field is renamed", () => {
    // The empty buffer this returns was the WHOLE response body pre-fix.
    const { audio, usage } = _parseOpenAITtsSseBuffer(
      Buffer.from(renamedFieldSse(), "utf-8"),
    );
    expect(audio.length).toBe(0);
    expect(usage).toEqual(DONE_USAGE);
  });

  it("falls back to original SSE bytes verbatim when the audio field moved", async () => {
    const sse = renamedFieldSse();
    const out = await _maybeTapOpenAITtsSse(
      "openai",
      TARGET,
      ARGS,
      makeSseResponse(sse),
    );
    const body = Buffer.from(await out.arrayBuffer());
    // Body equals the original raw SSE bytes (NOT an empty body).
    expect(body.equals(Buffer.from(sse, "utf-8"))).toBe(true);
    expect(body.length).toBeGreaterThan(0);
    // Original content-type preserved — NOT forced to audio/mpeg.
    expect(out.headers.get("content-type")).toBe("text/event-stream");
    // Captured usage still attached for metering.
    expect(out[TP_CAPTURED_USAGE]).toEqual(DONE_USAGE);
  });

  it("rebuilds an audio/mpeg body on well-formed frames", async () => {
    const { sse, audio } = wellformedSse();
    const out = await _maybeTapOpenAITtsSse(
      "openai",
      TARGET,
      ARGS,
      makeSseResponse(sse),
    );
    const body = Buffer.from(await out.arrayBuffer());
    expect(body.equals(audio)).toBe(true);
    expect(out.headers.get("content-type")).toBe("audio/mpeg");
    expect(out[TP_CAPTURED_USAGE]).toEqual(DONE_USAGE);
  });

  it("leaves a genuinely empty 200 unchanged (no throw, empty body)", async () => {
    const out = await _maybeTapOpenAITtsSse(
      "openai",
      TARGET,
      ARGS,
      makeSseResponse(""),
    );
    const body = Buffer.from(await out.arrayBuffer());
    expect(body.length).toBe(0);
  });

  it("skips malformed JSON frames and still extracts valid audio", async () => {
    const sse =
      "data: {not valid json\n\n" +
      frame({ type: "speech.audio.delta", audio: b64("GOOD") }) +
      frame({ type: "speech.audio.done", usage: DONE_USAGE });
    const out = await _maybeTapOpenAITtsSse(
      "openai",
      TARGET,
      ARGS,
      makeSseResponse(sse),
    );
    const body = Buffer.from(await out.arrayBuffer());
    expect(body.equals(Buffer.from("GOOD", "utf-8"))).toBe(true);
    expect(out.headers.get("content-type")).toBe("audio/mpeg");
    expect(out[TP_CAPTURED_USAGE]).toEqual(DONE_USAGE);
  });
});
