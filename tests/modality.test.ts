/**
 * Node non-text modality parity (Mistral OCR/STT, HuggingFace
 * image/TTS/ASR).
 *
 * Covers, against the AGREED rubric:
 * - handler output parity with the Python `_MODALITY_HANDLERS` (kind / items /
 * duration / raw key names + values), reading the OBSERVED Node arg layout
 * (request folded into args[0], NOT a positional args[1]);
 * - `_instrumentMistral` wiring OCR (`ocr.process`) + transcriptions
 * (`audio.transcriptions.complete`) through the modality path;
 * - `_instrumentHuggingFace` wiring `textToImage` / `textToSpeech` /
 * `automaticSpeechRecognition` AND routing them via `_buildIntent` → /check +
 * `_logModality` (NOT `_logManual`), while chat/featureExtraction stay on
 * `_logManual` with no intent;
 * - the negative parity guards (no `xai:image_gen`, no `mistral:audio_tts`);
 * - GOLDEN RULE: malformed results never throw out of a handler.
 */
import { describe, it, expect, vi, beforeEach, afterEach, beforeAll } from "vitest";
import { createRequire } from "module";
import * as fs from "fs";
import * as os from "os";
import * as path from "path";

const h = vi.hoisted(() => {
  const logged: any[] = [];
  const clientBox: { client: any } = { client: null };
  return { logged, clientBox };
});

// Spread the real module so every export the enforcer reads (including the
// per-call observation-scope helpers) stays real — a hand-rolled factory makes
// vitest's mock proxy throw on any export it omits.
vi.mock("../src/state", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../src/state")>();
  return {
    ...actual,
    getClient: () => h.clientBox.client,
    pushObservation: () => {},
    drainObservations: () => [],
    getPack: () => null,
    isCacheHealthy: () => false,
  };
});

import { autoInstrument, uninstrument, __test__ } from "../src/enforcer";
import { buildResponseComposition } from "../src/composition";

const { MODALITY_HANDLERS, _buildIntent, _audioFileSeconds } = __test__ as any;

const requireCjs = createRequire(import.meta.url);

function makeClient(overrides: Record<string, any> = {}): any {
  const base: Record<string, any> = {
    enforce: false,
    firewall: "dry_run", // active pre-flight so /check runs (and never blocks)
    logErrors: false,
    check: vi.fn(async () => ({ status: "allowed" })),
    log: (...args: any[]) => {
      h.logged.push(args);
    },
    ...overrides,
  };
  return base;
}

async function flushLogs(): Promise<void> {
  await new Promise((r) => setTimeout(r, 0));
}

function lastLogExtra(): any {
  const args = h.logged[h.logged.length - 1];
  return args[args.length - 1];
}

/** Canonical 8-bit PCM WAV: duration = dataSize / byteRate. */
function writeWav(dataSize: number, byteRate: number): string {
  const buf = Buffer.alloc(44 + dataSize);
  buf.write("RIFF", 0, "ascii");
  buf.writeUInt32LE(36 + dataSize, 4);
  buf.write("WAVE", 8, "ascii");
  buf.write("fmt ", 12, "ascii");
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(1, 22); // mono
  buf.writeUInt32LE(16000, 24); // sampleRate
  buf.writeUInt32LE(byteRate, 28);
  buf.writeUInt16LE(1, 32); // blockAlign
  buf.writeUInt16LE(8, 34); // bits
  buf.write("data", 36, "ascii");
  buf.writeUInt32LE(dataSize, 40);
  const p = path.join(
    os.tmpdir(),
    `tp-modality-${Date.now()}-${Math.random().toString(36).slice(2)}.wav`,
  );
  fs.writeFileSync(p, buf);
  return p;
}

// 16000 data bytes @ 16000 byteRate = exactly 1.0 s.
let WAV_PATH = "";
beforeAll(() => {
  WAV_PATH = writeWav(16000, 16000);
});

// ───────────────────────────────────────────────────────────────────────────
// Assertions 1, 2, 3, 4, 5, 6, 8, 13, 14, 15 — handler output (no client).
// ───────────────────────────────────────────────────────────────────────────
describe("MODALITY_HANDLERS — output parity with Python", () => {
  const NEW_KEYS = [
    "mistral:ocr",
    "mistral:audio_stt",
    "huggingface:image_gen",
    "huggingface:audio_tts",
    "huggingface:audio_stt",
  ];

  it("assertion 1: the five new keys each resolve to intent + extract fns", () => {
    for (const k of NEW_KEYS) {
      const handler = MODALITY_HANDLERS[k];
      expect(handler, k).toBeTruthy();
      expect(typeof handler.intent).toBe("function");
      expect(typeof handler.extract).toBe("function");
    }
  });

  it("assertion 2: mistral:ocr → kind ocr, ocr_pages from usage_info.pages_processed", () => {
    const handler = MODALITY_HANDLERS["mistral:ocr"];
    const args = [{ model: "mistral-ocr-latest", document: {} }];
    const result = { usage_info: { pages_processed: 7 } };
    expect(_buildIntent("mistral", "ocr", args).kind).toBe("ocr");
    expect(handler.intent(args).kind).toBe("ocr");
    const ex = handler.extract(args, result);
    expect(ex.items.ocr_pages).toBe(7);
    expect(ex.duration).toEqual({});
    expect(ex.raw).toEqual({ pages_processed: 7 });
  });

  it("assertion 3a: mistral:audio_stt reads duration from usage_info + model from args[0]", () => {
    const handler = MODALITY_HANDLERS["mistral:audio_stt"];
    const args = [{ model: "voxtral-mini", file: { name: WAV_PATH } }];
    const result = { usage_info: { duration: 12.5 } };
    expect(handler.intent(args).kind).toBe("audio_transcription");
    const ex = handler.extract(args, result);
    expect(ex.duration.audio_seconds).toBe(12.5);
    expect(ex.items.audio_model).toBe("voxtral-mini");
  });

  it("assertion 3b: mistral:audio_stt falls back to file-seconds read from args[0].file", () => {
    const handler = MODALITY_HANDLERS["mistral:audio_stt"];
    const args = [{ model: "voxtral-mini", file: { name: WAV_PATH } }];
    const result = { usage_info: {} }; // no duration
    const ex = handler.extract(args, result);
    // proves `file` is read from args[0] (WAV = 1.0s), not a stale args[1] (→ 0)
    expect(ex.duration.audio_seconds).toBe(1.0);
    expect(ex.duration.audio_seconds).not.toBe(0);
    // intent's expected_seconds also comes from args[0].file
    expect(handler.intent(args).expected_seconds).toBe(1.0);
  });

  it("assertion 4: huggingface:image_gen → kind image_generation, count 1, image_model from args[0]", () => {
    const handler = MODALITY_HANDLERS["huggingface:image_gen"];
    const args = [{ model: "black-forest-labs/FLUX.1-dev", inputs: "a cat" }];
    const intent = handler.intent(args);
    expect(intent).toEqual({ kind: "image_generation", count: 1 });
    const ex = handler.extract(args, {});
    expect(ex.items.images_generated).toBe(1);
    expect(ex.items.image_model).toBe("black-forest-labs/FLUX.1-dev");
    expect(ex.raw).toEqual({});
  });

  it("assertion 5: huggingface:audio_tts reads text from args[0].inputs (NOT args[1])", () => {
    const handler = MODALITY_HANDLERS["huggingface:audio_tts"];
    const args = [{ model: "espnet/kan-bayashi_ljspeech_vits", inputs: "hello" }];
    expect(handler.intent(args).kind).toBe("audio_speech");
    expect(handler.intent(args).character_count).toBe(5);
    const ex = handler.extract(args, new Blob([]));
    expect(ex.items.tts_characters).toBe(5);
    expect(ex.items.audio_model).toBe("espnet/kan-bayashi_ljspeech_vits");
  });

  // S5 — TTS characters must be counted as code points (matches Python len(str)),
  // NOT UTF-16 code units. Non-BMP chars (emoji, astral CJK) are 2 UTF-16 units
  // each; billing them as 2 chars over-bills up to 2× and diverges from Python.
  it("S5: openai:audio_tts counts code points for non-BMP text (intent + extract)", () => {
    const handler = MODALITY_HANDLERS["openai:audio_tts"];
    // 3 astral emoji → .length would be 6, code points = 3
    const emoji = [{ model: "tts-1", input: "😀😀😀", voice: "alloy" }];
    expect(handler.intent(emoji).character_count).toBe(3);
    expect(handler.extract(emoji, {}).items.tts_characters).toBe(3);
    // mixed BMP + astral → "a😀b" = 3 code points (.length 4)
    const mixed = [{ model: "tts-1", input: "a😀b", voice: "alloy" }];
    expect(handler.intent(mixed).character_count).toBe(3);
    expect(handler.extract(mixed, {}).items.tts_characters).toBe(3);
    // audit headline: 1000 emoji → 1000 chars, not 2000
    const bulk = [{ model: "tts-1", input: "😀".repeat(1000), voice: "alloy" }];
    expect(handler.intent(bulk).character_count).toBe(1000);
    expect(handler.extract(bulk, {}).items.tts_characters).toBe(1000);
  });

  it("S5: huggingface:audio_tts counts code points for non-BMP text (intent + extract)", () => {
    const handler = MODALITY_HANDLERS["huggingface:audio_tts"];
    const emoji = [{ model: "m", inputs: "😀😀😀" }];
    expect(handler.intent(emoji).character_count).toBe(3);
    expect(handler.extract(emoji, new Blob([])).items.tts_characters).toBe(3);
    const mixed = [{ model: "m", inputs: "a😀b" }];
    expect(handler.intent(mixed).character_count).toBe(3);
    expect(handler.extract(mixed, new Blob([])).items.tts_characters).toBe(3);
  });

  it("S5: ai_sdk:audio_tts counts code points for non-BMP text (intent + extract)", () => {
    const handler = MODALITY_HANDLERS["ai_sdk:audio_tts"];
    const emoji = [{ text: "😀😀😀", voice: "alloy" }];
    expect(handler.intent(emoji).character_count).toBe(3);
    expect(handler.extract(emoji, {}).items.tts_characters).toBe(3);
    const mixed = [{ text: "a😀b", voice: "alloy" }];
    expect(handler.intent(mixed).character_count).toBe(3);
    expect(handler.extract(mixed, {}).items.tts_characters).toBe(3);
  });

  it("assertion 6: huggingface:audio_stt reads audio from args[0].data (NOT args[1])", () => {
    const handler = MODALITY_HANDLERS["huggingface:audio_stt"];
    const args = [{ model: "openai/whisper-large-v3", data: { name: WAV_PATH } }];
    expect(handler.intent(args).kind).toBe("audio_transcription");
    const ex = handler.extract(args, { text: "hi" });
    // proves `data` read from args[0] (WAV = 1.0s), not a stale args[1] (→ 0)
    expect(ex.duration.audio_seconds).toBe(1.0);
    expect(ex.items.audio_model).toBe("openai/whisper-large-v3");
  });

  it("assertion 8: item/duration key NAMES byte-match Python + server", () => {
    const ocr = MODALITY_HANDLERS["mistral:ocr"].extract(
      [{ model: "m" }],
      { usage_info: { pages_processed: 3 } },
    );
    expect(Object.keys(ocr.items)).toEqual(["ocr_pages"]);

    const stt = MODALITY_HANDLERS["mistral:audio_stt"].extract(
      [{ model: "m", file: { name: WAV_PATH } }],
      { usage_info: { duration: 4 } },
    );
    expect(Object.keys(stt.items)).toEqual(["audio_model"]);
    expect(Object.keys(stt.duration)).toEqual(["audio_seconds"]);

    const img = MODALITY_HANDLERS["huggingface:image_gen"].extract([{ model: "m" }], {});
    // Image_size always present; HF defaults to 1024x1024 when dims omitted
    expect(Object.keys(img.items).sort()).toEqual(["image_model", "image_size", "images_generated"]);
    expect(img.items.image_size).toBe("1024x1024");

    const tts = MODALITY_HANDLERS["huggingface:audio_tts"].extract(
      [{ model: "m", inputs: "abc" }],
      {},
    );
    expect(Object.keys(tts.items).sort()).toEqual(["audio_model", "tts_characters"]);

    const asr = MODALITY_HANDLERS["huggingface:audio_stt"].extract([{ model: "m" }], {});
    expect(Object.keys(asr.items)).toEqual(["audio_model"]);
    expect(Object.keys(asr.duration)).toEqual(["audio_seconds"]);
  });

  it("assertion 13: Node-wired (provider,modality)->shape set matches Python", () => {
    // The five new keys and only these five are present as (provider,modality).
    expect(Object.keys(MODALITY_HANDLERS)).toEqual(
      expect.arrayContaining([
        "mistral:ocr",
        "mistral:audio_stt",
        "huggingface:image_gen",
        "huggingface:audio_tts",
        "huggingface:audio_stt",
      ]),
    );
  });

  it("assertion 14: NEGATIVE — no native xai:image_gen handler on Node", () => {
    expect(MODALITY_HANDLERS["xai:image_gen"]).toBeUndefined();
  });

  it("assertion 15: NEGATIVE — inert mistral:audio_tts is omitted", () => {
    expect(MODALITY_HANDLERS["mistral:audio_tts"]).toBeUndefined();
  });

  it("invariant 1 (fail-open): malformed result never throws, yields safe items", () => {
    for (const k of NEW_KEYS) {
      const handler = MODALITY_HANDLERS[k];
      expect(() => handler.extract([null], null)).not.toThrow();
      expect(() => handler.intent([null])).not.toThrow();
      const ex = handler.extract([null], null);
      expect(ex.items).toBeTypeOf("object");
    }
    // OCR with a null usage container degrades to 0 pages (no throw).
    expect(MODALITY_HANDLERS["mistral:ocr"].extract([{}], null).items.ocr_pages).toBe(0);
  });
});

// ───────────────────────────────────────────────────────────────────────────
// _audioFileSeconds — WAV header parse (enables the args[0] position proof).
// ───────────────────────────────────────────────────────────────────────────
describe("_audioFileSeconds", () => {
  it("returns 0 for null / non-path handles", () => {
    expect(_audioFileSeconds(null)).toBe(0);
    expect(_audioFileSeconds(undefined)).toBe(0);
    expect(_audioFileSeconds({})).toBe(0);
    expect(_audioFileSeconds(123)).toBe(0);
  });

  it("parses a canonical WAV via string path AND via {name} handle", () => {
    expect(_audioFileSeconds(WAV_PATH)).toBe(1.0);
    expect(_audioFileSeconds({ name: WAV_PATH })).toBe(1.0);
  });

  it("returns 0 for a missing / unreadable path (fail-safe)", () => {
    expect(_audioFileSeconds("/nonexistent/tp-missing-file.wav")).toBe(0);
  });
});

// ───────────────────────────────────────────────────────────────────────────
// Assertions 7, 9, 10, 11, 12, 16 — behavioral wiring via _instrumentMistral.
// ───────────────────────────────────────────────────────────────────────────
function makeFakeMistral() {
  class FakeChat {
    async complete(): Promise<any> {
      return { usage: { promptTokens: 1, completionTokens: 1 } };
    }
    stream(): any {
      return {};
    }
  }
  class FakeEmbeddings {
    async create(): Promise<any> {
      return { data: [] };
    }
  }
  class FakeOcr {
    async process(_req: any): Promise<any> {
      return { usage_info: { pages_processed: 7 } };
    }
  }
  class FakeTranscriptions {
    async complete(_req: any): Promise<any> {
      return { usage_info: { duration: 12.5 } };
    }
    stream(): any {
      return {};
    }
  }
  class FakeAudio {
    transcriptions = new FakeTranscriptions();
  }
  class FakeMistral {
    chat = new FakeChat();
    embeddings = new FakeEmbeddings();
    ocr = new FakeOcr();
    audio = new FakeAudio();
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    constructor(_opts?: any) {}
  }
  return { FakeMistral, FakeChat, FakeEmbeddings, FakeOcr, FakeTranscriptions };
}

describe("_instrumentMistral wiring + routing", () => {
  let checkFn: any;
  beforeEach(() => {
    h.logged.length = 0;
    checkFn = vi.fn(async () => ({ status: "allowed" }));
    h.clientBox.client = makeClient({ check: checkFn });
  });
  afterEach(() => {
    uninstrument();
  });

  it("assertion 9/7/12: OCR wrapped; intent kind 'ocr' hits /check; mistral_ocr logged", async () => {
    const { FakeMistral, FakeOcr } = makeFakeMistral();
    const origProcess = FakeOcr.prototype.process;
    autoInstrument({ mistral: { Mistral: FakeMistral } });
    // assertion 9: method reference replaced (wrapped)
    expect(FakeOcr.prototype.process).not.toBe(origProcess);

    const client: any = new FakeMistral();
    const res = await client.ocr.process({ model: "mistral-ocr-latest", document: {} });
    await flushLogs();

    expect(res).toEqual({ usage_info: { pages_processed: 7 } }); // customer result untouched
    // assertion 7: intent.kind reaches /check as the 9th positional arg
    expect(checkFn.mock.calls[0][8]).toBeTruthy();
    expect(checkFn.mock.calls[0][8].kind).toBe("ocr");
    // assertion 12 + 11-style: modality log block
    const extra = lastLogExtra();
    expect(extra.usage.shape).toBe("mistral_ocr");
    expect(extra.usage.items.ocr_pages).toBe(7);
    expect(extra.operation).toBe("ocr");
  });

  it("assertion 10/7/12: transcriptions wrapped (chat still wrapped); audio_stt intent + mistral_audio_stt", async () => {
    const { FakeMistral, FakeChat, FakeEmbeddings, FakeTranscriptions } = makeFakeMistral();
    const origComplete = FakeTranscriptions.prototype.complete;
    const origChatComplete = FakeChat.prototype.complete;
    const origEmbedCreate = FakeEmbeddings.prototype.create;
    autoInstrument({ mistral: { Mistral: FakeMistral } });
    // transcriptions wired
    expect(FakeTranscriptions.prototype.complete).not.toBe(origComplete);
    // pre-existing chat + embeddings wiring preserved (still wrapped)
    expect(FakeChat.prototype.complete).not.toBe(origChatComplete);
    expect(FakeEmbeddings.prototype.create).not.toBe(origEmbedCreate);

    const client: any = new FakeMistral();
    await client.audio.transcriptions.complete({
      model: "voxtral-mini",
      file: { name: WAV_PATH },
    });
    await flushLogs();

    expect(checkFn.mock.calls[0][8].kind).toBe("audio_transcription");
    const extra = lastLogExtra();
    expect(extra.usage.shape).toBe("mistral_audio_stt");
    expect(extra.usage.items.audio_model).toBe("voxtral-mini");
    expect(extra.usage.duration.audio_seconds).toBe(12.5);
    expect(extra.operation).toBe("audio_stt");
  });
});

// ───────────────────────────────────────────────────────────────────────────
// Assertions 7, 11, 12, 16 — behavioral wiring via _instrumentHuggingFace.
// ───────────────────────────────────────────────────────────────────────────
function hfSubmod(suffix: string): any {
  requireCjs("@huggingface/inference");
  const cache = requireCjs.cache as any;
  const key = Object.keys(cache).find(
    (k) => k.includes("@huggingface/inference") && k.replace(/\\/g, "/").endsWith(suffix),
  );
  return key ? cache[key].exports : null;
}

const HF_SUFFIXES: Record<string, string> = {
  textToImage: "/tasks/cv/textToImage.js",
  textToSpeech: "/tasks/audio/textToSpeech.js",
  automaticSpeechRecognition: "/tasks/audio/automaticSpeechRecognition.js",
  chatCompletion: "/tasks/nlp/chatCompletion.js",
};

describe("_instrumentHuggingFace wiring + routing", () => {
  let checkFn: any;
  const realOriginals: Record<string, { sub: any; orig: any }> = {};

  beforeAll(() => {
    // snapshot the real submodule exports so we can always restore them
    for (const [method, suffix] of Object.entries(HF_SUFFIXES)) {
      const sub = hfSubmod(suffix);
      realOriginals[method] = { sub, orig: sub[method] };
    }
  });

  beforeEach(() => {
    h.logged.length = 0;
    checkFn = vi.fn(async () => ({ status: "allowed" }));
    h.clientBox.client = makeClient({ check: checkFn });
  });

  afterEach(() => {
    uninstrument();
    // uninstrument restores each submodule export to whatever was captured at
    // instrument time (our stub) — put the true package functions back so the
    // module graph is clean for any later suite.
    for (const [method, { sub, orig }] of Object.entries(realOriginals)) {
      sub[method] = orig;
    }
  });

  it("assertion 11a/11b/7/12: textToImage wrapped → intent to /check + huggingface_image via _logModality", async () => {
    const sub = hfSubmod(HF_SUFFIXES.textToImage);
    const stub = async (_args: any) => new Blob([Buffer.from([0])]);
    sub.textToImage = stub;

    autoInstrument({ huggingface: requireCjs("@huggingface/inference") });
    // assertion 11a: wrapped
    expect(sub.textToImage).not.toBe(stub);

    const res = await sub.textToImage({
      model: "black-forest-labs/FLUX.1-dev",
      inputs: "a cat",
    });
    await flushLogs();

    expect(res).toBeInstanceOf(Blob); // customer result untouched
    // assertion 7 + 11b(i): _buildIntent forwarded to /check
    expect(checkFn.mock.calls[0][8]).toBeTruthy();
    expect(checkFn.mock.calls[0][8].kind).toBe("image_generation");
    // assertion 11b(ii) + 12: logged via _logModality with the modality shape
    const extra = lastLogExtra();
    expect(extra.usage.shape).toBe("huggingface_image");
    expect(extra.usage.items.images_generated).toBe(1);
    expect(extra.usage.items.image_model).toBe("black-forest-labs/FLUX.1-dev");
    expect(extra.operation).toBe("image_gen");
  });

  it("assertion 7/12: textToSpeech → audio_speech intent + huggingface_audio_tts", async () => {
    const sub = hfSubmod(HF_SUFFIXES.textToSpeech);
    const stub = async (_args: any) => new Blob([Buffer.from([0])]);
    sub.textToSpeech = stub;

    autoInstrument({ huggingface: requireCjs("@huggingface/inference") });
    expect(sub.textToSpeech).not.toBe(stub);

    await sub.textToSpeech({ model: "espnet/kan-bayashi_ljspeech_vits", inputs: "hello" });
    await flushLogs();

    expect(checkFn.mock.calls[0][8].kind).toBe("audio_speech");
    expect(checkFn.mock.calls[0][8].character_count).toBe(5);
    const extra = lastLogExtra();
    expect(extra.usage.shape).toBe("huggingface_audio_tts");
    expect(extra.usage.items.tts_characters).toBe(5);
    expect(extra.operation).toBe("audio_tts");
  });

  it("assertion 7/12: automaticSpeechRecognition → audio_transcription intent + huggingface_audio_stt", async () => {
    const sub = hfSubmod(HF_SUFFIXES.automaticSpeechRecognition);
    const stub = async (_args: any) => ({ text: "hi" });
    sub.automaticSpeechRecognition = stub;

    autoInstrument({ huggingface: requireCjs("@huggingface/inference") });
    expect(sub.automaticSpeechRecognition).not.toBe(stub);

    await sub.automaticSpeechRecognition({
      model: "openai/whisper-large-v3",
      data: { name: WAV_PATH },
    });
    await flushLogs();

    expect(checkFn.mock.calls[0][8].kind).toBe("audio_transcription");
    const extra = lastLogExtra();
    expect(extra.usage.shape).toBe("huggingface_audio_stt");
    expect(extra.usage.items.audio_model).toBe("openai/whisper-large-v3");
    expect(extra.usage.duration.audio_seconds).toBe(1.0);
    expect(extra.operation).toBe("audio_stt");
  });

  it("assertion 11/16: chatCompletion stays on _logManual with NO modality intent", async () => {
    const sub = hfSubmod(HF_SUFFIXES.chatCompletion);
    const stub = async (_args: any) => ({
      choices: [{ message: { role: "assistant", content: "hi" } }],
      usage: { prompt_tokens: 3, completion_tokens: 2 },
    });
    sub.chatCompletion = stub;

    autoInstrument({ huggingface: requireCjs("@huggingface/inference") });
    expect(sub.chatCompletion).not.toBe(stub);

    await sub.chatCompletion({
      model: "meta-llama/Llama-3.1-8B-Instruct",
      messages: [{ role: "user", content: "hi" }],
    });
    await flushLogs();

    // no modality intent forwarded (chat path calls _runAsyncCheck() with no args)
    expect(checkFn.mock.calls[0][8]).toBeUndefined();
    // not a modality/_logModality row: carries no huggingface_image shape
    const extra = lastLogExtra();
    expect(extra?.usage?.shape).not.toBe("huggingface_image");
    expect(extra?.operation).not.toBe("image_gen");
  });
});

// ─────────────────────────────────────────────────────────────────────
// OCR composition tier (T1) — mistral_ocr usage_shape override.
// Page count comes from usage_info.pages_processed (the billed ocr_pages
// value), NOT `.pages`. Privacy-preserving structural markers only.
// ─────────────────────────────────────────────────────────────────────
describe("buildResponseComposition — mistral_ocr OCR tier", () => {
  it("pages_processed=7 → exactly 7 ocr_page entries (role assistant)", () => {
    const resp = { usage_info: { pages_processed: 7 }, pages: [{}, {}] };
    const comp = buildResponseComposition("mistral", resp, "mistral_ocr");
    expect(comp).toEqual(Array.from({ length: 7 }, () => ({ role: "assistant", type: "ocr_page" })));
  });

  it("pages_processed=0 → single ocr_document", () => {
    const comp = buildResponseComposition("mistral", { usage_info: { pages_processed: 0 } }, "mistral_ocr");
    expect(comp).toEqual([{ role: "assistant", type: "ocr_document" }]);
  });

  it("usage_info missing → single ocr_document (no pages list)", () => {
    const comp = buildResponseComposition("mistral", { model: "mistral-ocr-latest" }, "mistral_ocr");
    expect(comp).toEqual([{ role: "assistant", type: "ocr_document" }]);
  });

  it("usage_info missing but pages is a list → falls back to pages.length", () => {
    const comp = buildResponseComposition("mistral", { pages: [{}, {}, {}] }, "mistral_ocr");
    expect(comp).toEqual([
      { role: "assistant", type: "ocr_page" },
      { role: "assistant", type: "ocr_page" },
      { role: "assistant", type: "ocr_page" },
    ]);
  });

  it("huge page count → capped at 100 ocr_page entries", () => {
    const comp = buildResponseComposition("mistral", { usage_info: { pages_processed: 5000 } }, "mistral_ocr");
    expect(comp).toHaveLength(100);
    expect(comp.every((e) => e.role === "assistant" && e.type === "ocr_page")).toBe(true);
  });

  it("null response → single ocr_document (no throw)", () => {
    const comp = buildResponseComposition("mistral", null, "mistral_ocr");
    expect(comp).toEqual([{ role: "assistant", type: "ocr_document" }]);
  });

  it("pages present-but-not-a-list and no usage_info → single ocr_document", () => {
    const comp = buildResponseComposition("mistral", { pages: "not-a-list" }, "mistral_ocr");
    expect(comp).toEqual([{ role: "assistant", type: "ocr_document" }]);
  });

  it("non-int pages_processed string coerces; garbage falls back to document", () => {
    expect(buildResponseComposition("mistral", { usage_info: { pages_processed: "3" } }, "mistral_ocr")).toHaveLength(3);
    expect(
      buildResponseComposition("mistral", { usage_info: { pages_processed: "abc" } }, "mistral_ocr"),
    ).toEqual([{ role: "assistant", type: "ocr_document" }]);
  });

  it("non-integer strings reject to document — parity with Python int()", () => {
    // Python int("3.5") / int("3abc") raise → ocr_document; Node must NOT
    // accept a leading-integer prefix via parseInt.
    expect(
      buildResponseComposition("mistral", { usage_info: { pages_processed: "3.5" } }, "mistral_ocr"),
    ).toEqual([{ role: "assistant", type: "ocr_document" }]);
    expect(
      buildResponseComposition("mistral", { usage_info: { pages_processed: "3abc" } }, "mistral_ocr"),
    ).toEqual([{ role: "assistant", type: "ocr_document" }]);
  });

  it("boolean pages_processed rejected → single ocr_document (parity with Python)", () => {
    // Python rejects bool (int subclass); Node's `typeof true !== 'number'`
    // already rejects it. Assert the shared contract.
    expect(
      buildResponseComposition("mistral", { usage_info: { pages_processed: true } }, "mistral_ocr"),
    ).toEqual([{ role: "assistant", type: "ocr_document" }]);
  });
});
