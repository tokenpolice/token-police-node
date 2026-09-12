/**
 * uninstrument() must restore probe-derived (not-root-exported) wrappers.
 *
 * Many SDK classes are patched via the "probe-instance trick": we build a
 * throw-away instance to reach a prototype that the package does NOT export at
 * its root (Cohere v2's V2Client, OpenRouter's Chat, Voyage, the
 * LangChain/LlamaIndex framework classes, Anthropic Batches). The old
 * uninstrument() re-resolved a string key (`require(moduleName)` + a path off
 * the package root) and therefore silently failed to restore any of these —
 * and for Voyage it actively wrote a bogus STATIC method onto the constructor
 * while leaving the prototype wrapper live.
 *
 * The fix captures a restore thunk at each patch site. These tests assert, for
 * representative probe-derived surfaces:
 * - after uninstrument() each patched method is reference-identical to the
 * original again;
 * - no static method leaks onto the class (the Voyage double-entry bug);
 * - one thunk throwing does not prevent the others from restoring;
 * - re-instrument() after uninstrument() re-wraps (idempotency markers cleared).
 *
 * The wrappers are only exercised for identity here (patch/restore), never
 * called, so no live client/session is needed.
 */
import { describe, it, expect, afterEach } from "vitest";
import { uninstrument, __test__ } from "../src/enforcer";

const {
  _instrumentCohere,
  _instrumentOpenRouter,
  _instrumentVoyage,
  _instrumentAnthropicBatches,
  _instrumentLangChainChatModels,
  _restoreThunks,
  _setInstrumented,
} = __test__ as any;

// uninstrument() early-returns unless _isInstrumented; the individual
// _instrumentX() helpers don't set it, so flip it on before restoring.
function restore(): void {
  _setInstrumented(true);
  uninstrument();
}

afterEach(() => {
  // Force a clean slate between tests (clears _originals + _restoreThunks).
  try {
    restore();
  } catch {
    /* ignore */
  }
});

// ── Fakes: each mimics the shape the corresponding _instrumentX() probes ──────

function makeFakeCohere() {
  class V2Client {
    async chat(): Promise<any> { return {}; }
    async chatStream(): Promise<any> { return {}; }
    async embed(): Promise<any> { return {}; }
    async embedAsync(): Promise<any> { return {}; }
  }
  class CohereClientV2 {
    clientV2: any;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    constructor(_opts?: any) { this.clientV2 = new V2Client(); }
  }
  return { module: { CohereClientV2 }, V2Client };
}

function makeFakeOpenRouter() {
  class Chat {
    async send(): Promise<any> { return {}; }
  }
  class OpenRouter {
    chat: any;
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    constructor(_opts?: any) { this.chat = new Chat(); }
  }
  return { module: { OpenRouter }, Chat };
}

function makeFakeVoyage() {
  class VoyageAIClient {
    async embed(): Promise<any> { return {}; }
    async multimodalEmbed(): Promise<any> { return {}; }
  }
  return { module: { VoyageAIClient }, VoyageAIClient };
}

function makeFakeAnthropicBatches() {
  class Batches {
    results(): any { return {}; }
  }
  return { module: { Anthropic: { Batches } }, Batches };
}

function makeFakeLangChainChat() {
  class BaseChatModel {
    async generate(): Promise<any> { return {}; }
    async _streamIterator(): Promise<any> { return {}; }
  }
  return { module: { BaseChatModel }, BaseChatModel };
}

// ── Cohere v2 (probe → V2Client.prototype) ───────────────────────────────────

describe("uninstrument restores Cohere v2 V2Client prototype methods", () => {
  it("wraps then restores chat/chatStream/embed/embedAsync to the originals", () => {
    const { module, V2Client } = makeFakeCohere();
    const proto = V2Client.prototype as any;
    const originals = {
      chat: proto.chat,
      chatStream: proto.chatStream,
      embed: proto.embed,
      embedAsync: proto.embedAsync,
    };

    _instrumentCohere(module);
    for (const m of Object.keys(originals) as Array<keyof typeof originals>) {
      expect(proto[m]).not.toBe(originals[m]); // wrapped
    }

    restore();
    for (const m of Object.keys(originals) as Array<keyof typeof originals>) {
      expect(proto[m]).toBe(originals[m]); // restored
    }
  });
});

// ── OpenRouter (probe → Chat.prototype.send) ─────────────────────────────────

describe("uninstrument restores OpenRouter Chat.prototype.send", () => {
  it("wraps then restores send to the original", () => {
    const { module, Chat } = makeFakeOpenRouter();
    const proto = Chat.prototype as any;
    const orig = proto.send;

    _instrumentOpenRouter(module);
    expect(proto.send).not.toBe(orig);

    restore();
    expect(proto.send).toBe(orig);
  });
});

// ── Voyage (the double-entry / bogus-static bug) ─────────────────────────────

describe("uninstrument restores Voyage prototype methods and leaks no static", () => {
  it("restores embed/multimodalEmbed and writes NO static method on the class", () => {
    const { module, VoyageAIClient } = makeFakeVoyage();
    const proto = VoyageAIClient.prototype as any;
    const origEmbed = proto.embed;
    const origMulti = proto.multimodalEmbed;

    _instrumentVoyage(module);
    expect(proto.embed).not.toBe(origEmbed);
    expect(proto.multimodalEmbed).not.toBe(origMulti);

    restore();
    // prototype wrappers restored…
    expect(proto.embed).toBe(origEmbed);
    expect(proto.multimodalEmbed).toBe(origMulti);
    // …and NO bogus static method appears on the constructor (the old bug wrote
    // the original onto the class itself as a static).
    expect(Object.prototype.hasOwnProperty.call(VoyageAIClient, "embed")).toBe(false);
    expect(
      Object.prototype.hasOwnProperty.call(VoyageAIClient, "multimodalEmbed"),
    ).toBe(false);
  });
});

// ── Anthropic Batches (direct proto patch + idempotency flag) ────────────────

describe("uninstrument restores Anthropic Batches.results", () => {
  it("wraps + flags, then restores and clears __tpBatchesPatched", () => {
    const { module, Batches } = makeFakeAnthropicBatches();
    const proto = Batches.prototype as any;
    const orig = proto.results;

    _instrumentAnthropicBatches(module);
    expect(proto.results).not.toBe(orig);
    expect(proto.__tpBatchesPatched).toBe(true);

    restore();
    expect(proto.results).toBe(orig);
    expect(proto.__tpBatchesPatched).toBeUndefined();
  });

  it("re-instruments after uninstrument (flag was cleared)", () => {
    const { module, Batches } = makeFakeAnthropicBatches();
    const proto = Batches.prototype as any;
    const orig = proto.results;

    _instrumentAnthropicBatches(module);
    restore();
    expect(proto.results).toBe(orig);

    // Second pass must wrap again — only possible if __tpBatchesPatched cleared.
    _instrumentAnthropicBatches(module);
    expect(proto.results).not.toBe(orig);
  });
});

// ── LangChain chat models (BaseChatModel.prototype) ──────────────────────────

describe("uninstrument restores LangChain BaseChatModel prototype methods", () => {
  it("wraps then restores generate/_streamIterator", () => {
    const { module, BaseChatModel } = makeFakeLangChainChat();
    const proto = BaseChatModel.prototype as any;
    const origGenerate = proto.generate;
    const origStream = proto._streamIterator;

    _instrumentLangChainChatModels(module);
    expect(proto.generate).not.toBe(origGenerate);
    expect(proto._streamIterator).not.toBe(origStream);

    restore();
    expect(proto.generate).toBe(origGenerate);
    expect(proto._streamIterator).toBe(origStream);
  });
});

// ── Fault isolation: one throwing thunk must not block the others ────────────

describe("uninstrument isolates a throwing restore thunk", () => {
  it("restores real surfaces even when an earlier thunk throws", () => {
    const { module, V2Client } = makeFakeCohere();
    const proto = V2Client.prototype as any;
    const origChat = proto.chat;

    _instrumentCohere(module);
    expect(proto.chat).not.toBe(origChat);

    // Inject a poisoned thunk to run BEFORE the real ones.
    _restoreThunks.unshift(() => {
      throw new Error("boom: broken restore");
    });

    // Must not throw out of uninstrument, and Cohere must still restore.
    expect(() => restore()).not.toThrow();
    expect(proto.chat).toBe(origChat);
  });
});

// ── Re-instrument after uninstrument (Cohere) ────────────────────────────────

describe("re-instrument after uninstrument re-wraps", () => {
  it("Cohere V2Client.chat is wrapped again on a second instrument", () => {
    const { module, V2Client } = makeFakeCohere();
    const proto = V2Client.prototype as any;
    const orig = proto.chat;

    _instrumentCohere(module);
    restore();
    expect(proto.chat).toBe(orig);

    _instrumentCohere(module);
    expect(proto.chat).not.toBe(orig);
  });
});
