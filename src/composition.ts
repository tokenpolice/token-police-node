/**
 * Prompt & Response Composition Parser.
 *
 * Privacy-preserving decomposition of LLM payloads into structural metadata.
 * Extracts role, type, length, and hash of each message segment WITHOUT storing
 * the actual text content.
 *
 * 3-Tier Fallback Strategy:
 * 1. Supported provider (OpenAI, Anthropic, Google GenAI) — exact per-message parsing
 * 2. OpenAI-compatible format — try messages[] as best-effort
 * 3. Complete fallback — single entry with role="complete_prompt"
 */
import { createHash } from "node:crypto";

interface CompositionEntry {
  role: string;
  type: string;
  length?: number;
  hash?: string;
  name?: string;
}

// ── Helpers ──────────────────────────────────────────────────────

function fastHash(text: string): string {
  return createHash("sha1").update(text, "utf8").digest("hex").slice(0, 16);
}

/**
 * Count a string's length in Unicode code points (not UTF-16 code units), so it
 * matches the Python SDK's `len(str)` — a non-BMP (astral) char counts 1 on both
 * sides. This is the single code-point counter for the SDK: the composition
 * `length` field, the OTel-fallback composition path, and the tool-arg/result
 * `hashLen` helper all route through this helper for a single counting contract.
 *
 * Allocation-free: walks the string once with a surrogate-pair-aware index step
 * instead of spreading into an array (`[...s]`), which allocated a full array
 * per call just to count. Byte-identical result to `[...s].length` for all
 * inputs, including lone/unpaired surrogates (each counts 1). Total and no-throw
 * on any string input; non-string values are the caller's concern.
 */
export function codePointLength(s: string): number {
  let count = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    // A high surrogate immediately followed by a low surrogate forms one astral
    // code point — consume the low half so the pair counts once. A lone/unpaired
    // surrogate falls through and counts as 1 (same as the array spread).
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < s.length) {
      const next = s.charCodeAt(i + 1);
      if (next >= 0xdc00 && next <= 0xdfff) i++;
    }
    count++;
  }
  return count;
}

// ── Canonical JSON serializer ─────────────────────────────────
// One serializer routed through every object-valued tool-arg site so the
// hashed string (and therefore the composition `hash`) is byte-identical to
// the Python SDK's `_canonical_json` for the same logical value.
// `JSON.stringify` cannot be used here: it (a) reorders integer-like object
// keys numerically, (b) emits `1.0` as `1` but `1e-7` in exponential notation
// where CPython differs, and (c) would couple us to V8's escaping. So the
// structure, number format and string escapes are all hand-emitted to the
// rules pinned to the cross-SDK canonical-JSON contract (see shared/sdk-usage-parity-fixture.json), matching CPython
// `json.dumps(..., sort_keys=True, separators=(",",":"), ensure_ascii=False)`
// plus the integer-valued-float / non-finite / plain-decimal number rules.
//
// Pure and total — it never throws: it does no I/O, and any value handling that
// could raise (a throwing getter, a circular ref → RangeError) is caught by the
// top-level guard which degrades to a string form. Tool args reaching the
// emit sites are JSON that arrived over the wire, so the non-JSON-native
// fallback is only a safety net.

// PINNED escape table = exact intersection of JS `JSON.stringify` and CPython
// `json.dumps(ensure_ascii=False)`. `/`, U+2028 and U+2029 are NOT escaped;
// all non-ASCII (incl. astral) is emitted literally.
function encodeCanonicalString(s: string): string {
  let out = '"';
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i);
    if (c === 0x22) out += '\\"';
    else if (c === 0x5c) out += "\\\\";
    else if (c === 0x08) out += "\\b";
    else if (c === 0x09) out += "\\t";
    else if (c === 0x0a) out += "\\n";
    else if (c === 0x0c) out += "\\f";
    else if (c === 0x0d) out += "\\r";
    else if (c < 0x20) out += "\\u" + c.toString(16).padStart(4, "0");
    else out += s[i];
  }
  return out + '"';
}

// Re-emit a runtime shortest-round-trip decimal string (`String(n)`) as a
// canonical PLAIN decimal — never exponential, no trailing fraction zeros.
// Operates on the string (does not reimplement float→digits), so Node and
// Python converge because both share the same significant-digit sequence.
function expandPlainDecimal(r: string): string {
  let sign = "";
  if (r[0] === "-") {
    sign = "-";
    r = r.slice(1);
  }
  let mantissa = r;
  let exp = 0;
  const eIdx = r.search(/[eE]/);
  if (eIdx >= 0) {
    mantissa = r.slice(0, eIdx);
    exp = parseInt(r.slice(eIdx + 1), 10);
  }
  let intPart = mantissa;
  let frac = "";
  const dot = mantissa.indexOf(".");
  if (dot >= 0) {
    intPart = mantissa.slice(0, dot);
    frac = mantissa.slice(dot + 1);
  }
  const D = intPart + frac;
  const k = intPart.length + exp;
  let out: string;
  if (k <= 0) out = "0." + "0".repeat(-k) + D;
  else if (k >= D.length) out = D + "0".repeat(k - D.length);
  else out = D.slice(0, k) + "." + D.slice(k);
  if (out.indexOf(".") >= 0) {
    out = out.replace(/0+$/, "");
    if (out.endsWith(".")) out = out.slice(0, -1);
  }
  if (sign && /[1-9]/.test(out)) out = sign + out;
  return out;
}

function encodeCanonicalNumber(n: number): string {
  if (!Number.isFinite(n)) return "null"; // NaN / ±Infinity → null (matches JSON.stringify)
  if (Object.is(n, -0)) return "0";
  return expandPlainDecimal(String(n));
}

function encodeCanonical(value: unknown): string {
  if (value === null || value === undefined) return "null";
  const t = typeof value;
  if (t === "boolean") return value ? "true" : "false";
  if (t === "number") return encodeCanonicalNumber(value as number);
  if (t === "bigint") return (value as bigint).toString();
  if (t === "string") return encodeCanonicalString(value as string);
  if (Array.isArray(value)) {
    let out = "[";
    for (let i = 0; i < value.length; i++) {
      if (i > 0) out += ",";
      const el = value[i];
      const et = typeof el;
      // JSON array holes / functions / symbols / undefined → null.
      out += el === undefined || et === "function" || et === "symbol" ? "null" : encodeCanonical(el);
    }
    return out + "]";
  }
  if (t === "object") {
    const proto = Object.getPrototypeOf(value as object);
    // Only plain objects (and null-proto bags) are walked as JSON objects.
    // Dates / custom class instances fall to the language-local string form
    // (within-SDK deterministic, no-throw; not a cross-SDK contract).
    if (proto === Object.prototype || proto === null) {
      const obj = value as Record<string, unknown>;
      const keys = Object.keys(obj).sort();
      const parts: string[] = [];
      for (const key of keys) {
        const mv = obj[key];
        const mt = typeof mv;
        if (mv === undefined || mt === "function" || mt === "symbol") continue; // dropped, matching JSON.stringify
        parts.push(encodeCanonicalString(key) + ":" + encodeCanonical(mv));
      }
      return "{" + parts.join(",") + "}";
    }
    return encodeCanonicalString(String(value));
  }
  // function / symbol at top level → string fallback (never thrown).
  return encodeCanonicalString(String(value));
}

/**
 * Canonical JSON serializer — byte-identical to the Python SDK's
 * `_canonical_json` for JSON-native values. Pure, total
 * and no-throw: a hostile getter / circular reference is caught here and
 * degraded to a string form rather than raising. Exported so it can be reused
 * by a separate tool-span hashing path.
 */
export function canonicalJson(value: unknown): string {
  try {
    return encodeCanonical(value);
  } catch {
    try {
      return encodeCanonicalString(String(value));
    } catch {
      return '""';
    }
  }
}

/**
 * (sha1-16 hex, code-point length) for a tool arg / result value. Raw content is
 * never kept. THE single home for this pair — previously duplicated verbatim in
 * telemetry.ts (`hashLen`), frameworkTools.ts (`hashLen`) and context.ts
 * (`_toolHashLen`). String args are hashed byte-unchanged (preserves the OTel
 * hash baseline); non-string args route through the canonical JSON serializer
 * (total/no-throw) for cross-SDK byte-identity with the Python SDK.
 * Length is Unicode code points so an astral char counts 1, matching Python's
 * len(). Total and no-throw.
 */
export function hashLen(val: unknown): [string, number] {
  if (val === null || val === undefined) return ["", 0];
  const s = typeof val === "string" ? val : canonicalJson(val);
  if (!s) return ["", 0];
  return [createHash("sha1").update(s).digest("hex").slice(0, 16), codePointLength(s)];
}

export function textEntry(role: string, content: unknown, name?: string): CompositionEntry {
  // Length + hash are computed over the whitespace-trimmed content so the SAME
  // logical text fingerprints identically across calls even when a framework
  // reformats surrounding whitespace between turns (e.g. CrewAI rstrips an
  // assistant message before feeding it into the next call). Trimming is safe —
  // these are privacy-preserving fingerprints, never the stored text.
  //
  // This helper is total: a malformed (non-string) text part must degrade to an
  // empty-text entry, never abort the whole composition — otherwise one bad part
  // would collapse the entire call down to a single coarse fallback entry,
  // silently losing per-message granularity. null/undefined → empty text; any
  // other non-string is routed through the canonical serializer so Node and
  // Python fingerprint the same JSON-shaped value identically (a bare String()
  // would diverge on numbers/bools/objects), with an empty-text fallback if
  // that fails. Covered by "textEntry never raises on malformed content",
  // "null text part keeps per-message composition" and "non-string text parity".
  let text: string;
  if (typeof content === "string") {
    text = content;
  } else if (content === null || content === undefined) {
    text = "";
  } else {
    try {
      text = canonicalJson(content);
    } catch {
      text = "";
    }
  }
  const normalized = text.trim();
  // Length is counted in Unicode CODE POINTS (not UTF-16 code units) so it
  // matches the Python SDK's `len(normalized)` for the same string — a non-BMP
  // (astral) char counts 1 on both sides.
  const entry: CompositionEntry = {
    role,
    type: "text",
    length: codePointLength(normalized),
    hash: fastHash(normalized),
  };
  if (name) entry.name = name;
  return entry;
}

function nonTextEntry(role: string, mediaType: string, name?: string): CompositionEntry {
  const entry: CompositionEntry = { role, type: mediaType };
  if (name) entry.name = name;
  return entry;
}

/**
 * Embedding input parser. Embeddings have only input (no role/turn
 * structure) — emit role="input" entries. Accepts:
 * * string → 1 text entry
 * * list of strings → N text entries
 * * list of ints (OpenAI pre-tokenized) → 1 text entry, length only
 * * Cohere multimodal / Mistral chunks → typed entries (text / image)
 *
 * Provider-specific kwarg keys: `input` (OpenAI/Together), `texts`
 * (Cohere/Voyage), `inputs` (Mistral/Cohere multimodal/HF Node SDK),
 * `contents` (Google), `text` (HF text-only).
 */
function parseEmbeddingInput(kwargs: Record<string, any>): CompositionEntry[] {
  if (!kwargs || typeof kwargs !== "object") return [];
  const payload =
    kwargs.input ?? kwargs.texts ?? kwargs.inputs ?? kwargs.contents ?? kwargs.text;
  if (payload == null) return [];

  if (typeof payload === "string") {
    return [textEntry("input", payload)];
  }
  if (Array.isArray(payload)) {
    if (payload.length > 0 && payload.every((t) => typeof t === "number" && Number.isInteger(t))) {
      return [{ role: "input", type: "text", length: payload.length }];
    }
    const out: CompositionEntry[] = [];
    for (const item of payload) {
      if (typeof item === "string") {
        out.push(textEntry("input", item));
      } else if (
        Array.isArray(item) &&
        item.length > 0 &&
        item.every((t) => typeof t === "number" && Number.isInteger(t))
      ) {
        // Pre-tokenized element (batch of int-arrays). Checked BEFORE the
        // generic object branch because `typeof [] === "object"`, so an array
        // would otherwise be swallowed there and never reach this case.
        out.push({ role: "input", type: "text", length: item.length });
      } else if (item && typeof item === "object") {
        const t = String(item.type ?? "").toLowerCase();
        if (t === "image" || t === "image_url") {
          out.push(nonTextEntry("input", "image"));
        } else if (t === "text") {
          const text = typeof item.text === "string" ? item.text : String(item.text ?? "");
          out.push(textEntry("input", text));
        } else {
          out.push({ role: "input", type: t || "unknown" });
        }
      } else {
        out.push({ role: "input", type: "unknown" });
      }
    }
    return out;
  }
  return [{ role: "input", type: "unknown" }];
}

/**
 * Best-effort prompt composition for non-text (image/audio/video) generation.
 * Matches when the kwargs look like a non-chat surface (no messages/contents,
 * but a top-level prompt / input / file). Returns null so the regular chat
 * parsers stay in charge for text calls.
 */
function tryModalityPrompt(kwargs: any): CompositionEntry[] | null {
  if (!kwargs || typeof kwargs !== "object") return null;
  if (
    Array.isArray(kwargs.messages) ||
    Array.isArray(kwargs.contents) ||
    Array.isArray(kwargs.input)
  ) {
    return null;
  }
  const out: CompositionEntry[] = [];
  if (typeof kwargs.prompt === "string" && kwargs.prompt) {
    out.push(textEntry("user", kwargs.prompt));
  } else if (typeof kwargs.input === "string" && kwargs.input) {
    out.push(textEntry("user", kwargs.input));
  } else if (typeof kwargs.text === "string" && kwargs.text) {
    out.push(textEntry("user", kwargs.text));
  }
  const file = kwargs.file ?? kwargs.audio;
  if (file != null) {
    let name: string | undefined;
    if (typeof file === "string") name = file;
    else if (file?.name && typeof file.name === "string") name = file.name;
    out.push(nonTextEntry("user", "audio", name));
  }
  return out.length > 0 ? out : null;
}

// MIME-type prefixes we treat as "this response is a binary audio body, not
// a text body". Matches OpenAI TTS (`audio/mpeg` / `application/octet-stream`),
// HF TTS, Mistral Voxtral, Together audio, etc.
const AUDIO_MIME_PREFIXES = ["audio/", "application/octet-stream"] as const;

/**
 * Positive identification of a binary audio response across SDKs.
 *
 * The OpenAI Node SDK returns the raw fetch `Response` for `audio.speech.create`
 * (`__binaryResponse: true`). That object has `.body` (ReadableStream),
 * `.arrayBuffer()`, `.blob()` and crucially `.text()` (as a FUNCTION) — meaning
 * `typeof response.text === "string"` is false, but a future SDK or wrapper
 * that exposes a `.text` *property* could accidentally feed the audio bytes
 * into our text parser. This helper short-circuits those bodies into an
 * `assistant/audio` entry before any text branch runs.
 *
 * Every accessor is wrapped in try/catch so the helper never throws.
 */
function looksLikeBinaryAudio(response: any): boolean {
  if (response == null) return false;
  // Raw binary payloads.
  try {
    if (response instanceof ArrayBuffer) return true;
  } catch {
    /* ignore */
  }
  try {
    if (typeof Buffer !== "undefined" && Buffer.isBuffer && Buffer.isBuffer(response)) {
      return true;
    }
  } catch {
    /* ignore */
  }
  try {
    if (ArrayBuffer.isView && ArrayBuffer.isView(response)) return true;
  } catch {
    /* ignore */
  }
  // Web Fetch `Response` — what OpenAI Node TTS hands back.
  try {
    if (typeof Response !== "undefined" && response instanceof Response) {
      // The content-type header is authoritative for a real fetch Response.
      let ct: string | undefined;
      let headerReadable = true;
      try {
        const raw = response.headers?.get?.("content-type");
        ct = typeof raw === "string" ? raw.toLowerCase().split(";")[0].trim() : undefined;
      } catch {
        // Exotic Response whose headers getter throws — treat as unreadable.
        headerReadable = false;
      }
      if (headerReadable && ct) {
        // Known content-type: audio (or generic binary octet-stream) → audio;
        // anything else (text/*, application/json, …) is a normal parseable
        // body, NOT audio — return false so an obviously-non-audio Response is
        // never mislabeled.
        return AUDIO_MIME_PREFIXES.some((p) => ct!.startsWith(p));
      }
      // Missing or unreadable content-type → assume the OpenAI TTS raw-fetch
      // case (the reason we special-case Response at all) and treat as audio.
      return true;
    }
  } catch {
    /* ignore */
  }
  // Class-name hints for vendor wrappers we don't (yet) import.
  let ctorName = "";
  try {
    ctorName = response.constructor?.name ?? "";
  } catch {
    /* ignore */
  }
  if (
    ctorName === "BinaryResponseContent" ||
    ctorName === "HttpxBinaryResponseContent" ||
    ctorName === "BinaryAPIResponse" ||
    ctorName === "StreamedBinaryAPIResponse"
  ) {
    return true;
  }
  // Headers + audio MIME (works for any thin wrapper around fetch Response).
  try {
    const headers = response.headers;
    if (headers != null) {
      let ct: any;
      try {
        ct = typeof headers.get === "function" ? headers.get("content-type") : headers["content-type"];
      } catch {
        ct = undefined;
      }
      if (typeof ct === "string") {
        const ctLower = ct.toLowerCase().split(";")[0].trim();
        if (AUDIO_MIME_PREFIXES.some((p) => ctLower.startsWith(p))) return true;
      }
    }
  } catch {
    /* ignore */
  }
  // Bytes-iterator duck-type: a `body` ReadableStream + an `arrayBuffer` /
  // `blob` reader, OR a Node `Readable` exposing `pipe` and bytes accessors.
  try {
    const hasReadableBody =
      response.body != null &&
      (typeof response.body.getReader === "function" || typeof response.body.pipe === "function");
    const hasBinaryReader =
      typeof response.arrayBuffer === "function" ||
      typeof response.blob === "function" ||
      typeof response.read === "function";
    if (hasReadableBody && hasBinaryReader) return true;
  } catch {
    /* ignore */
  }
  // Substring fall-back for class names we haven't catalogued.
  if (typeof ctorName === "string" && (ctorName.includes("Binary") || ctorName.includes("HttpxBinary"))) {
    return true;
  }
  return false;
}

/**
 * Best-effort response composition for non-text generation outputs (DALL-E,
 * Imagen, FLUX, Veo, Whisper, TTS).
 *
 * Order matters: the binary-audio check runs FIRST so it cannot be bypassed by
 * an attribute access on a vendor wrapper. Without this, an SDK that exposes
 * `.text` as a property on a binary body (the original Python
 * `HttpxBinaryResponseContent.text` issue) would land in the `.text`-as-
 * transcription branch and produce a fake `length:<byte_count>` text entry.
 */
function tryModalityResponse(response: any): CompositionEntry[] | null {
  if (response == null) return null;
  // 1. Binary audio FIRST, isolated from the rest of the try block so an
  // unrelated property access can't skip the check.
  try {
    if (looksLikeBinaryAudio(response)) {
      return [nonTextEntry("assistant", "audio")];
    }
  } catch {
    /* ignore */
  }
  try {
    if (Array.isArray(response.data) && response.data.length > 0) {
      const first = response.data[0];
      if (first && (first.url || first.b64_json)) {
        return response.data.map(() => nonTextEntry("assistant", "image"));
      }
    }
    if (Array.isArray(response.generatedImages) && response.generatedImages.length > 0) {
      return response.generatedImages.map(() => nonTextEntry("assistant", "image"));
    }
    if (Array.isArray(response.images) && response.images.length > 0 && typeof response.images[0] === "object") {
      return response.images.map(() => nonTextEntry("assistant", "image"));
    }
    if (Array.isArray(response.generatedVideos) && response.generatedVideos.length > 0) {
      return response.generatedVideos.map(() => nonTextEntry("assistant", "video"));
    }
    if (Array.isArray(response.videos) && response.videos.length > 0) {
      return response.videos.map(() => nonTextEntry("assistant", "video"));
    }
    // Transcription-style text body (Whisper, HF, Mistral). Only treat `.text`
    // as text if the response is NOT also bytes-iterator-shaped (defensive: any
    // body that exposes both is binary — the `.text` is just a decoded view).
    // NOT a transcription when the object is a structured chat response that
    // merely exposes a `.text` convenience getter — @google/genai's
    // GenerateContentResponse does exactly this, and matching it here
    // swallowed the functionCall parts (the response composed as a single
    // text entry while a tool call followed). Whisper-style transcription
    // bodies carry none of these container fields. Checked BEFORE touching
    // `.text`: the getter logs a console warning on multi-part responses.
    const isStructuredChat =
      Array.isArray(response.candidates) ||
      Array.isArray(response.choices) ||
      Array.isArray(response.content) ||
      Array.isArray(response.output) ||
      response.message != null;
    if (!isStructuredChat && typeof response.text === "string" && response.text) {
      const looksBinary =
        (response.body != null &&
          (typeof response.body.getReader === "function" ||
            typeof response.body.pipe === "function")) ||
        typeof response.arrayBuffer === "function" ||
        typeof response.blob === "function";
      if (looksBinary) {
        return [nonTextEntry("assistant", "audio")];
      }
      return [textEntry("assistant", response.text)];
    }
  } catch {
    return null;
  }
  return null;
}

/**
 * OCR response composition (Mistral OCR and any future `mistral_ocr`-shaped
 * body). Privacy-preserving structural markers only — the per-page text is
 * NEVER read. One `{role:"assistant", type:"ocr_page"}` per billed page
 * (capped at 100), or a single `{role:"assistant", type:"ocr_document"}`
 * when the page count is unknown/zero (still records that an OCR happened).
 *
 * Page count comes from `usage_info.pages_processed` (the value the SDK bills
 * as `ocr_pages`), falling back to `pages.length` only when `pages` is a real
 * array. Total/no-throw: any error returns the single-document fallback.
 */
function ocrComposition(response: any): CompositionEntry[] {
  const DOCUMENT_FALLBACK: CompositionEntry[] = [nonTextEntry("assistant", "ocr_document")];
  try {
    if (response == null) return DOCUMENT_FALLBACK;

    // usage_info: dict-or-object safe.
    let usageInfo: any = undefined;
    if (typeof response === "object") {
      usageInfo = response.usage_info;
    }

    let n = 0;
    // Primary source: usage_info.pages_processed → int.
    const rawPages = usageInfo != null && typeof usageInfo === "object"
      ? usageInfo.pages_processed
      : undefined;
    // String coercion mirrors Python's int(): only a pure integer string is
    // accepted ("3" → 3), NOT a leading-integer prefix ("3.5"/"3abc" → reject),
    // so Node and Python emit byte-identical composition for the same input.
    const parsed =
      typeof rawPages === "number"
        ? Math.trunc(rawPages)
        : typeof rawPages === "string" && /^\s*[+-]?\d+\s*$/.test(rawPages)
          ? parseInt(rawPages, 10)
          : NaN;
    if (Number.isFinite(parsed) && parsed > 0) {
      n = parsed;
    } else {
      // Fallback: length of the `pages` array (only if it really is one).
      const pages = response.pages;
      if (Array.isArray(pages)) {
        n = pages.length;
      }
    }

    if (n > 0) {
      const count = Math.min(n, 100);
      const entries: CompositionEntry[] = [];
      for (let i = 0; i < count; i++) {
        entries.push(nonTextEntry("assistant", "ocr_page"));
      }
      return entries;
    }
    return DOCUMENT_FALLBACK;
  } catch {
    return DOCUMENT_FALLBACK;
  }
}

// ═══════════════════════════════════════════════════════════════════
// Tier 1: Provider-specific parsers
// ═══════════════════════════════════════════════════════════════════

/**
 * Parse a Vercel AI SDK V2 `LanguageModelV2Prompt` — what `doGenerate` /
 * `doStream` receive as `args[0].prompt`. Items:
 * { role: "system", content: string }
 * { role: "user", content: Array<TextPart | FilePart> }
 * { role: "assistant", content: Array<TextPart | ToolCallPart | ReasoningPart | FilePart> }
 * { role: "tool", content: Array<ToolResultPart> }
 *
 * Parts:
 * { type: "text", text }
 * { type: "file", data, mediaType } → non_text "file" / "image"
 * { type: "tool-call", toolCallId, toolName, input } → tool_call
 * { type: "tool-result", toolCallId, toolName, output } → tool_result
 * { type: "reasoning", text } → text entry
 */
function parseXaiAiSdkPrompt(prompt: any[]): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  for (const msg of prompt ?? []) {
    if (msg == null || typeof msg !== "object") continue;
    const role: string = msg.role ?? "user";
    const content = msg.content;
    if (typeof content === "string") {
      if (content) result.push(textEntry(role, content));
      continue;
    }
    if (!Array.isArray(content)) continue;
    for (const part of content) {
      if (part == null || typeof part !== "object") {
        if (typeof part === "string" && part) result.push(textEntry(role, part));
        continue;
      }
      const pt = part.type ?? "text";
      if (pt === "text") {
        const t = part.text ?? "";
        if (typeof t === "string" && t) result.push(textEntry(role, t));
      } else if (pt === "reasoning") {
        const t = part.text ?? "";
        if (typeof t === "string" && t) result.push(textEntry(role, t));
      } else if (pt === "file") {
        const mime = String(part.mediaType ?? part.mimeType ?? "");
        const kind = mime.startsWith("image/")
          ? "image"
          : mime.startsWith("audio/")
            ? "audio"
            : "file";
        result.push(nonTextEntry(role, kind));
      } else if (pt === "tool-call") {
        let args = "";
        try {
          args =
            typeof part.input === "string"
              ? part.input
              : canonicalJson(part.input ?? "");
        } catch {
          args = String(part.input ?? "");
        }
        result.push({
          ...textEntry("tool_call", args, part.toolName ?? ""),
          type: "tool_call",
        });
      } else if (pt === "tool-result") {
        let output = "";
        try {
          const raw = part.output;
          if (raw == null) output = "";
          else if (typeof raw === "string") output = raw;
          else if (typeof raw === "object" && typeof raw.value === "string") output = raw.value;
          else output = canonicalJson(raw);
        } catch {
          output = String(part.output ?? "");
        }
        result.push(textEntry("tool_result", output, part.toolName ?? ""));
      }
    }
  }
  return result;
}

/**
 * Parse an AI SDK V2 `doGenerate` response (or accumulated synthetic from
 * doStream). Shape: { content: Array<{ type: "text" | "reasoning" |
 * "tool-call" | "file" | "source", ... }>, usage, finishReason }.
 */
function parseXaiAiSdkResponse(response: any): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  const content = response?.content;
  if (!Array.isArray(content)) return result;
  for (const part of content) {
    if (part == null || typeof part !== "object") continue;
    const pt = part.type ?? "text";
    if (pt === "text") {
      const t = part.text ?? "";
      if (typeof t === "string" && t) result.push(textEntry("assistant", t));
    } else if (pt === "reasoning") {
      const t = part.text ?? "";
      if (typeof t === "string" && t) result.push(textEntry("assistant", t));
    } else if (pt === "tool-call") {
      let args = "";
      try {
        args =
          typeof part.input === "string"
            ? part.input
            : canonicalJson(part.input ?? "");
      } catch {
        args = String(part.input ?? "");
      }
      result.push({
        ...textEntry("tool_call", args, part.toolName ?? ""),
        type: "tool_call",
      });
    }
  }
  return result;
}

function parseOpenAIMessages(messages: any[]): CompositionEntry[] {
  const result: CompositionEntry[] = [];

  for (const msg of messages) {
    // Skip null/undefined/non-object array elements — dereferencing `msg.role`
    // on one would throw and collapse the whole composition to a coarse
    // fallback, losing per-message granularity for the valid messages. Matches
    // the Python parser, which likewise produces no entry for such elements.
    if (msg == null || typeof msg !== "object") continue;
    const role: string = msg.role ?? "unknown";
    const content = msg.content;

    if (typeof content === "string" && role !== "tool") {
      result.push(textEntry(role, content));
    } else if (Array.isArray(content) && role !== "tool") {
      for (const part of content) {
        if (typeof part === "object" && part !== null) {
          const partType = part.type ?? "text";
          if (partType === "text") {
            result.push(textEntry(role, part.text ?? ""));
          } else if (partType === "image_url") {
            result.push(nonTextEntry(role, "image"));
          } else if (partType === "input_audio") {
            result.push(nonTextEntry(role, "audio"));
          } else {
            result.push(nonTextEntry(role, partType));
          }
        } else if (typeof part === "string") {
          result.push(textEntry(role, part));
        }
      }
    }

    // Cohere v2 assistant messages carry a `tool_plan` — the model's
    // chain-of-thought reflection before it emits tool calls. OpenAI-shaped
    // SDKs never set this field, so this is a no-op for them.
    const toolPlan = msg.tool_plan ?? msg.toolPlan;
    if (typeof toolPlan === "string" && toolPlan) {
      result.push(textEntry(role, toolPlan));
    }

    // Tool calls in assistant messages. The OpenAI SDK uses `tool_calls`;
    // the native OpenRouter SDK uses camelCase `toolCalls`.
    const toolCalls = msg.tool_calls ?? msg.toolCalls;
    if (Array.isArray(toolCalls)) {
      for (const tc of toolCalls) {
        const fn = tc.function ?? {};
        const fnName = fn.name ?? "";
        const fnArgs = fn.arguments ?? "";
        result.push({ ...textEntry("tool_call", fnArgs, fnName), type: "tool_call" });
      }
    }

    // Tool result message
    if (role === "tool") {
      const toolContent = typeof content === "string" ? content : canonicalJson(content ?? "");
      result.push(textEntry("tool_result", toolContent, msg.name ?? ""));
    }
  }

  return result;
}

/**
 * Parse OpenAI Responses API `input` (+ top-level `instructions`).
 *
 * The Responses input is either a single string (treated as a user message)
 * or a list of items. Each item is one of:
 * - { role, content } — content is string or list of
 * { type: "input_text"|"output_text", text }
 * (also "input_image" / "input_audio").
 * - { type: "function_call", name, arguments, call_id } — assistant tool call.
 * - { type: "function_call_output", call_id, output } — tool result; name
 * resolved from the matching prior
 * function_call item.
 * - { type: "reasoning", summary } — emitted as a non-text marker.
 */
function parseOpenAIResponsesInput(
  inputPayload: any,
  instructions?: any,
): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  // call_id -> tool name
  const toolNames: Record<string, string> = {};

  if (instructions) {
    if (typeof instructions === "string") {
      result.push(textEntry("system", instructions));
    } else if (Array.isArray(instructions)) {
      for (const block of instructions) {
        if (typeof block === "object" && block != null) {
          const text = block.text ?? block.content;
          if (typeof text === "string") {
            result.push(textEntry("system", text));
          }
        }
      }
    }
  }

  if (inputPayload == null) return result;

  if (typeof inputPayload === "string") {
    if (inputPayload) result.push(textEntry("user", inputPayload));
    return result;
  }

  if (!Array.isArray(inputPayload)) return result;

  for (const rawItem of inputPayload) {
    if (typeof rawItem === "string") {
      if (rawItem) result.push(textEntry("user", rawItem));
      continue;
    }
    if (typeof rawItem !== "object" || rawItem == null) continue;
    const item: any = rawItem;

    const itype = item.type;
    if (itype === "function_call") {
      const name = item.name ?? "";
      const callId = item.call_id ?? item.callId ?? "";
      if (callId) toolNames[callId] = name;
      result.push({
        ...textEntry("tool_call", item.arguments ?? "", name),
        type: "tool_call",
      });
      continue;
    }
    if (itype === "function_call_output") {
      const callId = item.call_id ?? item.callId ?? "";
      const name = toolNames[callId] ?? callId;
      let output = item.output ?? "";
      if (typeof output !== "string") {
        try {
          output = canonicalJson(output);
        } catch {
          output = String(output);
        }
      }
      result.push(textEntry("tool_result", output, name));
      continue;
    }
    if (itype === "reasoning") {
      result.push(nonTextEntry("assistant", "reasoning"));
      continue;
    }

    const role: string = item.role ?? itype ?? "user";
    const content = item.content;
    if (typeof content === "string") {
      if (content) result.push(textEntry(role, content));
    } else if (Array.isArray(content)) {
      for (const part of content) {
        if (typeof part === "object" && part != null) {
          const ptype = part.type ?? "text";
          if (
            ptype === "input_text" ||
            ptype === "output_text" ||
            ptype === "text" ||
            ptype === "summary_text"
          ) {
            const text = part.text ?? "";
            if (text) result.push(textEntry(role, text));
          } else if (
            ptype === "input_image" ||
            ptype === "image_url" ||
            ptype === "image"
          ) {
            result.push(nonTextEntry(role, "image"));
          } else if (ptype === "input_audio" || ptype === "audio") {
            result.push(nonTextEntry(role, "audio"));
          } else {
            result.push(nonTextEntry(role, ptype));
          }
        } else if (typeof part === "string") {
          if (part) result.push(textEntry(role, part));
        }
      }
    }
  }

  return result;
}

/**
 * Parse an OpenAI Responses API response (or accumulated synthetic dict).
 *
 * Walks `response.output` — a list of items:
 * - "message" → content array of { type: "output_text", text } parts
 * - "function_call" → { name, arguments, call_id }
 * - "reasoning" → emitted as a non-text marker
 */
function parseOpenAIResponsesResponse(response: any): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  const output = response?.output;
  if (!Array.isArray(output)) return result;

  for (const rawItem of output) {
    if (typeof rawItem !== "object" || rawItem == null) continue;
    const item: any = rawItem;
    const itype = item.type;
    if (itype === "message") {
      for (const part of item.content ?? []) {
        if (typeof part !== "object" || part == null) continue;
        const ptype = part.type ?? "";
        if (ptype === "output_text" || ptype === "text") {
          const text = part.text ?? "";
          if (text) result.push(textEntry("assistant", text));
        } else if (ptype === "refusal") {
          const refusal = part.refusal ?? "";
          if (refusal) result.push(textEntry("assistant", refusal));
        } else {
          result.push(nonTextEntry("assistant", ptype || "unknown"));
        }
      }
    } else if (itype === "function_call") {
      let args = item.arguments ?? "";
      if (typeof args !== "string") {
        try {
          args = canonicalJson(args);
        } catch {
          args = String(args);
        }
      }
      result.push({
        ...textEntry("tool_call", args, item.name ?? ""),
        type: "tool_call",
      });
    } else if (itype === "reasoning") {
      result.push(nonTextEntry("assistant", "reasoning"));
    } else if (itype === "image_generation_call") {
      // Built-in image_generation tool output — a real billed image. Without
      // this entry the generated image is invisible in the composition (only
      // the surrounding assistant text shows).
      result.push(nonTextEntry("assistant", "image"));
    }
  }
  return result;
}

function parseAnthropicMessages(
  messages: any[],
  system?: any,
): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  // tool_use_id -> tool name. Anthropic tool_result blocks only carry the
  // tool_use_id, so we resolve the human-readable name from the matching
  // tool_use block (which always precedes its result in the messages array).
  const toolNames: Record<string, string> = {};

  // System prompt (top-level parameter in Anthropic)
  if (system) {
    if (typeof system === "string") {
      result.push(textEntry("system", system));
    } else if (Array.isArray(system)) {
      for (const block of system) {
        if (typeof block === "object" && block?.type === "text") {
          result.push(textEntry("system", block.text ?? ""));
        }
      }
    }
  }

  for (const msg of messages) {
    // Skip null/undefined/non-object array elements — dereferencing `msg.role`
    // on one would throw and collapse the whole composition to a coarse
    // fallback, losing per-message granularity for the valid messages. Matches
    // the Python parser, which likewise produces no entry for such elements.
    if (msg == null || typeof msg !== "object") continue;
    const role: string = msg.role ?? "unknown";
    const content = msg.content;

    if (typeof content === "string") {
      result.push(textEntry(role, content));
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (typeof block === "object" && block !== null) {
          const blockType = block.type ?? "text";
          if (blockType === "text") {
            result.push(textEntry(role, block.text ?? ""));
          } else if (blockType === "image") {
            result.push(nonTextEntry(role, "image"));
          } else if (blockType === "tool_use") {
            const toolName = block.name ?? "";
            if (block.id) toolNames[block.id] = toolName;
            const inp = canonicalJson(block.input ?? {});
            result.push({ ...textEntry("tool_call", inp, toolName), type: "tool_call" });
          } else if (blockType === "tool_result") {
            const resolvedName =
              toolNames[block.tool_use_id] ?? block.tool_use_id ?? "";
            const trContent = block.content;
            if (Array.isArray(trContent)) {
              for (const sub of trContent) {
                if (typeof sub === "object" && sub?.type === "text") {
                  result.push(textEntry("tool_result", sub.text ?? "", resolvedName));
                }
              }
            } else if (typeof trContent === "string") {
              result.push(textEntry("tool_result", trContent, resolvedName));
            }
          } else {
            result.push(nonTextEntry(role, blockType));
          }
        } else if (typeof block === "string") {
          result.push(textEntry(role, block));
        }
      }
    }
  }

  return result;
}

function parseGoogleContents(
  contents: any,
  systemInstruction?: any,
): CompositionEntry[] {
  const result: CompositionEntry[] = [];

  // System instruction
  if (systemInstruction) {
    if (typeof systemInstruction === "string") {
      result.push(textEntry("system", systemInstruction));
    } else if (typeof systemInstruction === "object") {
      const parts = systemInstruction.parts ?? [];
      for (const p of parts) {
        if (typeof p === "object" && "text" in p) {
          result.push(textEntry("system", p.text));
        } else if (typeof p === "string") {
          result.push(textEntry("system", p));
        }
      }
    }
  }

  if (typeof contents === "string") {
    result.push(textEntry("user", contents));
    return result;
  }

  if (Array.isArray(contents)) {
    for (const content of contents) {
      if (typeof content === "string") {
        result.push(textEntry("user", content));
        continue;
      }
      if (typeof content !== "object" || content === null) continue;

      const role: string = content.role ?? "user";
      const parts = content.parts ?? [];
      for (const part of parts) {
        if (typeof part === "string") {
          result.push(textEntry(role, part));
        } else if (typeof part === "object" && part !== null) {
          // @google/genai uses camelCase keys; the legacy SDK used snake_case.
          const inlineData = part.inlineData ?? part.inline_data;
          const functionCall = part.functionCall ?? part.function_call;
          const functionResponse =
            part.functionResponse ?? part.function_response;
          if (part.text != null) {
            result.push(textEntry(role, part.text));
          } else if (inlineData) {
            const mime: string =
              inlineData.mimeType ?? inlineData.mime_type ?? "unknown";
            const media = mime.includes("image") ? "image" : mime.includes("audio") ? "audio" : mime;
            result.push(nonTextEntry(role, media));
          } else if (functionCall) {
            const args = canonicalJson(functionCall.args ?? {});
            result.push({ ...textEntry("tool_call", args, functionCall.name ?? ""), type: "tool_call" });
          } else if (functionResponse) {
            const resp = canonicalJson(functionResponse.response ?? {});
            result.push(textEntry("tool_result", resp, functionResponse.name ?? ""));
          }
        }
      }
    }
  }

  return result;
}

function parseBedrockConverse(
  messages: any[],
  system?: any,
): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  // toolUseId -> tool name, resolved from the toolUse block that precedes the
  // result (Converse toolResult blocks only carry the toolUseId).
  const toolNames: Record<string, string> = {};

  // Converse system prompt is a top-level list of content blocks.
  if (system) {
    if (typeof system === "string") {
      result.push(textEntry("system", system));
    } else if (Array.isArray(system)) {
      for (const block of system) {
        if (typeof block === "object" && block !== null && "text" in block) {
          result.push(textEntry("system", block.text ?? ""));
        }
      }
    }
  }

  for (const msg of messages) {
    const role: string = msg?.role ?? "unknown";
    const content = msg?.content;

    if (typeof content === "string") {
      result.push(textEntry(role, content));
    } else if (Array.isArray(content)) {
      for (const block of content) {
        if (typeof block === "string") {
          result.push(textEntry(role, block));
          continue;
        }
        if (typeof block !== "object" || block === null) continue;
        if ("text" in block) {
          result.push(textEntry(role, block.text ?? ""));
        } else if ("image" in block) {
          result.push(nonTextEntry(role, "image"));
        } else if ("document" in block) {
          result.push(nonTextEntry(role, "document"));
        } else if ("video" in block) {
          result.push(nonTextEntry(role, "video"));
        } else if ("toolUse" in block) {
          const tu = block.toolUse ?? {};
          const toolName = tu.name ?? "";
          if (tu.toolUseId) toolNames[tu.toolUseId] = toolName;
          const inp = canonicalJson(tu.input ?? {});
          result.push({ ...textEntry("tool_call", inp, toolName), type: "tool_call" });
        } else if ("toolResult" in block) {
          const tr = block.toolResult ?? {};
          const resolvedName = toolNames[tr.toolUseId] ?? tr.toolUseId ?? "";
          const trContent = tr.content;
          if (Array.isArray(trContent)) {
            for (const sub of trContent) {
              if (typeof sub !== "object" || sub === null) continue;
              if ("text" in sub) {
                result.push(textEntry("tool_result", sub.text ?? "", resolvedName));
              } else if ("json" in sub) {
                result.push(textEntry("tool_result", canonicalJson(sub.json ?? {}), resolvedName));
              } else {
                result.push(nonTextEntry("tool_result", "content", resolvedName));
              }
            }
          } else if (typeof trContent === "string") {
            result.push(textEntry("tool_result", trContent, resolvedName));
          }
        } else {
          const keys = Object.keys(block);
          result.push(nonTextEntry(role, keys[0] ?? "unknown"));
        }
      }
    }
  }

  return result;
}

function parseCohereResponse(response: any): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  const message = response?.message;
  if (!message || typeof message !== "object") return result;

  // tool_plan: chain-of-thought reflection the model emits before tool calls.
  const toolPlan = message.toolPlan ?? message.tool_plan;
  if (typeof toolPlan === "string" && toolPlan) {
    result.push(textEntry("assistant", toolPlan));
  }

  const content = message.content;
  if (typeof content === "string") {
    result.push(textEntry("assistant", content));
  } else if (Array.isArray(content)) {
    for (const block of content) {
      if (typeof block === "string") {
        result.push(textEntry("assistant", block));
      } else if (typeof block === "object" && block !== null) {
        const blockType = block.type ?? "text";
        if (blockType === "text") {
          result.push(textEntry("assistant", block.text ?? ""));
        } else {
          result.push(nonTextEntry("assistant", blockType));
        }
      }
    }
  }

  // cohere-ai is Fern-generated (camelCase); tolerate snake_case too.
  const toolCalls = message.toolCalls ?? message.tool_calls;
  if (Array.isArray(toolCalls)) {
    for (const tc of toolCalls) {
      const fn = tc?.function ?? {};
      const fnArgs =
        typeof fn.arguments === "string"
          ? fn.arguments
          : canonicalJson(fn.arguments ?? {});
      result.push({ ...textEntry("tool_call", fnArgs, fn.name ?? ""), type: "tool_call" });
    }
  }

  return result;
}

// ── LangChain ──────────────────────────────────────────────────────
// LangChain calls flow through BaseChatModel.{generate,stream,...}. The prompt
// is a payload of LangChain BaseMessage objects (or a string for stream); the
// response is an LLMResult whose generations carry AIMessage objects.

const LANGCHAIN_ROLE_MAP: Record<string, string> = {
  human: "user",
  ai: "assistant",
  system: "system",
  tool: "tool_result",
  function: "tool_result",
  // Streaming chunks. LangChain's *MessageChunk classes override `_getType()`
  // with the class name (e.g. AIMessageChunk → "AIMessageChunk") instead of
  // inheriting the lowercase parent type — without these we'd surface
  // "AIMessageChunk" as the role in agent / streaming flows that fold chunks
  // back into the message history.
  AIMessageChunk: "assistant",
  HumanMessageChunk: "user",
  SystemMessageChunk: "system",
  ToolMessageChunk: "tool_result",
  FunctionMessageChunk: "tool_result",
};

/** Resolve a LangChain BaseMessage's type ("human" / "ai" / "system" / ...). */
function langchainMessageType(msg: any): string {
  if (typeof msg?._getType === "function") {
    try {
      return String(msg._getType());
    } catch {
      /* fall through */
    }
  }
  if (typeof msg?.getType === "function") {
    try {
      return String(msg.getType());
    } catch {
      /* fall through */
    }
  }
  // Fallback to the constructor name (HumanMessage -> human, AIMessage -> ai...).
  const cls = String(msg?.constructor?.name ?? "");
  if (cls.startsWith("Human")) return "human";
  if (cls.startsWith("AI")) return "ai";
  if (cls.startsWith("System")) return "system";
  if (cls.startsWith("Tool")) return "tool";
  if (cls.startsWith("Function")) return "function";
  return "";
}

/** Parse a single LangChain BaseMessage (or chunk) into composition entries. */
function parseLangchainMessageObj(msg: any): CompositionEntry[] {
  const out: CompositionEntry[] = [];
  const msgType = langchainMessageType(msg);
  const role = LANGCHAIN_ROLE_MAP[msgType] ?? msgType ?? "user";
  const name = msg?.name ?? undefined;
  const content = msg?.content;
  // A tool-calling AIMessage exposes its calls in BOTH content[] (as tool_use
  // blocks) AND the normalized `.tool_calls` list. Track whether we already
  // emitted them from content[] so the `.tool_calls` loop below doesn't double
  // them (same guard the LlamaIndex parser uses).
  let toolCallBlocksSeen = false;

  if (typeof content === "string") {
    if (content) {
      if (role === "tool_result") {
        out.push(textEntry("tool_result", content, name));
      } else {
        out.push(textEntry(role, content));
      }
    }
  } else if (Array.isArray(content)) {
    for (const part of content) {
      if (typeof part === "string") {
        if (part) out.push(textEntry(role, part));
      } else if (typeof part === "object" && part !== null) {
        const ptype = part.type ?? "text";
        if (ptype === "text") {
          out.push(textEntry(role, part.text ?? ""));
        } else if (ptype === "image_url" || ptype === "image") {
          out.push(nonTextEntry(role, "image"));
        } else if (ptype === "input_audio" || ptype === "audio") {
          out.push(nonTextEntry(role, "audio"));
        } else if (ptype === "tool_use") {
          const inp = canonicalJson(part.input ?? {});
          out.push({ ...textEntry("tool_call", inp, part.name ?? ""), type: "tool_call" });
          toolCallBlocksSeen = true;
        } else {
          out.push(nonTextEntry(role, String(ptype)));
        }
      }
    }
  }

  // AIMessage.tool_calls — normalized [{ name, args, id, type }] list. Skip when
  // the same calls were already emitted as content[] tool_use blocks above,
  // otherwise each call lands twice.
  const toolCalls = msg?.tool_calls;
  if (Array.isArray(toolCalls) && !toolCallBlocksSeen) {
    for (const tc of toolCalls) {
      if (typeof tc !== "object" || tc === null) continue;
      const fnArgs =
        typeof tc.args === "string" ? tc.args : canonicalJson(tc.args ?? {});
      out.push({ ...textEntry("tool_call", fnArgs, tc.name ?? ""), type: "tool_call" });
    }
  }

  return out;
}

/**
 * Flatten a LangChain prompt payload into a flat list of message-like objects.
 * Accepts: string, a single BaseMessage, BaseMessage[], BaseMessage[][] (the
 * shape BaseChatModel.generate receives), or [role, content] tuples.
 */
function coerceLangchainMessages(payload: any): any[] {
  if (payload == null) return [];
  if (typeof payload === "string") return [payload];
  // PromptValue → list[BaseMessage]. AgentExecutor / RunnableSequence pass a
  // ChatPromptValue into BaseChatModel._streamIterator, so without this the
  // parser would fall through to the Tier 3 "complete_prompt" fallback.
  if (typeof payload?.toChatMessages === "function") {
    try {
      const msgs = payload.toChatMessages();
      if (Array.isArray(msgs)) return msgs;
    } catch {
      /* fall through */
    }
  }
  // A single BaseMessage
  if (typeof payload === "object" && "content" in payload && !Array.isArray(payload)) {
    return [payload];
  }
  if (Array.isArray(payload)) {
    const flat: any[] = [];
    for (const item of payload) {
      if (Array.isArray(item) && item.length > 0 && typeof item[0] !== "string") {
        flat.push(...item);
      } else {
        flat.push(item);
      }
    }
    return flat;
  }
  return [payload];
}

function parseLangchainMessages(payload: any): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  for (const msg of coerceLangchainMessages(payload)) {
    if (typeof msg === "string") {
      if (msg) result.push(textEntry("user", msg));
    } else if (
      Array.isArray(msg) &&
      msg.length === 2 &&
      typeof msg[0] === "string"
    ) {
      const [role, content] = msg;
      const text = typeof content === "string" ? content : canonicalJson(content ?? "");
      result.push(textEntry(role, text));
    } else if (typeof msg === "object" && msg !== null) {
      result.push(...parseLangchainMessageObj(msg));
    }
  }
  return result;
}

/** Parse a LangChain LLMResult (returned by BaseChatModel.generate). */
function parseLangchainResponse(response: any): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  const generations = response?.generations;
  if (!Array.isArray(generations)) return result;
  for (let genList of generations) {
    if (!Array.isArray(genList)) genList = [genList];
    for (const gen of genList) {
      const msg = gen?.message;
      if (msg && typeof msg === "object") {
        result.push(...parseLangchainMessageObj(msg));
      } else if (gen?.text) {
        result.push(textEntry("assistant", gen.text));
      }
    }
  }
  return result;
}

// ── LlamaIndex ─────────────────────────────────────────────────────
// LlamaIndex calls flow through provider LLM classes (@llamaindex/openai,
// @llamaindex/anthropic, @llamaindex/google) .{chat,complete}. Prompts
// arrive as ChatMessage[] (LlamaIndex's own type); responses are
// ChatResponse with `.message` (a ChatMessage). Streamed chunks may carry
// `.delta` text incrementally; the final/full chunk has `.message`.

const LLAMAINDEX_ROLE_MAP: Record<string, string> = {
  user: "user",
  human: "user",
  assistant: "assistant",
  ai: "assistant",
  model: "assistant",
  chatbot: "assistant",
  system: "system",
  developer: "system",
  tool: "tool_result",
  function: "tool_result",
  memory: "system",
};

function llamaindexRole(msg: any): string {
  const raw = String(msg?.role ?? "user").toLowerCase().trim();
  return LLAMAINDEX_ROLE_MAP[raw] ?? raw ?? "user";
}

function parseLlamaIndexMessageObj(msg: any): CompositionEntry[] {
  const out: CompositionEntry[] = [];
  let role = llamaindexRole(msg);
  const additional = msg?.additionalKwargs ?? msg?.additional_kwargs ?? {};
  const options = msg?.options ?? {};
  // LlamaIndex JS encodes a tool result as `{role:'user', options:{toolResult:{id,name,...}}}`
  // (because OpenAI's `role:'tool'` isn't emitted by LlamaIndex JS' message
  // builders). Detect it so the composition shows it as `tool_result` instead
  // of a generic user turn.
  const toolResult = options?.toolResult;
  if (toolResult && typeof toolResult === "object") {
    role = "tool_result";
  }
  const name =
    (typeof additional === "object" && additional !== null
      ? additional.name ?? additional.tool_call_id ?? additional.toolCallId
      : undefined) ??
    toolResult?.name ??
    undefined;

  // Newer LlamaIndex: message.blocks (TextBlock / ImageBlock / etc.).
  const blocks = msg?.blocks;
  let handledBlocks = false;
  if (Array.isArray(blocks) && blocks.length > 0) {
    for (const block of blocks) {
      if (block == null) continue;
      const blockType = String(
        block.type ?? block.blockType ?? block.constructor?.name ?? "",
      ).toLowerCase();
      if (blockType.includes("text")) {
        const text = String(block.text ?? "");
        if (text) {
          if (role === "tool_result") {
            out.push(textEntry("tool_result", text, name));
          } else {
            out.push(textEntry(role, text));
          }
        }
        handledBlocks = true;
      } else if (blockType.includes("image")) {
        out.push(nonTextEntry(role, "image"));
        handledBlocks = true;
      } else if (blockType.includes("audio")) {
        out.push(nonTextEntry(role, "audio"));
        handledBlocks = true;
      } else if (blockType.includes("document") || blockType.includes("file")) {
        out.push(nonTextEntry(role, "document"));
        handledBlocks = true;
      } else if (blockType.includes("toolcall") || blockType.includes("tool_call")) {
        const tcName = block.toolName ?? block.tool_name ?? block.name ?? "";
        const tcArgs = block.toolKwargs ?? block.tool_kwargs ?? block.input ?? {};
        const argsStr =
          typeof tcArgs === "string" ? tcArgs : canonicalJson(tcArgs ?? {});
        out.push({ ...textEntry("tool_call", argsStr, tcName), type: "tool_call" });
        handledBlocks = true;
      }
    }
  }

  if (!handledBlocks) {
    const content = msg?.content;
    if (typeof content === "string" && content) {
      if (role === "tool_result") {
        out.push(textEntry("tool_result", content, name));
      } else {
        out.push(textEntry(role, content));
      }
    }
  }

  // additional_kwargs.tool_calls — OpenAI-shaped tool_calls list.
  const toolCalls =
    typeof additional === "object" && additional !== null
      ? additional.tool_calls ?? additional.toolCalls
      : null;
  if (Array.isArray(toolCalls)) {
    for (const tc of toolCalls) {
      if (typeof tc !== "object" || tc === null) continue;
      const fn = tc.function ?? {};
      const fnName = fn.name ?? tc.name ?? "";
      const fnArgs = fn.arguments ?? "";
      const argsStr =
        typeof fnArgs === "string" ? fnArgs : canonicalJson(fnArgs ?? {});
      out.push({ ...textEntry("tool_call", argsStr, fnName), type: "tool_call" });
    }
  }

  // LlamaIndex JS encodes assistant tool calls as
  // `options.toolCall: [{ id, name, input }]` (singular `toolCall`, plural
  // array). Detect and emit one tool_call entry per call.
  const liToolCalls = options?.toolCall ?? options?.toolCalls;
  if (Array.isArray(liToolCalls)) {
    for (const tc of liToolCalls) {
      if (typeof tc !== "object" || tc === null) continue;
      const tcName = tc.name ?? tc.toolName ?? "";
      const tcInput = tc.input ?? tc.toolKwargs ?? tc.arguments ?? {};
      const argsStr =
        typeof tcInput === "string" ? tcInput : canonicalJson(tcInput ?? {});
      out.push({ ...textEntry("tool_call", argsStr, tcName), type: "tool_call" });
    }
  }

  return out;
}

function coerceLlamaIndexMessages(payload: any): any[] {
  if (payload == null) return [];
  if (typeof payload === "string") return [payload];
  // Single ChatMessage
  if (
    typeof payload === "object" &&
    "role" in payload &&
    !Array.isArray(payload)
  ) {
    return [payload];
  }
  if (Array.isArray(payload)) {
    const flat: any[] = [];
    for (const item of payload) {
      if (
        Array.isArray(item) &&
        !(item.length === 2 && typeof item[0] === "string")
      ) {
        flat.push(...item);
      } else {
        flat.push(item);
      }
    }
    return flat;
  }
  return [payload];
}

function parseLlamaIndexMessages(payload: any): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  for (const msg of coerceLlamaIndexMessages(payload)) {
    if (typeof msg === "string") {
      if (msg) result.push(textEntry("user", msg));
    } else if (typeof msg === "object" && msg !== null) {
      result.push(...parseLlamaIndexMessageObj(msg));
    }
  }
  return result;
}

function parseLlamaIndexResponse(response: any): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  if (response == null) return result;
  // Primary: ChatResponse.message — a ChatMessage.
  const msg = response.message;
  if (msg && typeof msg === "object" && "role" in msg) {
    result.push(...parseLlamaIndexMessageObj(msg));
    if (result.length > 0) return result;
  }
  // Streaming chunk fallback: .delta is the incremental text.
  const delta = response.delta;
  if (typeof delta === "string" && delta) {
    result.push(textEntry("assistant", delta));
    return result;
  }
  // CompletionResponse fallback: .text
  const text = response.text;
  if (typeof text === "string" && text) {
    result.push(textEntry("assistant", text));
  }
  return result;
}

function parseBedrockConverseResponse(response: any): CompositionEntry[] {
  const result: CompositionEntry[] = [];
  const content = response?.output?.message?.content;
  if (!Array.isArray(content)) return result;
  for (const block of content) {
    if (typeof block !== "object" || block === null) continue;
    if ("text" in block) {
      result.push(textEntry("assistant", block.text ?? ""));
    } else if ("toolUse" in block) {
      const tu = block.toolUse ?? {};
      const inp = canonicalJson(tu.input ?? {});
      result.push({ ...textEntry("tool_call", inp, tu.name ?? ""), type: "tool_call" });
    } else if ("reasoningContent" in block) {
      result.push(nonTextEntry("assistant", "reasoning"));
    }
  }
  return result;
}

// ═══════════════════════════════════════════════════════════════════
// Tier 3: Complete fallback
// ═══════════════════════════════════════════════════════════════════

function fallbackComposition(
  payload: unknown,
  roleName: string = "complete_prompt",
): CompositionEntry[] {
  try {
    const text = typeof payload === "string" ? payload : canonicalJson(payload);
    return [textEntry(roleName, text)];
  } catch {
    return [];
  }
}

// ═══════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════

export function buildPromptComposition(
  provider: string,
  kwargs: Record<string, any>,
  operation?: string,
): CompositionEntry[] {
  try {
    const providerLower = (provider ?? "").toLowerCase().trim();

    // Embedding operation — kwargs carry an embedding input (string /
    // list-of-strings / multimodal segments) rather than chat messages.
    if (operation === "embedding") {
      return parseEmbeddingInput(kwargs);
    }

    // Tier 0: non-text generation (image / audio / video). Detect by request
    // shape and short-circuit before any chat parser runs.
    const modalityComp = tryModalityPrompt(kwargs);
    if (modalityComp != null) return modalityComp;

    // Tier 1: Supported providers
    // LangChain — prompt payload is BaseMessage objects (generate) or a string
    // / PromptValue (stream).
    if (providerLower === "langchain") {
      const comp = parseLangchainMessages(kwargs.messages);
      if (comp.length > 0) return comp;
    }

    // LlamaIndex — prompt payload is ChatMessage objects.
    if (providerLower === "llamaindex") {
      const comp = parseLlamaIndexMessages(kwargs.messages);
      if (comp.length > 0) return comp;
    }

    // OpenAI Responses API — input is either a string or a list of items;
    // top-level `instructions` is the system prompt. Structurally distinct
    // from Chat Completions; needs its own parser.
    if (providerLower === "openai_responses") {
      const comp = parseOpenAIResponsesInput(kwargs.input, kwargs.instructions);
      if (comp.length > 0) return comp;
    }

    // Vercel AI SDK — doGenerate/doStream receive a `prompt` array in the
    // LanguageModel V2/V3 shape (system as plain string, user/assistant as
    // part arrays). Parse that specifically.
    if (providerLower === "xai" || providerLower === "ai_sdk") {
      const prompt = kwargs.prompt;
      if (Array.isArray(prompt)) {
        const comp = parseXaiAiSdkPrompt(prompt);
        if (comp.length > 0) return comp;
      }
    }

    // Cohere v2 and Mistral use the OpenAI-compatible messages[] format.
    if (["openai", "grok", "openrouter", "litellm", "cerebras", "together", "cohere", "mistral", ""].includes(providerLower)) {
      const messages = kwargs.messages;
      if (Array.isArray(messages)) {
        return parseOpenAIMessages(messages);
      }
    }

    if (providerLower === "anthropic") {
      const messages = kwargs.messages;
      if (Array.isArray(messages)) {
        return parseAnthropicMessages(messages, kwargs.system);
      }
    }

    if (["google", "google_genai", "gemini"].includes(providerLower)) {
      const contents = kwargs.contents;
      if (contents != null) {
        return parseGoogleContents(contents, kwargs.system_instruction);
      }
    }

    if (providerLower === "bedrock") {
      const messages = kwargs.messages;
      if (Array.isArray(messages)) {
        return parseBedrockConverse(messages, kwargs.system);
      }
    }

    // Tier 2: Try OpenAI-compatible format as fallback
    if (Array.isArray(kwargs.messages)) {
      return parseOpenAIMessages(kwargs.messages);
    }

    if (kwargs.contents != null) {
      return parseGoogleContents(kwargs.contents, kwargs.system_instruction);
    }

    // Tier 3: Complete fallback
    const safeKwargs = { ...kwargs };
    delete safeKwargs.stream;
    delete safeKwargs.timeout;
    return fallbackComposition(safeKwargs, "complete_prompt");
  } catch {
    try {
      return fallbackComposition(kwargs, "complete_prompt");
    } catch {
      return [];
    }
  }
}

// usage.shape → modality entry type. When the caller has already resolved an
// authoritative usage shape (resolved by the intercept table in enforcer.ts)
// we short-circuit every heuristic parser — no `.text` read on a binary body
// can sneak through.
const SHAPE_TO_MEDIA: Record<string, string> = {
  openai_audio_tts: "audio",
  huggingface_audio_tts: "audio",
  google_tts: "audio",
  openai_images: "image",
  google_imagen: "image",
  together_image: "image",
  huggingface_image: "image",
  xai_image: "image",
  google_veo: "video",
  bedrock_image: "image",
};

// Embedding shapes — response is a vector, never displayable. Authoritative
// short-circuit in buildResponseComposition so parsers never inspect the
// response body looking for `.text` / `.choices`.
const EMBEDDING_SHAPES = new Set<string>([
  "openai_embeddings",
  "google_genai_embeddings",
  "cohere_embed",
  "mistral_embed",
  "voyage_embed",
  "bedrock_titan_embed",
  "huggingface_embed",
  "together_embed",
]);

// OCR shapes — response is a page-structured document (e.g. Mistral OCR).
// It carries none of `.text`/`.choices`/`.content`, so every heuristic parser
// misses and it falls to the Tier-3 hash+length fallback. Authoritative
// short-circuit in buildResponseComposition emits privacy-preserving
// per-page markers instead.
const OCR_SHAPES = new Set<string>(["mistral_ocr"]);

export function buildResponseComposition(
  provider: string,
  response: any,
  usageShape?: string,
): CompositionEntry[] {
  try {
    // Authoritative override from the caller — skips every heuristic.
    // usage.shape is a closed enum shared with the service; unknown shapes
    // fall through to the heuristics below.
    if (usageShape) {
      // Embeddings — output is a vector, never displayable.
      if (EMBEDDING_SHAPES.has(usageShape)) return [];
      const media = SHAPE_TO_MEDIA[usageShape];
      if (media != null) {
        return [nonTextEntry("assistant", media)];
      }
      // OCR — page-structured document. Emit one non-text marker per billed
      // page; never inspect the (text-bearing) page bodies.
      if (OCR_SHAPES.has(usageShape)) {
        return ocrComposition(response);
      }
    }
    const providerLower = (provider ?? "").toLowerCase().trim();

    // Defensive check: a bytes-iterator or raw-bytes body must never be fed
    // to a text parser, regardless of provider tag.
    try {
      if (looksLikeBinaryAudio(response)) {
        return [nonTextEntry("assistant", "audio")];
      }
    } catch {
      /* ignore */
    }

    // Tier 0: non-text response (image / audio / video / transcription).
    const modalityComp = tryModalityResponse(response);
    if (modalityComp != null) return modalityComp;

    // LangChain — response is an LLMResult whose generations carry AIMessages.
    if (providerLower === "langchain") {
      const result = parseLangchainResponse(response);
      if (result.length > 0) return result;
    }

    // LlamaIndex — response is a ChatResponse with `.message` (a ChatMessage).
    if (providerLower === "llamaindex") {
      const result = parseLlamaIndexResponse(response);
      if (result.length > 0) return result;
    }

    // Vercel AI SDK — doGenerate's result has `content: Array<part>`. (The
    // streaming-fallback synthetic produced by _streamAccumulatorToResponse
    // uses the same shape.)
    if (providerLower === "xai" || providerLower === "ai_sdk") {
      const result = parseXaiAiSdkResponse(response);
      if (result.length > 0) return result;
    }

    // OpenAI Responses API — response.output is a LIST of items
    // (Bedrock Converse uses an OBJECT-shaped output, checked below). Match
    // by provider OR by structural shape so accumulated synthetic responses
    // hand back to us with no provider tag still parse.
    if (providerLower === "openai_responses") {
      const result = parseOpenAIResponsesResponse(response);
      if (result.length > 0) return result;
    } else if (Array.isArray(response?.output) && response.output.length > 0) {
      const first = response.output[0];
      if (
        typeof first === "object" &&
        first != null &&
        (first.type === "message" ||
          first.type === "function_call" ||
          first.type === "reasoning")
      ) {
        const result = parseOpenAIResponsesResponse(response);
        if (result.length > 0) return result;
      }
    }

    // AWS Bedrock Converse — response is shaped { output: { message: { content } } }
    if (Array.isArray(response?.output?.message?.content)) {
      const result = parseBedrockConverseResponse(response);
      if (result.length > 0) return result;
    }

    // Cohere v2 — response carries an assistant `message` object
    // (no `choices`/`candidates`; content lives under `.message`).
    if (response?.message && !response?.choices && !response?.candidates) {
      const result = parseCohereResponse(response);
      if (result.length > 0) return result;
    }

    // OpenAI-style response
    if (response?.choices && Array.isArray(response.choices)) {
      const result: CompositionEntry[] = [];
      for (const choice of response.choices) {
        const msg = choice.message;
        if (msg) {
          if (typeof msg.content === "string") {
            result.push(textEntry("assistant", msg.content));
          }
          // OpenAI SDK uses `tool_calls`; native OpenRouter SDK uses `toolCalls`.
          const toolCalls = msg.tool_calls ?? msg.toolCalls;
          if (Array.isArray(toolCalls)) {
            for (const tc of toolCalls) {
              const fn = tc.function ?? {};
              result.push({
                ...textEntry("tool_call", fn.arguments ?? "", fn.name ?? ""),
                type: "tool_call",
              });
            }
          }
        }
      }
      if (result.length > 0) return result;
    }

    // Anthropic-style response
    if (Array.isArray(response?.content)) {
      const result: CompositionEntry[] = [];
      for (const block of response.content) {
        const blockType = block.type ?? "text";
        if (blockType === "text") {
          result.push(textEntry("assistant", block.text ?? ""));
        } else if (blockType === "tool_use") {
          const inp = canonicalJson(block.input ?? {});
          result.push({ ...textEntry("tool_call", inp, block.name ?? ""), type: "tool_call" });
        }
      }
      if (result.length > 0) return result;
    }

    // Google GenAI-style response
    if (Array.isArray(response?.candidates)) {
      const result: CompositionEntry[] = [];
      for (const candidate of response.candidates) {
        const content = candidate.content;
        if (content?.parts) {
          for (const part of content.parts) {
            // @google/genai uses camelCase; the legacy SDK used snake_case.
            // Gemini TTS / native-audio returns audio as inlineData
            // (mime audio/*) — without this branch those parts are dropped
            // and the whole response collapses to Tier-3 complete_response.
            const functionCall = part.functionCall ?? part.function_call;
            const inlineData = part.inlineData ?? part.inline_data;
            if (part.text) {
              result.push(textEntry("assistant", part.text));
            } else if (functionCall) {
              const args = canonicalJson(functionCall.args ?? {});
              result.push({
                ...textEntry("tool_call", args, functionCall.name ?? ""),
                type: "tool_call",
              });
            } else if (inlineData) {
              const mime: string =
                inlineData.mimeType ?? inlineData.mime_type ?? "unknown";
              const media = mime.includes("image")
                ? "image"
                : mime.includes("audio")
                  ? "audio"
                  : mime;
              result.push(nonTextEntry("assistant", media));
            }
          }
        }
      }
      if (result.length > 0) return result;
    }

    // Tier 3: Complete fallback
    return fallbackComposition(String(response), "complete_response");
  } catch {
    try {
      return fallbackComposition(String(response), "complete_response");
    } catch {
      return [];
    }
  }
}

/**
 * Return an ORDERED list of `{ id, name }` for tool calls in the LLM `response`
 * that carry a non-empty provider tool-call id. Used to auto-correlate the
 * model's requested tool-call ids to the subsequent toolSpan()/tool() executions
 * (consumed in context.ts). Provider-agnostic and best-effort: returns `[]` on
 * anything it can't parse and never throws — the caller runs on the customer's
 * hot path. Holds id + name only, never args/results (privacy). Gemini/Google
 * FunctionCall.id is optional — when the provider populates it we stash it;
 * when omitted we contribute nothing (never invent ids).
 */
export function extractPendingToolCalls(
  provider: string,
  response: any,
): Array<{ id: string; name: string }> {
  try {
    const out: Array<{ id: string; name: string }> = [];
    const add = (cid: any, nm: any): void => {
      const id = String(cid ?? "");
      if (id) out.push({ id, name: String(nm ?? "") });
    };
    /** OpenAI `{id, function:{name}}` or LangChain `{id, name, args}` shapes. */
    const addToolCall = (tc: any): void => {
      if (!tc || typeof tc !== "object") return;
      const fn = tc.function ?? {};
      add(tc.id ?? tc.call_id ?? tc.callId, fn.name ?? tc.name);
    };

    // OpenAI Responses API — output[] `function_call` items carry `call_id`.
    if (Array.isArray(response?.output)) {
      for (const item of response.output) {
        if (item && typeof item === "object" && item.type === "function_call") {
          add(item.call_id ?? item.callId, item.name);
        }
      }
      if (out.length > 0) return out;
    }

    // Anthropic — top-level content[] of text / tool_use blocks (id + name).
    if (Array.isArray(response?.content)) {
      for (const block of response.content) {
        if (block && typeof block === "object" && block.type === "tool_use") {
          add(block.id, block.name);
        }
      }
      if (out.length > 0) return out;
    }

    // OpenAI chat-completions — choices[].message.{tool_calls|toolCalls}[]
    // (openai, mistral, together, cerebras, openrouter, huggingface, groq…).
    // Name: function.name (OpenAI) or top-level name (LangChain-shaped).
    if (Array.isArray(response?.choices)) {
      for (const choice of response.choices) {
        const msg = choice?.message;
        const toolCalls = msg?.tool_calls ?? msg?.toolCalls;
        if (Array.isArray(toolCalls)) {
          for (const tc of toolCalls) addToolCall(tc);
        }
      }
      if (out.length > 0) return out;
    }

    // Cohere v2 — assistant `message.{toolCalls|tool_calls}[]` ({id, function:{name}}).
    if (response?.message) {
      const toolCalls = response.message.toolCalls ?? response.message.tool_calls;
      if (Array.isArray(toolCalls)) {
        for (const tc of toolCalls) addToolCall(tc);
      }
      if (out.length > 0) return out;
    }

    // LlamaIndex TS — assistant tool calls hang off the message's options as
    // `options.toolCall: [{id, name, input}]` (singular key, array value; some
    // versions spell it `toolCalls`); some provider adapters instead surface
    // them the OpenAI way under `additionalKwargs.tool_calls`.
    // parseLlamaIndexMessageObj reads BOTH shapes to build the composition but
    // drops the id — this keeps only the id. Options first, returning on
    // whichever source yields an id, so an adapter that populates both can't
    // double-count (mirrors the Python branch). No other provider shape carries
    // `message.options`, and the additionalKwargs read is reached only after
    // every earlier branch declined, so nothing above or below is shadowed.
    if (response?.message) {
      const liOptions = response.message.options;
      if (liOptions) {
        const liToolCalls = liOptions.toolCall ?? liOptions.toolCalls;
        if (Array.isArray(liToolCalls)) {
          for (const tc of liToolCalls) {
            if (!tc || typeof tc !== "object") continue;
            add(tc.id ?? tc.toolCallId, tc.name ?? tc.toolName);
          }
        }
        if (out.length > 0) return out;
      }
      const liExtra =
        response.message.additionalKwargs ?? response.message.additional_kwargs;
      if (liExtra && typeof liExtra === "object") {
        const extraCalls = liExtra.tool_calls ?? liExtra.toolCalls;
        if (Array.isArray(extraCalls)) {
          for (const tc of extraCalls) addToolCall(tc);
        }
        if (out.length > 0) return out;
      }
    }

    // @ai-sdk/xai — content[] `tool-call` parts carry `toolCallId` + `toolName`.
    if (Array.isArray(response?.content)) {
      for (const part of response.content) {
        if (part && typeof part === "object" && part.type === "tool-call") {
          add(part.toolCallId, part.toolName);
        }
      }
      if (out.length > 0) return out;
    }

    // LangChain AIMessage / xAI-like — top-level tool_calls: [{id, name, args}].
    {
      const tcs = response?.tool_calls ?? response?.toolCalls;
      if (Array.isArray(tcs)) {
        for (const tc of tcs) addToolCall(tc);
        if (out.length > 0) return out;
      }
    }

    // LangChain LLMResult — generations[][].message.tool_calls (generate path).
    if (Array.isArray(response?.generations)) {
      for (const genList of response.generations) {
        const list = Array.isArray(genList) ? genList : [genList];
        for (const gen of list) {
          const msg = gen?.message;
          const toolCalls = msg?.tool_calls ?? msg?.toolCalls;
          if (Array.isArray(toolCalls)) {
            for (const tc of toolCalls) addToolCall(tc);
          }
        }
      }
      if (out.length > 0) return out;
    }

    // Gemini / Google GenAI — candidates[].content.parts[].functionCall
    // (camelCase REST) or function_call (snake_case). id is optional.
    if (Array.isArray(response?.candidates)) {
      for (const cand of response.candidates) {
        const parts = cand?.content?.parts;
        if (!Array.isArray(parts)) continue;
        for (const part of parts) {
          const fc = part?.function_call ?? part?.functionCall;
          if (fc && typeof fc === "object") {
            add(fc.id, fc.name);
          }
        }
      }
      if (out.length > 0) return out;
    }

    return out;
  } catch {
    return [];
  }
}
