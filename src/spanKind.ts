/**
 * Canonical span_kind taxonomy + derivation, kept in lockstep with the
 * server-side derivation and the Python SDK.
 *
 * span_kind describes what a span row *is*:
 * - Structural anchors (no usage / no cost): `agent` (workflow / sub-agent root),
 * `tool` (tool / function execution), `chain` (composite/chain anchor).
 * - Billable model calls: `llm` (text chat) plus per-modality kinds derived from
 * the call's `operation`: embedding / image / tts / stt / video / ocr / rerank /
 * moderation.
 *
 * span_kind is re-derived authoritatively server-side, so this is a
 * best-effort, fail-safe convenience: an SDK failure must never propagate
 * into the caller, so any failure here falls back to `"llm"`.
 */

// Structural, no-usage anchor rows. Trusted verbatim; never re-derived.
export const STRUCTURAL_SPAN_KINDS = new Set(["agent", "tool", "chain"]);

// operation -> span_kind. chat/unknown collapse to the text-chat default `llm`.
const OPERATION_TO_SPAN_KIND: Record<string, string> = {
  chat: "llm",
  unknown: "llm",
  embedding: "embedding",
  image_gen: "image",
  audio_tts: "tts",
  audio_stt: "stt",
  video_gen: "video",
  ocr: "ocr",
  rerank: "rerank",
  moderation: "moderation",
};

/**
 * Resolve the modality-aware span_kind. Structural kinds the caller already set
 * (agent/tool/chain) are returned verbatim — those rows carry the default
 * `operation="chat"` which must NOT turn them into `llm`. Everything else derives
 * from `operation`, defaulting to `llm`. Never throws.
 */
export function spanKindFor(operation?: string, incomingKind?: unknown): string {
  try {
    if (typeof incomingKind === "string" && STRUCTURAL_SPAN_KINDS.has(incomingKind)) {
      return incomingKind;
    }
    return OPERATION_TO_SPAN_KIND[operation ?? ""] || "llm";
  } catch {
    return "llm";
  }
}
