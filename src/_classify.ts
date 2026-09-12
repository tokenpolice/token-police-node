/**
 * Provider-agnostic LLM exception classifier.
 * Behavior is kept in parity with the TokenPolice Python SDK.
 */

import { createHash } from "crypto";
import { getClient } from "./state";

const RATE_LIMIT_NAMES = new Set(["RateLimitError", "RateLimit", "TooManyRequests"]);
const TIMEOUT_NAMES = new Set([
  "Timeout", "TimeoutError", "ReadTimeout", "ConnectTimeout",
  "APITimeoutError", "APIConnectionTimeoutError",
]);
const AUTH_NAMES = new Set([
  "AuthenticationError", "PermissionDeniedError", "Unauthorized",
  "InvalidAPIKey", "Forbidden",
]);
const BADREQ_NAMES = new Set([
  "BadRequestError", "InvalidRequestError", "UnprocessableEntityError",
]);
const SERVER_NAMES = new Set([
  "InternalServerError", "APIError", "ServiceUnavailable", "BadGateway",
]);
const NETWORK_NAMES = new Set([
  "APIConnectionError", "ConnectionError", "NetworkError", "FetchError",
]);

export type ErrorKind =
  | "rate_limited" | "timeout" | "auth_error" | "server_error"
  | "client_error" | "network_error" | "unknown";

export interface Classification {
  error_kind: ErrorKind;
  http_status: number;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function nameOf(exc: any): string {
  try { return exc?.constructor?.name || ""; } catch { return ""; }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _directStatusOf(exc: any): number {
  try {
    for (const attr of ["status", "statusCode", "status_code", "httpStatus", "http_status"]) {
      const v = exc?.[attr];
      if (typeof v === "number" && v >= 100 && v < 600) return v;
    }
    const r = exc?.response;
    if (r) {
      const v = r.status ?? r.statusCode ?? r.status_code;
      if (typeof v === "number" && v >= 100 && v < 600) return v;
    }
    // AWS SDK v3 ServiceExceptions carry the status ONLY at
    // $metadata.httpStatusCode ($-prefixed fields; no top-level status, no
    // .response) — e.g. bedrock AccessDeniedException -> 403.
    const m = exc?.$metadata?.httpStatusCode;
    if (typeof m === "number" && Number.isInteger(m) && m >= 100 && m < 600) return m;
    // @huggingface/inference v4 HttpRequestError subclasses (ProviderApiError,
    // HubApiError) carry the status ONLY at httpResponse.status — no top-level
    // status attr, no .response, and a message with no parsable code. Read
    // .status alone: httpResponse.body holds the raw provider error payload
    // and must never be read here (token-metadata-only invariant).
    const h = exc?.httpResponse?.status;
    if (typeof h === "number" && Number.isInteger(h) && h >= 100 && h < 600) return h;
  } catch {
    // ignored
  }
  return 0;
}

// Wrapper-attr names an app/provider shim may use to hold the underlying
// error. Parity with the Python SDK's equivalent attribute list: JS `cause` is
// the analog of Python's `__cause__` (Python's implicit `__context__` has no JS
// equivalent, so it is intentionally omitted); `originalError` is the JS camel
// variant of `original_error`.
const _WRAPPER_ATTRS = [
  "cause", "inner", "original", "originalError", "original_error", "original_exception",
];

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _statusFromMessage(exc: any): number {
  // Providers/app shims that throw plain Errors encode the code in the message
  // ("OpenAI HTTP 429: …", "status code: 400"). Total: never throws.
  try {
    const msg = String(exc?.message ?? "");
    const m = msg.match(/(?:HTTP|status(?:\s+code)?)[:\s]+(\d{3})\b/i);
    if (m) {
      const code = Number(m[1]);
      if (code >= 100 && code < 600) return code;
    }
  } catch {
    // ignored
  }
  return 0;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function _chainLinks(exc: any, maxDepth = 4): any[] {
  // Breadth-first, cycle-safe, bounded walk of the wrapped/chained exceptions
  // behind `exc`, parity with the Python SDK's chain-walk. The visited `Set` is
  // seeded with the top exception and any already-seen link is skipped before
  // it is enqueued/scanned — this replaces the old self-loop-only break, which
  // is insufficient for a multi-attr BFS. Each wrapper-attr read is
  // individually try-guarded so a throwing getter on one attr continues to the
  // next sibling attr rather than aborting the whole scan. Never throws.
  const links: any[] = [];
  const seen = new Set<object>();
  if (exc && typeof exc === "object") seen.add(exc);
  let frontier: any[] = [exc];
  for (let depth = 0; depth < maxDepth; depth++) {
    const nxt: any[] = [];
    for (const cur of frontier) {
      for (const attr of _WRAPPER_ATTRS) {
        let link: any;
        try {
          link = cur?.[attr];
        } catch {
          continue;
        }
        if (link && typeof link === "object" && !seen.has(link)) {
          seen.add(link);
          nxt.push(link);
          links.push(link);
        }
      }
    }
    if (nxt.length === 0) break;
    frontier = nxt;
  }
  return links;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function httpStatusOf(exc: any): number {
  const direct = _directStatusOf(exc);
  if (direct) return direct;
  // Wrapped errors (stream-iteration wrappers, app shims) often hide the
  // HTTP-carrying error one or more levels down. Everything below is gated on
  // the direct read missing, and stays inside this `try {}` so httpStatusOf
  // never throws (returns 0 on any failure).
  try {
    // (a) Walk the wrapper chain (BFS, depth-capped, cycle-safe) and re-run the
    // direct attribute checks on each link.
    const links = _chainLinks(exc, 4);
    for (const link of links) {
      const v = _directStatusOf(link);
      if (v) return v;
    }
    // (b) Last resort: parse the code from the message ("OpenAI HTTP 429: …",
    // "status code: 400"), top exception first, then the chained links.
    const topMsg = _statusFromMessage(exc);
    if (topMsg) return topMsg;
    for (const link of links) {
      const v = _statusFromMessage(link);
      if (v) return v;
    }
  } catch {
    // ignored
  }
  return 0;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function classifyException(exc: any): Classification {
  try {
    const name = nameOf(exc);
    const status = httpStatusOf(exc);
    if (RATE_LIMIT_NAMES.has(name) || status === 429) return { error_kind: "rate_limited", http_status: status || 429 };
    if (TIMEOUT_NAMES.has(name) || name.includes("Timeout")) return { error_kind: "timeout", http_status: status };
    if (AUTH_NAMES.has(name) || status === 401 || status === 403) return { error_kind: "auth_error", http_status: status || 401 };
    if (NETWORK_NAMES.has(name)) return { error_kind: "network_error", http_status: status };
    if (status >= 500 && status < 600) return { error_kind: "server_error", http_status: status };
    if (status >= 400 && status < 500) return { error_kind: "client_error", http_status: status };
    if (SERVER_NAMES.has(name)) return { error_kind: "server_error", http_status: status };
    if (BADREQ_NAMES.has(name)) return { error_kind: "client_error", http_status: status || 400 };
  } catch {
    // ignored
  }
  return { error_kind: "unknown", http_status: 0 };
}

export interface CallOutcome {
  status: "success" | "failed";
  duration_ms: number;
  error_kind?: ErrorKind;
  http_status?: number;
  error_message?: string;
  error_message_hash?: string;
  error_class?: string;
}

/**
 * Error-message detail mode. Controls what (if any) message-derived signal
 * leaves the process on a failed call. Covered by errorMessageScrub.test.ts.
 * - "none": drop the raw string and every message-derived field (no hash,
 * no class). Only the safe classifier enums survive.
 * - "redacted": (default) drop the raw string; ship a SHA-256 hash of it plus
 * the exception class name (central builders only).
 * - "raw": opt-in — restore the legacy verbatim 500-char string.
 */
export type ErrorDetailMode = "none" | "redacted" | "raw";

function coerceErrorDetail(mode: unknown): ErrorDetailMode {
  return mode === "none" || mode === "redacted" || mode === "raw" ? mode : "redacted";
}

/**
 * Single resolution point for the configured error-detail mode. Reads the
 * global client's `errorDetail`; a missing/throwing client (or any non-member
 * value) yields the SAFE default "redacted" — never "raw". Total: never throws.
 */
export function resolveErrorDetail(): ErrorDetailMode {
  try {
    const d = getClient()?.errorDetail;
    if (d === "none" || d === "redacted" || d === "raw") return d;
  } catch {
    // ignored
  }
  return "redacted";
}

/**
 * Error-message scrub helper — pure and total. Takes the raw error value
 * (`unknown`) so the stringify + truncate + SHA-256 all happen inside this
 * guard; no call site stringifies first (a hostile `toString`/`message`
 * getter is contained here). Returns only message-derived fields; on any
 * internal failure returns `{}` (the safest shape). Does no I/O, never throws.
 *
 * - "none": `{}` (nothing derived from the message).
 * - "redacted": `{ error_message_hash }` (SHA-256 hex of the FULL pre-truncation
 * string; empty string → `{}`, we don't ship a hash-of-empty).
 * - "raw": `{ error_message }` (legacy `String(...).slice(0,500)`).
 */
export function scrubErrorMessage(raw: unknown, mode: ErrorDetailMode): Partial<CallOutcome> {
  const m = coerceErrorDetail(mode);
  if (m === "none") return {};
  let s: string;
  try {
    const base =
      raw !== null && raw !== undefined && typeof raw === "object" && "message" in raw
        ? (raw as { message?: unknown }).message
        : raw;
    s = String(base ?? "");
  } catch {
    return {};
  }
  if (m === "raw") return { error_message: s.slice(0, 500) };
  // redacted
  if (s === "") return {};
  try {
    return { error_message_hash: createHash("sha256").update(s, "utf8").digest("hex") };
  } catch {
    return {};
  }
}

/**
 * Convert a wall-clock delta (ms, float) to wire `duration_ms` (UInt32).
 *
 * Positive sub-millisecond deltas must not collapse to 0 — report 1 so
 * "ran but fast" is distinguishable from "no duration recorded". True-zero /
 * negative / non-finite deltas stay 0. Multi-ms keeps Node's historical
 * `Math.round` (Python floors; only the sub-ms zero is harmonized).
 * Pure arithmetic; never throws.
 */
export function toDurationMs(deltaMs: number): number {
  try {
    if (!Number.isFinite(deltaMs) || !(deltaMs > 0)) return 0;
    return Math.max(1, Math.round(deltaMs));
  } catch {
    return 0;
  }
}

/**
 * Monotonic clock in milliseconds for duration measurement.
 * Prefers `performance.now()` (sub-ms); falls back to `Date.now()` if the mono
 * clock is unavailable. Never throws into customer code.
 */
export function safeMonoNow(): number {
  try {
    const p = globalThis.performance;
    if (p && typeof p.now === "function") {
      const n = p.now();
      if (typeof n === "number" && Number.isFinite(n)) return n;
    }
  } catch {
    /* fall through */
  }
  try {
    return Date.now();
  } catch {
    return 0;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function buildCallOutcome(exc: any, durationMs: number): CallOutcome {
  if (exc === null || exc === undefined) {
    return { status: "success", duration_ms: Math.floor(durationMs) };
  }
  const c = classifyException(exc);
  const mode = resolveErrorDetail();
  const outcome: CallOutcome = {
    status: "failed",
    duration_ms: Math.floor(durationMs),
    error_kind: c.error_kind,
    http_status: c.http_status,
    ...scrubErrorMessage(exc, mode),
  };
  // error_class = exception constructor name (a type name, not message
  // content). Added ONLY in redacted mode by this central builder (which holds
  // the exception object). Best-effort: nameOf is itself try/guarded.
  if (mode === "redacted") {
    const cls = nameOf(exc);
    if (cls) outcome.error_class = cls;
  }
  return outcome;
}
