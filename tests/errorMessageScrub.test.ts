import { describe, it, expect, afterEach } from "vitest";
import { createHash } from "crypto";
import { TokenPolice, type TokenPoliceOptions } from "../src/client";
import {
  scrubErrorMessage,
  resolveErrorDetail,
  buildCallOutcome,
} from "../src/_classify";
import { setClient, getClient } from "../src/state";
import { tool } from "../src/context";

// ── raw provider/tool exception text must not ship by default ─────
//
// On a FAILED call the raw error string used to be captured verbatim
// (String(...).slice(0,500)) into call_outcome.error_message and forwarded to
// the server — a plaintext leak (provider 400s / tool errors echo prompt
// content). The fix: a centralized, pure & TOTAL scrub helper + an
// `errorDetail` config ("none" | "redacted" | "raw", default "redacted").
// Mirrors token-police-python/tests/test_error_message_scrub.py.

const API_KEY = "tp_sk_test123";

// SHA-256 hex of the exact string "boom" — a SHARED constant the Python test
// asserts against too (assertion 17: cross-SDK hash parity).
const SHA256_BOOM =
  "81f52337ebb4cb1669bb802c708807dde0519d15cb102a6313d26ad5cd821713";

function sha256hex(s: string): string {
  return createHash("sha256").update(s, "utf8").digest("hex");
}

function makeClient(opts: Partial<TokenPoliceOptions> & { errorDetail?: string }): TokenPolice {
  return new TokenPolice({ apiKey: API_KEY, ...opts } as TokenPoliceOptions);
}

describe("Error message scrub", () => {
  const created: TokenPolice[] = [];
  function track(c: TokenPolice): TokenPolice {
    created.push(c);
    return c;
  }
  // Install a client carrying the desired errorDetail so buildCallOutcome /
  // the emit sites resolve to that mode via getClient().
  function useMode(mode: string): void {
    setClient(track(makeClient({ errorDetail: mode })));
  }

  afterEach(() => {
    setClient(null as unknown as TokenPolice);
    while (created.length) created.pop()?.closeSync();
  });

  // ── Config surface (assertions 1, 3) ─────────────────────────────────
  it("default error detail is redacted", () => {
    expect(track(makeClient({})).errorDetail).toBe("redacted");
  });

  it("unknown error detail falls back to redacted", () => {
    for (const bad of ["verbose", "Raw", "", "plaintext", "RAW", "none "]) {
      expect(() => track(makeClient({ errorDetail: bad }))).not.toThrow();
      expect(track(makeClient({ errorDetail: bad })).errorDetail).toBe("redacted");
    }
    // valid values pass through
    for (const good of ["none", "redacted", "raw"]) {
      expect(track(makeClient({ errorDetail: good })).errorDetail).toBe(good);
    }
  });

  // ── Helper directly (pure function) ──────────────────────────────────
  it("helper: none mode returns no message-derived fields", () => {
    expect(scrubErrorMessage("boom", "none")).toEqual({});
    expect(scrubErrorMessage(new Error("boom"), "none")).toEqual({});
  });

  it("helper: redacted mode returns the SHA-256 hash, not the raw string", () => {
    const out = scrubErrorMessage("boom", "redacted");
    expect(out).toEqual({ error_message_hash: SHA256_BOOM });
    expect(out.error_message).toBeUndefined();
  });

  it("helper: redacted hash is over the FULL pre-truncation string (assertion 7)", () => {
    const long = "x".repeat(600);
    const out = scrubErrorMessage(long, "redacted");
    expect(out.error_message_hash).toBe(sha256hex(long));
    // NOT the hash of the 500-char slice.
    expect(out.error_message_hash).not.toBe(sha256hex(long.slice(0, 500)));
  });

  it("helper: empty raw produces neither message nor hash (assertion 8)", () => {
    expect(scrubErrorMessage("", "redacted")).toEqual({});
    expect(scrubErrorMessage(null, "redacted")).toEqual({});
    expect(scrubErrorMessage(undefined, "redacted")).toEqual({});
  });

  it("helper: raw mode reproduces the legacy 500-char string (assertion 10)", () => {
    expect(scrubErrorMessage("boom", "raw")).toEqual({ error_message: "boom" });
    const long = "y".repeat(600);
    const out = scrubErrorMessage(long, "raw");
    expect(out.error_message).toBe(long.slice(0, 500));
    expect((out.error_message as string).length).toBe(500);
    // raw mode emits NO hash, NO class.
    expect(out.error_message_hash).toBeUndefined();
    expect(out.error_class).toBeUndefined();
  });

  it("helper: unknown mode is defensively coerced to redacted", () => {
    expect(scrubErrorMessage("boom", "verbose" as never)).toEqual({
      error_message_hash: SHA256_BOOM,
    });
  });

  // ── Pathological input (assertion 15, Golden Rule) ───────────────────
  it("pathological error object never throws (fed to the shared helper directly)", () => {
    // An object whose message/toString/Symbol.toPrimitive getters all throw —
    // the SAME helper the string-only emit sites call.
    const hostile: Record<string | symbol, unknown> = {};
    Object.defineProperty(hostile, "message", {
      get() {
        throw new Error("getter explodes");
      },
    });
    hostile.toString = () => {
      throw new Error("toString explodes");
    };
    Object.defineProperty(hostile, Symbol.toPrimitive, {
      get() {
        throw new Error("toPrimitive explodes");
      },
    });

    for (const mode of ["none", "redacted", "raw"] as const) {
      let out: unknown;
      expect(() => {
        out = scrubErrorMessage(hostile, mode);
      }).not.toThrow();
      // Degraded but valid: never leaks, never the hostile object.
      expect(out).toBeTypeOf("object");
      expect((out as { error_message?: unknown }).error_message).not.toBe(hostile);
    }
    // And via the central builder (which also routes through the helper).
    setClient(track(makeClient({ errorDetail: "redacted" })));
    expect(() => buildCallOutcome(hostile, 10)).not.toThrow();
  });

  // ── Central builder shapes ───────────────────────────────────────────
  it("redacted central builder: hash + class, no raw message (assertion 5)", () => {
    useMode("redacted");
    const out = buildCallOutcome(new Error("boom"), 10);
    expect(out.error_message).toBeUndefined();
    expect(out.error_message_hash).toBe(SHA256_BOOM);
    expect(out.error_class).toBe("Error");
    expect(out.error_kind).toBeDefined();
    expect(out.http_status).toBeDefined();
    expect(out.status).toBe("failed");
  });

  it("none central builder: no message, no hash, no class (assertion 9)", () => {
    useMode("none");
    const out = buildCallOutcome(new Error("boom"), 10);
    expect(out.error_message).toBeUndefined();
    expect(out.error_message_hash).toBeUndefined();
    expect(out.error_class).toBeUndefined();
    // safe classifier enums survive
    expect(out.error_kind).toBeDefined();
    expect(out.http_status).toBeDefined();
  });

  it("raw central builder: legacy verbatim message, no hash/class (assertion 10/11)", () => {
    useMode("raw");
    const out = buildCallOutcome(new Error("boom"), 10);
    expect(out.error_message).toBe("boom");
    expect(out.error_message_hash).toBeUndefined();
    expect(out.error_class).toBeUndefined();
    // 600-char message → exactly first 500.
    const out2 = buildCallOutcome(new Error("z".repeat(600)), 10);
    expect(out2.error_message).toBe("z".repeat(500));
  });

  // ── Success path untouched (assertion 19) ────────────────────────────
  it("success path is byte-for-byte unchanged in every mode (assertion 19)", () => {
    for (const mode of ["none", "redacted", "raw"]) {
      useMode(mode);
      expect(buildCallOutcome(null, 10)).toEqual({ status: "success", duration_ms: 10 });
      expect(buildCallOutcome(undefined, 10)).toEqual({ status: "success", duration_ms: 10 });
    }
  });

  // ── error_kind / http_status preserved in every mode (assertion 20) ──
  it("error_kind/http_status preserved across none/redacted/raw (assertion 20)", () => {
    const err = Object.assign(new Error("rate limited"), { status: 429 });
    for (const mode of ["none", "redacted", "raw"]) {
      useMode(mode);
      const out = buildCallOutcome(err, 5);
      expect(out.error_kind).toBe("rate_limited");
      expect(out.http_status).toBe(429);
    }
  });

  // ── resolveErrorDetail single resolution point (assertion 24) ────────
  it("resolveErrorDetail falls back to redacted with no/throwing client (assertion 24)", () => {
    setClient(null as unknown as TokenPolice);
    expect(resolveErrorDetail()).toBe("redacted");
    setClient(track(makeClient({ errorDetail: "raw" })));
    expect(resolveErrorDetail()).toBe("raw");
    expect(getClient()?.errorDetail).toBe("raw");
  });

  // ── Behavioral: string-only emit site (context.ts tool path) routed ──
  it("context tool() failure path is scrubbed (redacted default) (assertion 12)", () => {
    const logged: unknown[][] = [];
    // Mock client: log capture + errorDetail (default redacted when absent).
    setClient({
      log: (...args: unknown[]) => logged.push(args),
      // no errorDetail → resolveErrorDetail() yields "redacted"
    } as unknown as TokenPolice);

    const boom = tool({ name: "boom" }, () => {
      throw new Error("kaboom-secret");
    });
    expect(() => boom()).toThrow("kaboom-secret");
    expect(logged).toHaveLength(1);
    // tp.log positional: span=args[10], extras=args[13].
    const extras = logged[0][13] as { call_outcome: Record<string, unknown> };
    const co = extras.call_outcome;
    expect(co.status).toBe("failed");
    expect(co.error_message).toBeUndefined();
    expect(co.error_message_hash).toBe(sha256hex("kaboom-secret"));
    // Raw error text never appears anywhere in the logged payload.
    expect(JSON.stringify(logged[0])).not.toContain("kaboom-secret");
  });

  it("context tool() failure path under raw mode restores the message", () => {
    const logged: unknown[][] = [];
    setClient({
      log: (...args: unknown[]) => logged.push(args),
      errorDetail: "raw",
    } as unknown as TokenPolice);

    const boom = tool({ name: "boom" }, () => {
      throw new Error("kaboom-raw");
    });
    expect(() => boom()).toThrow("kaboom-raw");
    const extras = logged[0][13] as { call_outcome: Record<string, unknown> };
    expect(extras.call_outcome.error_message).toBe("kaboom-raw");
    expect(extras.call_outcome.error_message_hash).toBeUndefined();
  });
});
