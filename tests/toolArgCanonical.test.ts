/**
 * Tool-arg hash/length SINK parity (Node side).
 *
 * The 3 Node tool-arg sinks fingerprint an arg as (sha1-16, code-point length):
 * - frameworkTools `hashLen` — driven via the public `emitToolRow`
 * - context `_toolHashLen` — driven via the public `toolSpan`
 * - telemetry `hashLen` — driven via the public `TokenPoliceSpanProcessor.onEnd`
 *
 * A STRING arg is hashed BYTE-UNCHANGED (preserves the OTel span-attr hash
 * baseline — NOT canonicalized/quoted); only its LENGTH is counted in code
 * points. A NON-STRING arg routes through the `canonicalJson` then hashed.
 *
 * Reads the SHARED oracle `shared/composition-canonical-fixture.json`
 * `tool_arg_cases` block (also read by test_tool_arg_canonical.py) — every
 * pinned value was COMPUTED by running both runtimes' serializers + sha1 +
 * code-point count and cross-checked byte-identical.
 */
import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { createHash } from "node:crypto";
import { setClient } from "../src/state";
import { TPSession, _getSessionStorage, toolSpan } from "../src/context";
import { emitToolRow } from "../src/frameworkTools";
import { TokenPoliceSpanProcessor } from "../src/telemetry";

type Branch = "string" | "canonical";
interface ToolArgCase {
  _name: string;
  _branch: Branch;
  input: unknown;
  canonical?: string;
  hash: string;
  length: number;
}

const fixture = JSON.parse(
  readFileSync(join(__dirname, "../../shared/composition-canonical-fixture.json"), "utf8"),
) as { tool_arg_cases: ToolArgCase[] };

const cases = fixture.tool_arg_cases;

const EXTRAS_ARG = 13;

function sha116(s: string): string {
  return createHash("sha1").update(s, "utf8").digest("hex").slice(0, 16);
}
function codePoints(s: string): number {
  return [...s].length;
}

const SESSION = new TPSession({
  userId: "u1",
  paidPlan: "pro",
  workflowName: "wf",
  traceId: "a".repeat(32),
  rootSpanId: "b".repeat(16),
});
function withSession<T>(fn: () => T): T {
  return _getSessionStorage().run(SESSION, fn);
}

let logged: any[];
beforeEach(() => {
  logged = [];
  setClient({ log: (...args: any[]) => logged.push(args) } as any);
});
afterEach(() => {
  setClient(null as any);
});

// ── The 3 Node sinks, each driven through its public path ────────────

/** frameworkTools.hashLen via emitToolRow → extras.tool.param_hash/param_length. */
function sinkFrameworkTools(arg: unknown): [string, number] {
  logged = [];
  withSession(() => emitToolRow({ name: "t", input: arg, output: undefined }));
  const tool = logged[0][EXTRAS_ARG].tool;
  return [tool.param_hash, tool.param_length];
}

/** context._toolHashLen via toolSpan → extras.tool.param_hash/param_length. */
function sinkContext(arg: unknown): [string, number] {
  logged = [];
  withSession(() => toolSpan({ name: "t", args: arg }, () => undefined));
  const tool = logged[0][EXTRAS_ARG].tool;
  return [tool.param_hash, tool.param_length];
}

/** telemetry.hashLen via TokenPoliceSpanProcessor.onEnd (tp.kind=tool). */
function sinkTelemetry(arg: unknown): [string, number] {
  logged = [];
  const proc = new TokenPoliceSpanProcessor();
  proc.onEnd({
    attributes: { "tp.kind": "tool", "traceloop.entity.input": arg },
    name: "t",
    startTime: [0, 0] as [number, number],
    endTime: [1, 0] as [number, number],
    spanContext: () => ({
      traceId: "0000000000000000000000000000abcd",
      spanId: "0000000000001234",
    }),
    parentSpanId: undefined,
    status: { code: 0 },
  } as any);
  const tool = logged[0][EXTRAS_ARG].tool;
  return [tool.param_hash, tool.param_length];
}

const SINKS: Array<[string, (a: unknown) => [string, number]]> = [
  ["frameworkTools.hashLen (emitToolRow)", sinkFrameworkTools],
  ["context._toolHashLen (toolSpan)", sinkContext],
  ["telemetry.hashLen (onEnd)", sinkTelemetry],
];

// ── Assertions 9, 10, 12, 18: cross-SDK byte-identity via shared oracle ──
describe("Sinks reproduce the shared oracle (assertions 9,10,12,18)", () => {
  for (const [sinkName, sink] of SINKS) {
    for (const c of cases) {
      it(`${sinkName}:: ${c._name}`, () => {
        const [hash, length] = sink(c.input);
        expect(hash).toBe(c.hash);
        expect(length).toBe(c.length);
        if (c._branch === "canonical") {
          // hash/length are over the canonical string oracle
          expect(sha116(c.canonical as string)).toBe(c.hash);
          expect(codePoints(c.canonical as string)).toBe(c.length);
        } else {
          // string branch: raw input hashed byte-unchanged (assertion 6/11)
          expect(sha116(c.input as string)).toBe(c.hash);
          expect(codePoints(c.input as string)).toBe(c.length);
        }
      });
    }
  }
});

// ── Assertion 6: ASCII string passthrough (RED-guard vs canonicalizing) ──
describe("String HASH byte-unchanged (assertion 6)", () => {
  for (const [sinkName, sink] of SINKS) {
    it(`${sinkName}: "hello" → raw sha1, length 5 (NOT quoted)`, () => {
      const [hash, length] = sink("hello");
      // pre-fix value: sha1 of the RAW string "hello" (not '"hello"')
      expect(hash).toBe("aaf4c61ddcc5e8a2");
      expect(hash).not.toBe(sha116('"hello"')); // would be the value if we canonicalized
      expect(length).toBe(5);
    });
  }
});

// ── Assertion 11: astral STRING length = code points (RED vs UTF-16) ──
describe("Astral STRING branch length = code points (assertion 11)", () => {
  for (const [sinkName, sink] of SINKS) {
    it(`${sinkName}: raw "😀" → length 1 (not UTF-16 2), hash unchanged`, () => {
      const [hash, length] = sink("😀");
      expect(length).toBe(1); // RED vs Node pre-fix ".length" === 2
      expect(hash).toBe(sha116("😀")); // raw string hashed, unchanged
    });
    it(`${sinkName}: raw '{"emoji":"😀"}' → length 13 (not 15)`, () => {
      const [hash, length] = sink('{"emoji":"😀"}');
      expect(length).toBe(13);
      expect(hash).toBe("f71d0df8d03ac2e0");
    });
    it(`${sinkName}: BMP string no-regression '{"query":"weather"}'`, () => {
      const [hash, length] = sink('{"query":"weather"}');
      expect(length).toBe(19);
      expect(hash).toBe("ca7991fe11831ace");
    });
  }
});

// ── Assertion 7: null/undefined early return preserved ──
describe("Null/undefined early return preserved (assertion 7)", () => {
  for (const [sinkName, sink] of SINKS) {
    it(`${sinkName}: null → ["",0]`, () => {
      expect(sink(null)).toEqual(["", 0]);
    });
    it(`${sinkName}: undefined → ["",0]`, () => {
      expect(sink(undefined)).toEqual(["", 0]);
    });
  }
});

// ── Assertion 8: no-throw on hostile non-string input ──
describe("No-throw on hostile input (assertion 8)", () => {
  function hostiles(): unknown[] {
    const circular: any = { a: 1 };
    circular.self = circular;
    const throwingToJSON: any = {
      toJSON() {
        throw new Error("toJSON explodes");
      },
    };
    const throwingGetter: any = {};
    Object.defineProperty(throwingGetter, "boom", {
      enumerable: true,
      get() {
        throw new Error("getter explodes");
      },
    });
    return [circular, throwingToJSON, throwingGetter, 10n, new Date(0)];
  }
  for (const [sinkName, sink] of SINKS) {
    for (const h of hostiles()) {
      it(`${sinkName}: hostile ${Object.prototype.toString.call(h)} → tuple, no throw`, () => {
        let out: [string, number] | undefined;
        expect(() => {
          out = sink(h);
        }).not.toThrow();
        expect(Array.isArray(out)).toBe(true);
        expect(typeof out![0]).toBe("string");
        expect(typeof out![1]).toBe("number");
      });
    }
  }
});

// ── Assertion 3/14 (source guard for the non-drivable telemetry sink too):
// no residual JSON.stringify / String(val) feeds any of the 3 sink bodies ──
describe("No residual language-local serializer in sink bodies (assertions 3,14)", () => {
  const files = ["../src/frameworkTools.ts", "../src/context.ts", "../src/telemetry.ts"];
  const fnNames = ["hashLen", "_toolHashLen"];
  for (const f of files) {
    const src = readFileSync(join(__dirname, f), "utf8");
    it(`${f}: sink routes non-string through canonicalJson + counts code points`, () => {
      // find each sink function body and assert it uses canonicalJson + [...s].length
      for (const name of fnNames) {
        const re = new RegExp(`function ${name}\\(val: unknown\\)[\\s\\S]*?\\n}`);
        const m = src.match(re);
        if (!m) continue;
        const body = m[0];
        expect(body).toContain("canonicalJson(val)");
        expect(body).toContain("[...s].length");
        expect(body).not.toContain("JSON.stringify(val)");
        // bare String(val) must not feed the hash (canonicalJson subsumes it)
        expect(body).not.toMatch(/:\s*String\(val\)/);
      }
    });
  }
});
