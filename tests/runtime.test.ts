/**
 * Unit tests for deployment-mode detection (src/runtime.ts).
 *
 * The broad `VERCEL` / `VERCEL_ENV` markers were removed from
 * SERVERLESS_MARKERS because Vercel sets them in EVERY deployment context
 * (including long-lived Node servers), so any such process was mis-forced to
 * "serverless" — silently disabling the SSE Decision-Pack stream. Real Vercel
 * serverless runs on AWS Lambda (AWS_LAMBDA_FUNCTION_NAME / LAMBDA_TASK_ROOT)
 * and Vercel Edge sets VERCEL_EDGE_REGION + the EdgeRuntime global, so genuine
 * serverless/edge stays correctly classified.
 *
 * Env hygiene: every case uses vi.stubEnv + afterEach(vi.unstubAllEnvs) so a
 * VERCEL var leaking from the CI runner cannot pollute cases or other suites.
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { detectDeploymentMode, resolveDeployment } from "../src/runtime";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

afterEach(() => {
  vi.unstubAllEnvs();
  // Clean up the edge sentinel global if a test set it.
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  delete (globalThis as any).EdgeRuntime;
});

/**
 * Neutralize any deployment markers inherited from the CI/host environment so
 * each test starts from a known-clean baseline before stubbing its own case.
 */
function clearAllMarkers() {
  const all = [
    "AWS_LAMBDA_FUNCTION_NAME",
    "LAMBDA_TASK_ROOT",
    "VERCEL",
    "VERCEL_ENV",
    "NETLIFY",
    "FUNCTIONS_WORKER_RUNTIME",
    "FUNCTION_TARGET",
    "K_SERVICE",
    "CF_PAGES",
    "VERCEL_EDGE_REGION",
    "DENO_DEPLOY",
  ];
  for (const m of all) vi.stubEnv(m, "");
}

describe("runtime: SERVERLESS_MARKERS source", () => {
  // (a) source/diff assertion — only the two VERCEL entries were removed,
  // the other six + order intact; EDGE_MARKERS untouched.
  test("VERCEL and VERCEL_ENV are gone from SERVERLESS_MARKERS; rest intact", () => {
    const src = fs.readFileSync(
      path.join(__dirname, "..", "src", "runtime.ts"),
      "utf8",
    );
    const block = src.slice(
      src.indexOf("const SERVERLESS_MARKERS"),
      src.indexOf("const EDGE_MARKERS"),
    );
    expect(block).not.toContain('"VERCEL"');
    expect(block).not.toContain('"VERCEL_ENV"');
    // K_SERVICE removed too (fires in long-lived Cloud Run *services*);
    // true GCP FaaS still detected via the retained FUNCTION_TARGET marker.
    expect(block).not.toContain('"K_SERVICE"');
    // Remaining five in exact prior order.
    expect(block).toContain('"AWS_LAMBDA_FUNCTION_NAME"');
    const order = [
      "AWS_LAMBDA_FUNCTION_NAME",
      "LAMBDA_TASK_ROOT",
      "NETLIFY",
      "FUNCTIONS_WORKER_RUNTIME",
      "FUNCTION_TARGET",
    ];
    const idxs = order.map((m) => block.indexOf(m));
    expect(idxs.every((i) => i >= 0)).toBe(true);
    expect(idxs).toEqual([...idxs].sort((a, b) => a - b));
    // EDGE_MARKERS untouched (still holds VERCEL_EDGE_REGION as a distinct string).
    expect(src).toContain(
      'const EDGE_MARKERS = ["CF_PAGES", "VERCEL_EDGE_REGION", "DENO_DEPLOY"]',
    );
  });
});

describe("runtime: detectDeploymentMode", () => {
  // (b) VERCEL-only → daemon
  test("env with ONLY VERCEL + VERCEL_ENV → daemon", () => {
    clearAllMarkers();
    vi.stubEnv("VERCEL", "1");
    vi.stubEnv("VERCEL_ENV", "production");
    expect(detectDeploymentMode()).toBe("daemon");
  });

  // (c) VERCEL + Lambda → serverless (genuine Vercel serverless via Lambda marker)
  test("VERCEL + AWS_LAMBDA_FUNCTION_NAME → serverless", () => {
    clearAllMarkers();
    vi.stubEnv("VERCEL", "1");
    vi.stubEnv("AWS_LAMBDA_FUNCTION_NAME", "my-fn");
    expect(detectDeploymentMode()).toBe("serverless");
  });

  // (c) Vercel Edge marker → edge
  test("VERCEL_EDGE_REGION → edge", () => {
    clearAllMarkers();
    vi.stubEnv("VERCEL", "1");
    vi.stubEnv("VERCEL_EDGE_REGION", "iad1");
    expect(detectDeploymentMode()).toBe("edge");
  });

  // (c) untouched serverless markers still work
  test("NETLIFY (no VERCEL) → serverless", () => {
    clearAllMarkers();
    vi.stubEnv("NETLIFY", "1");
    expect(detectDeploymentMode()).toBe("serverless");
  });

  // RED→GREEN pin: K_SERVICE alone means a long-lived Cloud Run *service*,
  // which should behave like a daemon (regain SSE), not serverless.
  test("K_SERVICE alone (Cloud Run service) → daemon", () => {
    clearAllMarkers();
    vi.stubEnv("K_SERVICE", "svc");
    expect(detectDeploymentMode()).toBe("daemon");
  });

  // Over-removal guard: a true Cloud Run *function* / Cloud Function sets
  // BOTH K_SERVICE and FUNCTION_TARGET — still classified serverless via the
  // retained FUNCTION_TARGET marker.
  test("K_SERVICE + FUNCTION_TARGET (Cloud Run function) → serverless", () => {
    clearAllMarkers();
    vi.stubEnv("K_SERVICE", "svc");
    vi.stubEnv("FUNCTION_TARGET", "handler");
    expect(detectDeploymentMode()).toBe("serverless");
  });

  // FaaS-intact: the retained FUNCTION_TARGET marker independently
  // classifies GCP Cloud Functions as serverless.
  test("FUNCTION_TARGET alone → serverless", () => {
    clearAllMarkers();
    vi.stubEnv("FUNCTION_TARGET", "handler");
    expect(detectDeploymentMode()).toBe("serverless");
  });

  // (c) bare non-cloud env → daemon
  test("bare non-cloud env → daemon", () => {
    clearAllMarkers();
    expect(detectDeploymentMode()).toBe("daemon");
  });

  // reviewer optional #1 — EdgeRuntime global sentinel → edge
  test("globalThis.EdgeRuntime sentinel → edge", () => {
    clearAllMarkers();
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (globalThis as any).EdgeRuntime = "edge-runtime";
    expect(detectDeploymentMode()).toBe("edge");
  });

  // reviewer optional #2 — edge takes precedence over serverless
  test("VERCEL_EDGE_REGION + NETLIFY → edge (edge over serverless)", () => {
    clearAllMarkers();
    vi.stubEnv("VERCEL_EDGE_REGION", "iad1");
    vi.stubEnv("NETLIFY", "1");
    expect(detectDeploymentMode()).toBe("edge");
  });

  // (e) no-throw: detection stays try/catch → daemon
  test("detection never throws", () => {
    clearAllMarkers();
    expect(() => detectDeploymentMode()).not.toThrow();
  });

  // explicit override still honored (resolveDeployment)
  test("explicit deployment override wins over auto-detect", () => {
    clearAllMarkers();
    vi.stubEnv("NETLIFY", "1"); // would auto-detect serverless
    expect(resolveDeployment("daemon")).toBe("daemon");
    expect(resolveDeployment("auto")).toBe("serverless");
  });
});
