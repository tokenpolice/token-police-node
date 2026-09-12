/**
 * Deployment-mode detection for the Node SDK.
 *
 * Precedence (env vars → daemon) is kept in parity with the Python SDK.
 * Edge runtimes (Cloudflare Workers, Vercel Edge, Deno) detect via
 * runtime-specific globals as well as env vars.
 */

const SERVERLESS_MARKERS = [
  "AWS_LAMBDA_FUNCTION_NAME",
  "LAMBDA_TASK_ROOT",
  "NETLIFY",
  "FUNCTIONS_WORKER_RUNTIME",
  "FUNCTION_TARGET",
];

const EDGE_MARKERS = ["CF_PAGES", "VERCEL_EDGE_REGION", "DENO_DEPLOY"];

export function detectDeploymentMode(): "daemon" | "serverless" | "edge" {
  try {
    // Edge runtimes expose a globalThis.EdgeRuntime sentinel.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    if (typeof (globalThis as any).EdgeRuntime !== "undefined") return "edge";
    for (const m of EDGE_MARKERS) {
      if (process.env[m]) return "edge";
    }
    for (const m of SERVERLESS_MARKERS) {
      if (process.env[m]) return "serverless";
    }
    return "daemon";
  } catch {
    return "daemon";
  }
}

export function resolveDeployment(explicit?: string): "daemon" | "serverless" | "edge" {
  if (!explicit || explicit === "auto") return detectDeploymentMode();
  if (explicit === "daemon" || explicit === "serverless" || explicit === "edge") return explicit;
  return detectDeploymentMode();
}
