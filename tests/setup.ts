/**
 * Suite-wide safety net: the SDK's own tests must never reach a real collector.
 *
 * Why this file exists
 * --------------------
 * `init()` resolves `baseUrl` from `TOKENPOLICE_BASE_URL` and falls back to the
 * PRODUCTION collector (`client.ts`). With `firewall: 'enforce'`/`'dry_run'` in
 * daemon deployment it also starts a real SSE reader against that URL. So any
 * test that inits without an explicit `baseUrl` opened a live connection to
 * prod from the test runner.
 *
 * That is a race, not just untidiness. The bogus `tp_sk_...` keys tests use earn
 * an HTTP 401, and a terminal status makes the reader call
 * `state.invalidatePack()` (`stream.ts`) — correct product behaviour, since a
 * stale pack must never outlive a revoked key. If that lands between a test
 * seeding a snapshot and the enforcer's cache-health read, local rule
 * evaluation is skipped for that call and any assertion about a locally
 * evaluated REROUTE/BLOCK fails.
 *
 * The Python SDK flaked exactly this way in the nightly provider-drift run on
 * 2026-09-14. Node shares the default, the stream and the invalidate-on-401
 * path, so it carries the same landmine; this file defuses both suites the same
 * way. Before the collector was deployed the tests were safe by accident — the
 * hostname did not resolve, and a connect failure is TRANSIENT, which does not
 * invalidate the pack. Going live turned a no-op into a race.
 *
 * Two independent protections, because either alone can be defeated:
 *   1. Default every test's collector to a closed loopback port, restoring the
 *      old "connect fails, transient, pack untouched" behaviour deterministically.
 *   2. Refuse, and loudly fail the test, if a TokenPolice control-plane request
 *      is aimed at anything but loopback anyway — e.g. a future test that
 *      hardcodes a production URL. Without this, protection 1 silently stops
 *      covering new tests and the flake comes back.
 *
 * The fix lives here and NOT in `src/`: invalidating the pack on a terminal
 * stream status is the behaviour we want to ship. The tests were wrong.
 */
import { beforeEach, afterEach } from "vitest";

/**
 * Discard port on loopback. Nothing listens, so a connect is refused
 * immediately — fast, offline, and (unlike a 401 from a real collector) a
 * TRANSIENT failure that leaves the rule pack alone. Tests that want a
 * different URL still pass `baseUrl` explicitly; that wins over the env var.
 */
export const DEAD_COLLECTOR_URL = "http://127.0.0.1:9";

/**
 * Hosts a test may send TokenPolice control-plane traffic to. Several suites
 * stand up their own stub collector on loopback.
 */
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "localhost"]);

/**
 * Only TokenPolice's own endpoints are policed. Provider tests legitimately
 * point mock fetches at `api.openai.com`, `api.anthropic.com` and friends, and
 * must pass through untouched.
 */
const GUARD_PATH_PREFIX = "/v1/guard/";

// Set at module scope as well as per-test: setup files are evaluated before the
// test file is imported, so this also covers a module-level `init()`.
process.env.TOKENPOLICE_BASE_URL = DEAD_COLLECTOR_URL;

let escapes: string[] = [];

function isEscaping(rawUrl: string): boolean {
  let url: URL;
  try {
    url = new URL(rawUrl);
  } catch {
    // A relative or unparseable URL cannot leave the machine on its own.
    return false;
  }
  if (!url.pathname.startsWith(GUARD_PATH_PREFIX)) return false;
  // URL strips the brackets from an IPv6 hostname, so `::1` compares directly.
  return !LOOPBACK_HOSTS.has(url.hostname);
}

function urlOf(input: unknown): string {
  if (typeof input === "string") return input;
  if (input instanceof URL) return input.toString();
  const maybeRequest = input as { url?: unknown } | null;
  return typeof maybeRequest?.url === "string" ? maybeRequest.url : "";
}

let guardedFetch: typeof globalThis.fetch;
let previousFetch: typeof globalThis.fetch;

beforeEach(() => {
  process.env.TOKENPOLICE_BASE_URL = DEAD_COLLECTOR_URL;
  escapes = [];
  previousFetch = globalThis.fetch;
  guardedFetch = ((input: Parameters<typeof fetch>[0], init?: Parameters<typeof fetch>[1]) => {
    const url = urlOf(input);
    if (isEscaping(url)) {
      escapes.push(url);
      // Reject the way a closed port does, so the SDK's fail-open paths behave
      // as they would offline. The test still fails in afterEach — the SDK
      // swallows this, so a rejection alone would be invisible.
      return Promise.reject(
        new TypeError("blocked by tests/setup.ts: TokenPolice request left loopback"),
      );
    }
    return previousFetch(input, init);
  }) as typeof globalThis.fetch;
  globalThis.fetch = guardedFetch;
});

afterEach(() => {
  // Only restore if the test did not install its own fetch. Overwriting a test's
  // stub here would fight its own cleanup; a stub does not reach the network
  // anyway, so nothing is left unguarded that matters.
  if (globalThis.fetch === guardedFetch) globalThis.fetch = previousFetch;

  if (escapes.length > 0) {
    const list = [...new Set(escapes)].sort().join("\n  ");
    escapes = [];
    throw new Error(
      `TokenPolice request(s) addressed a non-loopback host during this test:\n  ${list}\n` +
        `Pass baseUrl to init(), or point it at ${DEAD_COLLECTOR_URL}. Tests must ` +
        `never reach a real collector (see the header comment in tests/setup.ts). ` +
        `Note: a stream left running by an EARLIER test can surface here instead.`,
    );
  }
});
