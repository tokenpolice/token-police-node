/**
 * No-op REROUTE detection.
 *
 * A REROUTE rule armed on an alias slug (e.g. `claude-haiku-4-5`) also matches
 * apps that pinned a dated snapshot of the SAME model
 * (`claude-haiku-4-5-20251001`). Applying the swap there changes nothing about
 * cost, floods reroute analytics with phantom REQUEST_REROUTED events, and
 * silently unpins the customer's snapshot. Such a directive is treated as
 * nothing-to-do on both enforcement paths (local evaluator + apply).
 *
 * Kept in parity with the server `/check` guard and the Python SDK.
 *
 * Bedrock note (`preferRequestedModel` only). A Bedrock model id is itself the
 * billing identity: `[<region>.]<vendor>.<model>[-v<n>[:<m>]]`. The provider
 * echoes only the bare `<model>` part on the streaming taps, so a call made
 * against `us.anthropic.claude-haiku-4-5-20251001-v1:0` comes back as
 * `claude-haiku-4-5-20251001` — the SAME family, which is why the echo is
 * treated as a family match and the requested id wins. The requested id must be
 * reported VERBATIM: a cross-region inference profile carries its own
 * region-specific rate (64 of 99 CRIS cards differ from their base card, from
 * -68.8% to +31.4%), so reporting the base id — or any normalized form — would
 * misprice the call in BOTH directions. The candidates built below exist solely
 * to decide family equality; none of them is ever reported or priced.
 */

// Exactly one trailing dated-snapshot suffix: `-20251001`, `-2024-08-06`,
// `@20240620`. Deliberately narrow — `-latest`, gemini's `-002` and Bedrock's
// `-v1:0` are NOT snapshot dates and must never be stripped.
const DATE_SUFFIX_RE = /(?:-20\d{6}|-20\d{2}-\d{2}-\d{2}|@20\d{6})$/;

/** Remove exactly one trailing dated-snapshot suffix; otherwise unchanged. */
export function stripDateSuffix(model: string): string {
  return model.replace(DATE_SUFFIX_RE, "");
}

/**
 * True when rerouting `requestedModel` to `targetModel` would change nothing.
 *
 * The date suffix is stripped from the REQUESTED side only: a target that names
 * an explicit snapshot is a deliberate pin, so alias → dated (and dated → a
 * *different* dated) stay genuine reroutes.
 */
export function isNoopReroute(requestedModel: unknown, targetModel: unknown): boolean {
  if (typeof requestedModel !== "string" || typeof targetModel !== "string") return false;
  const requested = requestedModel.trim().toLowerCase();
  const target = targetModel.trim().toLowerCase();
  if (!requested || !target) return false;
  if (requested === target) return true;
  return stripDateSuffix(requested) === target;
}

// Bedrock id shape, used ONLY to test family equality in
// `preferRequestedModel` — never to build a string that is reported or priced.
// Region prefix of a cross-region inference profile (CRIS).
// KEEP IN SYNC — this same region list is duplicated, with no shared contract
// and no parity test, at:
//     token-police-python/token_police/reroute_noop.py
//     collector-server/src/lib/cost-calculator.js  (CRIS region-strip tier)
// A new AWS region prefix added in only one of the three degrades silently:
// under-billed no-op reroutes here, $0 `unknown_model` rows in the collector.
const BEDROCK_REGION_PREFIX = /^(?:us-gov|us|eu|apac|global)\./;
// Vendor namespace every Bedrock model id carries after the optional region.
const BEDROCK_VENDOR_NS =
  /^(?:ai21|amazon|anthropic|cohere|deepseek|luma|meta|minimax|mistral|moonshotai|nvidia|openai|qwen|stability|twelvelabs|writer)\./;
// Bedrock's own version tag — model identity, not a date. Two real shapes:
// `-v1:0` / `-v0:2` (explicit `v`), and a BARE numeric tag `-1:0`
// (`us.openai.gpt-oss-120b-1:0`, live in the catalog today). So the `v` is
// optional WHEN a `:<n>` part is present; without a colon an explicit `v` is
// required. That second clause is the guard that keeps a dated snapshot suffix
// (`-20251001`) out — a bare `-\d+$` would eat it and corrupt the family gate.
const BEDROCK_VERSION_SUFFIX = /(?:-v?\d+:\d+|-v\d+)$/;

/**
 * The forms a provider may echo for a Bedrock id, peeled one wrapper at a time:
 * region-less, then vendor-less, then version-less. Empty when `model` is not
 * shaped like a Bedrock id — the vendor namespace is REQUIRED after the
 * optional region, which is what keeps the widening tightly shape-gated:
 * anything not unmistakably a Bedrock id gets no family widening at all.
 * `model` is the already trimmed+lowercased requested string.
 */
function bedrockFamilyCandidates(model: string): string[] {
  const noRegion = model.replace(BEDROCK_REGION_PREFIX, "");
  if (!BEDROCK_VENDOR_NS.test(noRegion)) return [];
  const noVendor = noRegion.replace(BEDROCK_VENDOR_NS, "");
  const noVersion = noVendor.replace(BEDROCK_VERSION_SUFFIX, "");
  // `model` itself is already covered by the exact-match branch above.
  return [noRegion, noVendor, noVersion].filter((c) => c && c !== model);
}

/**
 * Pick the model string to report when the response echoes a different
 * one than was requested.
 *
 * The auto-instrumented span path reports the (post-rewrite) REQUEST model
 * while the manual/streaming taps read the provider's echo, so one rerouted
 * model splits into two rows in cost-by-model. Canonical form is the requested
 * model — but only when the echo is the SAME family, i.e. the request plus a
 * dated snapshot suffix, or a Bedrock id whose echo dropped the region/vendor
 * wrapper. A genuinely different served model (OpenRouter auto-routing, a
 * gateway substitution) fails those gates and keeps its echo.
 */
export function preferRequestedModel(requestedModel: unknown, extractedModel: string): string {
  if (typeof requestedModel !== "string" || typeof extractedModel !== "string") return extractedModel;
  const requested = requestedModel.trim().toLowerCase();
  const extracted = extractedModel.trim().toLowerCase();
  if (!requested || !extracted) return extractedModel;
  if (requested === extracted) return extractedModel;
  if (stripDateSuffix(extracted) === requested) return requestedModel;
  // Bedrock: the echo is the requested id with one or more of its wrappers
  // (region, vendor, version tag) peeled off, ± a dated snapshot suffix. Same
  // family → report the requested id verbatim, because the region-qualified
  // profile id is what AWS actually bills.
  for (const candidate of bedrockFamilyCandidates(requested)) {
    if (candidate === extracted || stripDateSuffix(extracted) === candidate) return requestedModel;
  }
  return extractedModel;
}
