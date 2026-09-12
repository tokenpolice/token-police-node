/**
 * Pure-function local enforcement evaluator.
 * Behavior — eval order and shapes — is kept in parity with the TokenPolice
 * Python SDK.
 */

import { isNoopReroute } from "./rerouteNoop";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export interface Decision {
  status: "allowed" | "blocked" | "rerouted";
  rule_id: string | null;
  mode: "enforce" | "dry_run";
  reroute: { from: Any; to: Any } | null;
  reason?: string;
  /** Human-readable rule name from the directive pack; omit when absent. */
  rule_name?: string;
}

export interface Observation {
  rule_id: string;
  outcome: "would_block" | "would_reroute" | "reroute_rejected";
  mode: "enforce" | "dry_run";
  rejection_reason?: string;
  reroute?: { from: Any; to: Any };
  /** Human-readable rule name from the directive pack; omit when absent. */
  rule_name?: string;
}

/**
 * `{ rule_name }` when the directive carries a name, else `{}` — spread into an
 * observation so the audit/routing feeds show a label instead of the rule UUID.
 * Omit-when-absent keeps the payload shape unchanged for unnamed directives.
 */
function ruleNameOf(d: Any): { rule_name?: string } {
  return typeof d?.name === "string" && d.name ? { rule_name: d.name as string } : {};
}

export interface EvalResult {
  decision: Decision;
  observations: Observation[];
  /**
   * @internal Metadata, never part of the decision: an entity-list-driven
   * directive's selector matched this call but the computed group tag was
   * absent from its streamed `entities` set — i.e. an `entity_blocked` /
   * `entity_rerouted` delta the SDK never received would have flipped this
   * result. Read only by the enforcer's stale-stream gate. Present (true) only
   * when it happened; the decision object itself is unchanged either way.
   */
  armableMiss?: boolean;
}

function payloadField(payload: Any, field: string): Any {
  if (!field) return null;
  // Dotted paths (e.g. "intent.kind") for modality-aware rules. Falls back
  // to flat-key + metadata bag so legacy rules stay valid.
  if (field.indexOf(".") !== -1) {
    let cur: Any = payload;
    for (const part of field.split(".")) {
      if (cur == null || typeof cur !== "object") {
        cur = null;
        break;
      }
      cur = cur[part];
    }
    if (cur != null) return cur;
    const md = payload?.metadata;
    if (md && typeof md === "object" && field in md) return md[field];
    return null;
  }
  if (payload && payload[field] !== undefined) return payload[field];
  const md = payload?.metadata;
  if (md && typeof md === "object" && field in md) return md[field];
  return null;
}

// One leaf condition { field, operator, value }. No field => matches everything.
// Full operator set; unknown operator fails closed.
function matchesLeaf(payload: Any, cond: Any): boolean {
  if (!cond) return true;
  const op = cond.operator || cond.op;
  const field = cond.field;
  const value = cond.value;
  if (!field) return true;
  const v = payloadField(payload, field);
  if (op === "EQ") return v === value;
  if (op === "NEQ") return v !== value;
  // EXISTS treats "" as absent to match the SQL side (col != '').
  if (op === "EXISTS") return v !== null && v !== undefined && v !== "";
  if (op === "CONTAINS") return v !== null && v !== undefined && String(v).includes(String(value ?? ""));
  if (op === "IN") return Array.isArray(value) && value.includes(v);
  return false;
}

// matchCondition selector: empty => match all; composite { combinator,
// conditions } => AND/OR (empty AND true, empty OR false); else a bare leaf.
// Semantics are pinned to the server-side rule matcher, so a locally-decided
// call and a remote /check decision always agree. Covered by localMatcher.test.ts.
export function matchesCondition(payload: Any, match: Any): boolean {
  if (!match) return true;
  if (Array.isArray(match.conditions)) {
    const conds = match.conditions as Any[];
    const combinator = String(match.combinator || "AND").toUpperCase();
    return combinator === "OR"
      ? conds.some((c) => matchesLeaf(payload, c))
      : conds.every((c) => matchesLeaf(payload, c));
  }
  return matchesLeaf(payload, match);
}

export function generateGroupByTag(payload: Any, groupBy: Any): string {
  if (!groupBy || !Array.isArray(groupBy) || groupBy.length === 0) return "global";
  const parts: string[] = [];
  for (const f of groupBy) {
    const v = payloadField(payload, f);
    // Group-tag contract (must match the server-side evaluation exactly): any
    // JS-falsy value (null/undefined/""/0/-0/false/NaN) collapses to the
    // "unknown" sentinel; kept values are String()-coerced and joined with "_".
    // Covered by groupByTag.test.ts.
    parts.push(v ? String(v) : "unknown");
  }
  return parts.join("_");
}

// ── Effective provider — pure trim + lowercase of the call's provider ──
// Alias slugs a price source or an SDK instrumentor may emit, mapped onto the
// canonical serving-provider slug the runtime identity layer produces. Kept
// inline (production code never reads the shared test-only contract JSON); mirror
// the published provider-slug `aliases` table when it changes so both sides of a
// provider comparison canonicalize identically. Kept byte-for-byte in sync with
// the Python SDK's `_PROVIDER_ALIASES`.
const _PROVIDER_ALIASES: Record<string, string> = {
  together_ai: "together",
  togetherai: "together",
  fireworks_ai: "fireworks",
  vertex_ai: "vertex-ai",
  "vertex_ai-language-models": "vertex-ai",
  google_vertexai: "vertex-ai",
  azure: "azure-openai",
  azure_text: "azure-openai",
  azure_ai: "azure-ai",
  gemini: "google",
  google_genai: "google",
  "x-ai": "xai",
  grok: "xai",
  moonshotai: "moonshot",
  minimaxi: "minimax",
  amazon: "bedrock",
  aws: "bedrock",
  cohere_chat: "cohere",
  huggingface_free_tier: "huggingface",
  hf: "huggingface",
  openrouter_byok: "openrouter",
  zhipuai: "zhipu",
  zai: "zhipu",
};

// Runtime-only provider aliases (not published provider slugs).
// Keep outside _PROVIDER_ALIASES so the published provider-slug table parity stays intact.
const _RUNTIME_ONLY_PROVIDER_ALIASES: Record<string, string> = {
  // SDK Responses-API pseudo-provider: parse/shape only; serving identity is openai.
  openai_responses: "openai",
};

export function effectiveProvider(provider: string): string {
  const slug = (provider || "").toLowerCase().trim();
  // Canonicalize alias slugs so equivalent providers (e.g. together_ai vs
  // together) compare equal regardless of which side named the alias.
  return _PROVIDER_ALIASES[slug] ?? _RUNTIME_ONLY_PROVIDER_ALIASES[slug] ?? slug;
}

function buildPayload(session: Any, observed: Any): Any {
  let md: Any = {};
  try {
    if (session?.metadata) md = { ...session.metadata };
  } catch { md = {}; }
  const userId = session?.user_id ?? session?.userId ?? null;
  const paidPlan = session?.paid_plan ?? session?.paidPlan ?? null;
  if (paidPlan && !("paid_plan" in md)) md.paid_plan = paidPlan;
  const intent = (observed && typeof observed === "object" && observed.intent) || {};
  const modality =
    (observed && typeof observed === "object" && observed.modality) ||
    (intent && typeof intent === "object" && (intent as Any).kind) ||
    "chat";
  if (modality && md && typeof md === "object" && !("modality" in md)) md.modality = modality;
  // Payload field order is a pinned contract with the server-side evaluator:
  // `end_user_id` and `metadata` sit before the `...md` spread (metadata keys
  // may override them); every other canonical field follows the spread
  // (canonical value wins). This keeps the local fast path and the remote
  // /check path resolving reserved keys identically. `operation` always equals
  // the resolved `modality` and sits after the spread, so a metadata key named
  // "operation" can never override it. Covered by sessionLocalEval.test.ts.
  const payload: Any = {
    end_user_id: userId ?? "anonymous",
    metadata: md,
    ...md,
    paid_plan: paidPlan ?? "free",
    user_id: userId ?? "anonymous",
    model: observed?.model ?? "",
    // Serving-provider identity (not the internal parse pseudo-provider). Aligns
    // groupBy/match tags with server-side entity tag arming on /log.
    provider: effectiveProvider(String(observed?.provider ?? "")),
    trace_id: observed?.trace_id ?? "",
    // session_id lets the local fast path compute a per-session group tag and
    // consult the directive's entities set, so allowed sessions still skip /check.
    session_id: session?.sessionId ?? session?.session_id ?? null,
    modality,
    operation: modality,
    intent: intent || {},
  };
  // Not a matchable selector field — SDK-only signal for the REROUTE
  // guard when the client base_url host is unrecognized.
  if (observed && typeof observed === "object" && observed.serving_unverified === true) {
    payload.serving_unverified = true;
  }
  return payload;
}

function allowed(): Decision {
  return { status: "allowed", rule_id: null, mode: "enforce", reroute: null };
}

export function evaluate(pack: Any, session: Any, observed: Any, forceShadow: boolean = false): EvalResult {
  try {
    return evaluateInner(pack, session, observed, forceShadow);
  } catch {
    return { decision: allowed(), observations: [] };
  }
}

function evaluateInner(pack: Any, session: Any, observed: Any, forceShadow: boolean): EvalResult {
  const observations: Observation[] = [];
  if (!pack || typeof pack !== "object") return { decision: allowed(), observations };

  const loopSet = new Set<string>(Array.isArray(pack.loop_blocks) ? pack.loop_blocks : []);
  const traceId = observed?.trace_id || "";
  if (traceId && loopSet.has(traceId)) {
    return {
      decision: { status: "blocked", rule_id: null, mode: "enforce", reroute: null, reason: "loop_detected" },
      observations,
    };
  }

  if (!Array.isArray(pack.directives)) return { decision: allowed(), observations };
  const directives = (pack.directives as Any[])
    .filter((d) => d && typeof d === "object")
    .slice()
    .sort((a, b) => {
      const pa = typeof a.priority === "number" ? a.priority : 100;
      const pb = typeof b.priority === "number" ? b.priority : 100;
      if (pa !== pb) return pa - pb;
      // Tie-break by id ascending using JS `<`/`>` — a UTF-16 code-unit compare
      // that equals the Python SDK's code-point order over the lowercase-
      // alphanumeric CUID id space, matching the server-side tiebreak. Not
      // localeCompare (locale collation is case-insensitive and can disagree
      // with code-point order on mixed-case/digit ids).
      const ai = String(a.id || ""), bi = String(b.id || "");
      return ai < bi ? -1 : ai > bi ? 1 : 0;
    });

  const payload = buildPayload(session, observed || {});
  const obsProvider = effectiveProvider(String(payload.provider || ""));

  // PASS 0: UNCONDITIONAL_BLOCK
  for (const d of directives) {
    if (d.kind !== "UNCONDITIONAL_BLOCK") continue;
    const sel = d.selector || {};
    if (!matchesCondition(payload, sel.match)) continue;
    if (forceShadow || d.mode === "dry_run") {
      observations.push({ rule_id: d.id, outcome: "would_block", mode: "dry_run", ...ruleNameOf(d) });
      continue;
    }
    return { decision: { status: "blocked", rule_id: d.id, mode: "enforce", reroute: null }, observations };
  }

  // Set when an entity-gated directive matched but its tag was not (yet) armed
  // locally — see EvalResult.armableMiss. Never affects the decision below.
  // Only directives that would ENFORCE if armed count: the shadow predicate
  // below is the same `forceShadow || d.mode === "dry_run"` each pass uses to
  // decide observe-vs-act, so a dry_run entity directive (which /check refuses
  // to block on anyway) can never trigger a re-verify round-trip.
  let armableMiss = false;
  const wouldEnforce = (d: Any): boolean => !forceShadow && d.mode !== "dry_run";

  // PASS 1: ENTITY_BLOCK
  for (const d of directives) {
    if (d.kind !== "ENTITY_BLOCK") continue;
    const sel = d.selector || {};
    if (!matchesCondition(payload, sel.match)) continue;
    const tag = generateGroupByTag(payload, sel.group_by || []);
    const entities = new Set<string>(Array.isArray(d.entities) ? d.entities : []);
    if (!entities.has(tag)) { if (wouldEnforce(d)) armableMiss = true; continue; }
    if (forceShadow || d.mode === "dry_run") {
      observations.push({ rule_id: d.id, outcome: "would_block", mode: "dry_run", ...ruleNameOf(d) });
      continue;
    }
    return { decision: { status: "blocked", rule_id: d.id, mode: "enforce", reroute: null }, observations };
  }

  // PASS 2: REROUTE
  for (const d of directives) {
    if (d.kind !== "REROUTE") continue;
    const sel = d.selector || {};
    if (!matchesCondition(payload, sel.match)) continue;
    const reroute = d.reroute || {};
    const target = reroute.to || {};
    if (!target.provider || !target.model) continue;
    if (d.entities != null) {
      const tag = generateGroupByTag(payload, sel.group_by || []);
      const entities = new Set<string>(Array.isArray(d.entities) ? d.entities : []);
      if (!entities.has(tag)) { if (wouldEnforce(d)) armableMiss = true; continue; }
    }
    // Cross-provider reroute is rejected (same-provider only) — normalize the directive's
    // target provider through the SAME helper as obsProvider so alias slugs
    // (e.g. together_ai vs together) compare equal instead of spuriously
    // rejecting an equivalent-provider reroute.
    // Serving_unverified (unrecognized custom base_url) also refuses —
    // we cannot prove the target model is servable; provider field still holds
    // the module slug for matchConditions/groupBy (check/log mirror).
    // serving_unverified outranks cross_provider_unsupported, mirroring
    // _applyReroute's precedence — State A/B selection is invisible to the
    // operator, so both must report the same reason for the same refusal.
    if (
      payload.serving_unverified === true ||
      effectiveProvider(String(target.provider)) !== obsProvider
    ) {
      observations.push({
        rule_id: d.id,
        outcome: "reroute_rejected",
        mode: d.mode || "enforce",
        rejection_reason: payload.serving_unverified === true
          ? "serving_unverified"
          : "cross_provider_unsupported",
        reroute: { from: { provider: obsProvider, model: payload.model }, to: target },
        ...ruleNameOf(d),
      });
      continue;
    }
    // Target resolves to the model already requested (alias vs its dated
    // snapshot) — nothing to reroute, so no decision and no observation either
    // (a would_reroute here is just as phantom as an applied one).
    if (typeof payload.model === "string" && payload.model &&
        isNoopReroute(payload.model, target.model)) {
      continue;
    }
    if (forceShadow || d.mode === "dry_run") {
      observations.push({
        rule_id: d.id, outcome: "would_reroute", mode: "dry_run",
        reroute: { from: { provider: obsProvider, model: payload.model }, to: target },
        ...ruleNameOf(d),
      });
      continue;
    }
    // Thread directive `name` into the decision so the synthetic
    // /check-shaped reroute can stamp `_tp_routing.rule_name`.
    const ruleName =
      typeof d.name === "string" && d.name ? (d.name as string) : undefined;
    return {
      decision: {
        status: "rerouted", rule_id: d.id, mode: "enforce",
        reroute: { from: { provider: obsProvider, model: payload.model }, to: target },
        ...(ruleName ? { rule_name: ruleName } : {}),
      },
      observations,
    };
  }

  // Spread-when-true so an allow with no armable miss keeps its exact prior shape.
  return { decision: allowed(), observations, ...(armableMiss ? { armableMiss: true } : {}) };
}
