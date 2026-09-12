/**
 * Per-call keyed store for the applied-reroute provenance marker
 * (`metadata._tp_routing`).
 *
 * WHY THIS EXISTS: `_applyReroute` records the swap it just made on the
 * SESSION (`session.metadata._tp_routing`), and session metadata is copied
 * into EVERY row the SDK emits from then on. So one rerouted call painted its
 * marker onto every later row of that session — tool spans (which execute no
 * model call at all), agent/chain anchors, and unrelated sibling calls in
 * another modality (a TTS row carrying an image rule's marker was observed
 * live). Per-call routing provenance in the trace display could not be
 * trusted.
 *
 * The session write itself is UNCHANGED — it is the /check-payload input that
 * rules may match on, and both SDKs' shapes are pinned by tests. What changed
 * is the ROW side: every copy of session metadata into a `/log` row now drops
 * `_tp_routing` (see `copySessionMetadata`), and only a MODEL-CALL row whose
 * own obs key matches the rerouted call's re-adds it from this store.
 *
 * Mirrors `localDecisionStore.ts` (the applied `local_decision` stash) — same
 * key source (the call's obs key), same 300s window, same session scoping,
 * same swallow-everything discipline — with ONE deliberate difference:
 *
 *   PEEK-MANY. A read NEVER removes the record. One rerouted call can emit
 *   several rows (a stream-failure row plus a framework manual row, batch
 *   result rows, ...) and every one of them is that call's row, so every one
 *   must carry the marker. `localDecisionStore` claims at most once because a
 *   `local_decision` is an AUDIT EVENT (emitting it twice double-counts);
 *   `_tp_routing` is pure display provenance with no server-side reader, so
 *   duplication across a call's own rows is correct, not a hazard.
 *
 * Attribution is EXACT — deliberately stricter than `claimLocalDecision`,
 * which falls back to an untagged record for a keyed claimant:
 *   - a keyed row matches only a record stashed under that same key;
 *   - a keyless row matches only an untagged record.
 * A near-miss therefore degrades to "no marker on the row" (what the row
 * looked like before any reroute existed) instead of stamping a stranger's
 * from->to onto it — the exact failure this fix exists to remove.
 *
 * NEVER filter these rows on model equality instead: the served model
 * legitimately differs from `actual_model` on gateway / dated-snapshot echoes
 * (HuggingFace, Anthropic), and live rows prove that filter wrong.
 *
 * No `drop` helper here, and no lock — both for the same reasons as the local
 * decision store: Node's event loop makes each read-then-splice below atomic,
 * and Node has no `_rebuild_after_reroute` equivalent (the Python twin's
 * stream-manager rebuild is the only path in either SDK that withdraws an
 * applied-reroute claim, so only the Python twin carries a drop).
 *
 * SCOPE: entries live on the SESSION object, not a module global, so the
 * untagged fallback can never reach across sessions. This module is a LEAF (no
 * imports) and deliberately does NOT extend `state.ts`'s export surface:
 * several vitest `vi.mock("../src/state")` factories are non-spread and throw
 * on any export they don't declare.
 *
 * GOLDEN RULE: every function here swallows its own failures and degrades to
 * "no marker" — never a throw into the customer's call, never a wrong
 * attribution.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export type RoutingMarker = Record<string, unknown>;

/** The metadata key carrying applied-reroute provenance. */
export const TP_ROUTING_KEY = "_tp_routing";

/** The span attribute the OTel paths mirror `TP_ROUTING_KEY` into. */
export const TP_ROUTING_ATTR = `tp.meta.${TP_ROUTING_KEY}`;

type RoutingMarkerEntry = {
  routing: RoutingMarker;
  /** The rerouted call's obs key; null when no scope was active (degraded). */
  key: string | null;
  /** Date.now() at stash time; 0 when the clock read failed. */
  ts: number;
};

/**
 * EXPIRY window for stashed markers. Same 300s as `localDecisionStore` (and
 * the observation queue's `state.OBS_STALE_MS`): the window must exceed the
 * longest plausible LLM call duration, or a slow/streaming call's own rows
 * would lose the marker while the call is still running. Duplicated as a
 * literal rather than imported so a partial `vi.mock("../src/state")` can
 * never break the store. Keep the three in sync.
 */
export const ROUTING_MARKER_STALE_MS = 300_000;

/** Hard cap on entries per session (drop-oldest), for a dead clock. */
const ROUTING_MARKER_CAP = 64;

/** The session's entry list, or null when absent/corrupt. Never throws. */
function _entries(session: Any): RoutingMarkerEntry[] | null {
  try {
    if (!session) return null;
    const list = session._tp_routing_markers;
    return Array.isArray(list) ? (list as RoutingMarkerEntry[]) : null;
  } catch {
    return null;
  }
}

/**
 * Drop expired entries in place. Runs at the top of stash AND peek, so an
 * expired marker dies silently instead of decorating a much later row of the
 * same session. A clock that cannot be read sweeps nothing — the cap still
 * bounds the list. Never throws.
 */
function _sweep(list: RoutingMarkerEntry[]): void {
  try {
    const now = Date.now();
    if (!(now > 0)) return;
    for (let i = list.length - 1; i >= 0; i -= 1) {
      const e = list[i];
      if (!e || now - e.ts > ROUTING_MARKER_STALE_MS) list.splice(i, 1);
    }
  } catch {
    // fail-open: an unswept list is still bounded by ROUTING_MARKER_CAP
  }
}

/**
 * Stash the marker `_applyReroute` just wrote onto the session, under the
 * rerouted call's own obs key.
 *
 * Same key => REPLACE (a call that re-decides keeps last-wins, and never
 * leaves a superseded marker behind). A null key always appends: two untagged
 * entries may belong to two different calls.
 */
export function stashRoutingMarker(
  session: Any,
  routing: RoutingMarker,
  key: string | null,
): void {
  try {
    if (!session || !routing) return;
    let list = _entries(session);
    if (!list) {
      list = [];
      session._tp_routing_markers = list;
    }
    _sweep(list);
    let ts = 0;
    try {
      ts = Date.now();
    } catch {
      ts = 0;
    }
    if (key) {
      for (let i = list.length - 1; i >= 0; i -= 1) {
        if (list[i] && list[i].key === key) {
          list[i] = { routing, key, ts };
          return;
        }
      }
    }
    list.push({ routing, key, ts });
    while (list.length > ROUTING_MARKER_CAP) list.shift();
  } catch {
    // fail-open: an un-stashed marker just means the row shows no reroute
  }
}

/**
 * Read (WITHOUT removing) the marker belonging to the call identified by
 * `key`. Returns undefined when this call was not the rerouted one.
 *
 * Exact match only, in both directions — see the header: a keyed row never
 * takes an untagged record, and a keyless row never takes a keyed one.
 */
export function peekRoutingMarker(
  session: Any,
  key: string | null,
): RoutingMarker | undefined {
  try {
    const list = _entries(session);
    if (!list || list.length === 0) return undefined;
    _sweep(list);
    const want = key || null;
    for (let i = list.length - 1; i >= 0; i -= 1) {
      const e = list[i];
      // Newest wins; the record STAYS for this call's other rows.
      if (e && e.key === want && e.routing) return e.routing;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

/**
 * Copy session metadata into a row's `/log` metadata, MINUS `_tp_routing`.
 *
 * Every row-metadata build goes through here (or through the equivalent
 * `tp.meta.*` read-back skip in `telemetry.ts`), so the marker is absent by
 * default and only the rerouted call's own model-call rows put it back via
 * `peekRoutingMarker`. Tool and agent/chain rows never do. Never throws; the
 * marker can only ever be omitted, never added, by a failure here.
 */
export function copySessionMetadata(
  dest: Record<string, unknown>,
  sessionMetadata: Record<string, unknown> | null | undefined,
): void {
  try {
    // Null-check, never a truthiness test, on `dest`: it is normally an empty
    // object here (falsy in the Python twin, where the same guard written as
    // `if not dest` silently drops every customer metadata key).
    if (dest == null || !sessionMetadata) return;
    for (const [k, v] of Object.entries(sessionMetadata)) {
      if (k === TP_ROUTING_KEY) continue;
      dest[k] = v;
    }
  } catch {
    // fail-open
  }
}

/**
 * Stamp a MODEL-CALL row's metadata with this call's own routing marker, if it
 * had one. No-op for every other row. `serialize` matches the OTel paths,
 * whose metadata values are JSON strings (the session-metadata -> span
 * attribute hop stringifies objects); manual emitters pass the object through
 * unchanged, exactly as they did before this fix.
 */
export function stampRoutingMarker(
  dest: Record<string, unknown>,
  session: Any,
  key: string | null,
  serialize: boolean = false,
): void {
  try {
    if (dest == null) return;
    const routing = peekRoutingMarker(session, key);
    if (!routing) return;
    if (!serialize) {
      dest[TP_ROUTING_KEY] = routing;
      return;
    }
    try {
      dest[TP_ROUTING_KEY] = JSON.stringify(routing);
    } catch {
      // un-serializable => leave the row unmarked rather than ship "[object Object]"
    }
  } catch {
    // fail-open
  }
}

/**
 * Build a row's `/log` metadata from the session's metadata alone (no
 * `workflow_name` / `session_id` injection) for the two call sites that used
 * to hand `session.metadata` itself to `tp.log` by reference — the
 * locally-blocked row and the call-failure row. Returns a fresh object; the
 * session's own metadata object is never mutated.
 *
 * `key` is the emitting call's obs key, so a failure row of the rerouted call
 * keeps its marker while a sibling's failure row does not.
 */
export function rowMetadataFromSession(
  session: Any,
  key: string | null,
): Record<string, unknown> | undefined {
  try {
    const src = session && session.metadata;
    if (!src || typeof src !== "object") return undefined;
    const out: Record<string, unknown> = {};
    copySessionMetadata(out, src as Record<string, unknown>);
    stampRoutingMarker(out, session, key);
    return out;
  } catch {
    // Never fall back to the raw session object here: that is exactly the
    // leak this store exists to close.
    return undefined;
  }
}
