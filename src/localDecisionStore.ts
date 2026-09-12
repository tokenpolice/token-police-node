/**
 * Per-call keyed store for the applied `local_decision` audit stash.
 *
 * WHY THIS EXISTS: `local_decision` is the SOLE provenance the collector turns
 * into a `REQUEST_REROUTED` audit event — there is no server-side fallback. It
 * used to live in ONE flat slot on the session (`session._local_decision`), so
 * N concurrent calls overwrote each other (N−1 decisions lost) and the first
 * call to finish drained the survivor — one applied-reroute event per burst,
 * stamped on an arbitrary row with an arbitrary sibling's rule/from/to. Worse
 * than the under-count: in a heterogeneous burst an un-rerouted call's /log
 * could carry a sibling's decision (a false-positive reroute event).
 *
 * The store mirrors the observation queue's per-call keying (PR #345) — same
 * key source (the call's obs key), same 300s window — with three deliberate
 * differences, because for a local decision MIS-ATTRIBUTION is worse than
 * stranding:
 *   1. a drain claims AT MOST ONE record (never "everything claimable");
 *   2. the store never honors the obs queue's greedy drain-all window — a
 *      terminal flush must not stamp an orphaned reroute onto an arbitrary row;
 *   3. an expired entry is SWEPT, not claimed — the obs queue ships its stale
 *      orphans late on an arbitrary /log; this store drops them.
 *
 * No lock here, unlike the Python twin (`enforcer._local_decisions_lock`):
 * Node's event loop makes every read-then-splice below atomic. The Python twin
 * also carries a `_drop_local_decision` helper for its `_rebuild_after_reroute`
 * path (a stream manager that could not be rebuilt around the swap); Node has
 * no such path, so no drop helper exists here.
 *
 * SCOPE: entries live on the SESSION object (not a module-global) so the
 * store's untagged-claim fallback can never reach across sessions — a scoping
 * property the old single slot had and that this fix must not weaken. This
 * module is a leaf (no imports): both `enforcer.ts` and `telemetry.ts` use it,
 * and it deliberately does NOT extend `state.ts`'s export surface (several
 * vitest `vi.mock("../src/state")` factories are non-spread and throw on any
 * export they don't declare).
 *
 * GOLDEN RULE: every function here swallows its own failures. A failure
 * degrades to "no decision" (the event is lost, exactly as it is lost today
 * when a drain misses) — never to a wrong attribution, and never to a throw
 * into the customer's call.
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

export type LocalDecision = Record<string, unknown>;

type LocalDecisionEntry = {
  ld: LocalDecision;
  /** The owning call's obs key; null when no scope was active (degraded). */
  key: string | null;
  /** Date.now() at stash time; 0 when the clock read failed. */
  ts: number;
};

/**
 * EXPIRY window for stashed-but-unclaimed decisions. Deliberately the same
 * 300s the observation queue uses (`state.OBS_STALE_MS`) — the window must
 * exceed the longest plausible LLM call duration, or a concurrent /log could
 * retire a slow/streaming call's decision while that call is still running.
 * Duplicated as a literal rather than imported from `state.ts` so a partial
 * `vi.mock("../src/state")` (which throws on undeclared exports) can never
 * break the store. Keep the two in sync.
 *
 * The obs queue CLAIMS its stale entries (for observations, late delivery
 * beats loss). This store deliberately does the opposite: an expired entry is
 * SWEPT — dropped silently on the next stash/claim — and never handed to a
 * stranger's row, because a stranded decision costs one missing event while a
 * misattributed one is a false-positive REQUEST_REROUTED with another rule's
 * from/to. Cost: a stream that runs longer than 300s loses its event.
 */
export const LOCAL_DECISION_STALE_MS = 300_000;

/**
 * Hard cap on entries per session (drop-oldest). Bounds the list even if the
 * clock is unreadable and the sweep below can never run.
 */
const LOCAL_DECISION_CAP = 64;

/** The session's entry list, or null when absent/corrupt. Never throws. */
function _entries(session: Any): LocalDecisionEntry[] | null {
  try {
    if (!session) return null;
    const list = session._local_decisions;
    return Array.isArray(list) ? (list as LocalDecisionEntry[]) : null;
  } catch {
    return null;
  }
}

/**
 * Drop expired entries in place. Runs at the top of stash AND claim, so an
 * expired decision dies silently instead of riding a later, unrelated /log
 * (see the constant above). A clock that cannot be read sweeps nothing — the
 * cap still bounds the list. Never throws.
 */
function _sweep(list: LocalDecisionEntry[]): void {
  try {
    const now = Date.now();
    if (!(now > 0)) return;
    for (let i = list.length - 1; i >= 0; i -= 1) {
      const e = list[i];
      if (!e || now - e.ts > LOCAL_DECISION_STALE_MS) list.splice(i, 1);
    }
  } catch {
    // fail-open: an unswept list is still bounded by LOCAL_DECISION_CAP
  }
}

/**
 * Stash one call's applied decision under its obs key.
 *
 * Same key ⇒ REPLACE: repeated stashes within one logical call (a check that
 * re-decides, or an outer+inner wrapper pair sharing one key) keep today's
 * last-wins semantics and never leave a duplicate behind for a sibling to
 * claim. A null key (no obs scope — degraded/mocked paths) always appends:
 * two untagged entries may belong to two different calls, and dropping one
 * would lose an event the old slot would also have lost.
 */
export function stashLocalDecision(
  session: Any,
  ld: LocalDecision,
  key: string | null,
): void {
  try {
    if (!session || !ld) return;
    let list = _entries(session);
    if (!list) {
      list = [];
      session._local_decisions = list;
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
          list[i] = { ld, key, ts };
          return;
        }
      }
    }
    list.push({ ld, key, ts });
    while (list.length > LOCAL_DECISION_CAP) list.shift();
  } catch {
    // fail-open: a decision we could not stash is simply not audited
  }
}

/**
 * Claim AT MOST ONE decision for a /log about to be dispatched.
 *
 * Expired entries are swept first (see `_sweep`), so what remains is only live
 * decisions. Precedence — own key → untagged:
 *   1. the entry stashed under `key` (this call's own decision; the newest
 *      when several share the key, and every same-key entry is consumed so a
 *      superseded one can never resurface);
 *   2. the newest untagged entry (degraded paths with no obs scope — this
 *      reproduces the old single-slot behavior exactly).
 * Anything else stays put for its OWN call's drain — that is the whole fix.
 * There is deliberately NO stale-claim arm: an orphan is swept, never attached
 * to a stranger's row.
 *
 * Deliberately ignores the observation queue's greedy drain-all window: a
 * terminal flush must never stamp an orphaned reroute onto an unrelated row.
 */
export function claimLocalDecision(
  session: Any,
  key: string | null,
): LocalDecision | undefined {
  try {
    const list = _entries(session);
    if (!list || list.length === 0) return undefined;
    _sweep(list);

    if (key) {
      let found: LocalDecision | undefined;
      for (let i = list.length - 1; i >= 0; i -= 1) {
        const e = list[i];
        if (e && e.key === key) {
          // Newest wins; every same-key entry is removed either way.
          if (found === undefined) found = e.ld;
          list.splice(i, 1);
        }
      }
      if (found !== undefined) return found;
    }

    for (let i = list.length - 1; i >= 0; i -= 1) {
      const e = list[i];
      if (e && e.key === null) {
        list.splice(i, 1);
        return e.ld;
      }
    }

    return undefined;
  } catch {
    return undefined;
  }
}
