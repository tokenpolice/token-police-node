/**
 * B4 — per-call keyed store for the applied-reroute provenance marker
 * (`metadata._tp_routing`).
 *
 * Bug: `_applyReroute` wrote `_tp_routing` onto `session.metadata`, and every
 * subsequent row-emission site copied session metadata WHOLESALE into its
 * `/log` payload — so one rerouted call's marker rode EVERY later row of that
 * session (tool rows, agent/chain structural rows, unrelated sibling calls'
 * rows, even other modalities).
 *
 * Fix: `session._tp_routing_markers` — a bounded, session-scoped, per-call
 * keyed list — via `stashRoutingMarker` / `peekRoutingMarker` in
 * `src/routingMarkerStore.ts`, PLUS `copySessionMetadata` (strip on every
 * copy) / `stampRoutingMarker` (re-add ONLY on the owning model-call row) /
 * `rowMetadataFromSession` (the two call sites that used to hand
 * `session.metadata` to `tp.log` by reference).
 *
 * Deliberately PEEK-MANY (never destructive) — unlike B3's `local_decision`
 * claim-once store — because one rerouted call can emit several of its OWN
 * rows (stream-failure + framework manual row, batch rows, ...) and every one
 * of them must carry the marker; `_tp_routing` has no server-side reader, so
 * duplication across a call's own rows is correct, not a hazard.
 *
 * This file covers the MEDIUM "store unit" tier:
 *   (1) own-key hit / other-key miss / keyed invisible to keyless / untagged
 *       only reachable by a keyless peek;
 *   (2) peek is non-destructive;
 *   (3) same-key stash REPLACES;
 *   (4) FIFO cap 64 evicts oldest;
 *   (5) >300s expiry, sweeping both on stash and on peek;
 *   (6) fail-open on hostile/garbage inputs;
 *   (7) session.metadata is never touched by any helper here — entries live
 *       on a SEPARATE `_tp_routing_markers` list.
 * Plus direct coverage of `copySessionMetadata` / `stampRoutingMarker` /
 * `rowMetadataFromSession`, the three row-side helpers the row-scope tests
 * (`rerouteMarkerRowScope.test.ts`) exercise through the real SDK paths.
 */
import { describe, it, expect, afterEach, vi } from "vitest";
import {
  stashRoutingMarker,
  peekRoutingMarker,
  copySessionMetadata,
  stampRoutingMarker,
  rowMetadataFromSession,
  ROUTING_MARKER_STALE_MS,
  TP_ROUTING_KEY,
  TP_ROUTING_ATTR,
  type RoutingMarker,
} from "../src/routingMarkerStore";

type Entry = { routing: RoutingMarker; key: string | null; ts: number };

function entries(session: Record<string, unknown>): Entry[] {
  return (session._tp_routing_markers as Entry[]) ?? [];
}

afterEach(() => {
  vi.restoreAllMocks();
});

// ══════════════════════════════════════════════════════════════════════════
// (1) Attribution — exact match only, in both directions
// ══════════════════════════════════════════════════════════════════════════
describe("routingMarkerStore — attribution", () => {
  it("own-key peek hits; a different key misses entirely (no untagged fallback)", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "mine" }, "key-a");

    expect(peekRoutingMarker(session, "key-a")).toEqual({ id: "mine" });
    // Deliberately stricter than claimLocalDecision: a near-miss key gets
    // NOTHING, never a stranger's record.
    expect(peekRoutingMarker(session, "key-b")).toBeUndefined();
  });

  it("a keyed record is invisible to a keyless (untagged) peek", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "mine" }, "key-a");

    expect(peekRoutingMarker(session, null)).toBeUndefined();
  });

  it("an untagged record is reachable ONLY by a keyless peek, never by any keyed one", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "untagged" }, null);

    expect(peekRoutingMarker(session, "some-key")).toBeUndefined();
    expect(peekRoutingMarker(session, null)).toEqual({ id: "untagged" });
  });

  it("two different keyed records coexist and each is reachable only by its own key", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "a" }, "key-a");
    stashRoutingMarker(session, { id: "b" }, "key-b");

    expect(peekRoutingMarker(session, "key-a")).toEqual({ id: "a" });
    expect(peekRoutingMarker(session, "key-b")).toEqual({ id: "b" });
    expect(peekRoutingMarker(session, "key-c")).toBeUndefined();
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (2) Peek is non-destructive (peek-many)
// ══════════════════════════════════════════════════════════════════════════
describe("routingMarkerStore — peek-many (never removes)", () => {
  it("peeking the same key repeatedly returns the marker every time", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "mine" }, "key-a");

    expect(peekRoutingMarker(session, "key-a")).toEqual({ id: "mine" });
    expect(peekRoutingMarker(session, "key-a")).toEqual({ id: "mine" });
    expect(peekRoutingMarker(session, "key-a")).toEqual({ id: "mine" });
    expect(entries(session)).toHaveLength(1);
  });

  it("a sibling key's peek does not disturb another key's entry", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "mine" }, "key-a");
    stashRoutingMarker(session, { id: "sibling" }, "key-b");

    expect(peekRoutingMarker(session, "key-b")).toEqual({ id: "sibling" });
    expect(peekRoutingMarker(session, "key-a")).toEqual({ id: "mine" });
    expect(entries(session)).toHaveLength(2);
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (3) Same-key stash REPLACES, never appends
// ══════════════════════════════════════════════════════════════════════════
describe("routingMarkerStore — same-key replace", () => {
  it("re-stashing under the same key replaces the entry (last-wins)", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "first" }, "key-a");
    stashRoutingMarker(session, { id: "second" }, "key-a");

    expect(entries(session)).toHaveLength(1);
    expect(peekRoutingMarker(session, "key-a")).toEqual({ id: "second" });
  });

  it("a null key always APPENDS — two untagged entries belong to two different calls", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "a" }, null);
    stashRoutingMarker(session, { id: "b" }, null);

    expect(entries(session)).toHaveLength(2);
    // Newest untagged entry wins a keyless peek.
    expect(peekRoutingMarker(session, null)).toEqual({ id: "b" });
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (4) FIFO cap — 64 entries, drop-oldest
// ══════════════════════════════════════════════════════════════════════════
describe("routingMarkerStore — FIFO cap", () => {
  it("caps at 64 entries, dropping the oldest", () => {
    const session: Record<string, unknown> = {};
    const cap = 64; // ROUTING_MARKER_CAP in src/routingMarkerStore.ts (not exported)
    const total = cap + 6;
    for (let i = 0; i < total; i += 1) {
      stashRoutingMarker(session, { id: i }, `key-${i}`);
    }

    const list = entries(session);
    expect(list).toHaveLength(cap);
    const ids = list.map((e) => (e.routing as { id: number }).id);
    // Oldest 6 (0..5) evicted; 6..(total-1) survive, oldest-first.
    expect(ids).toEqual(Array.from({ length: cap }, (_, i) => total - cap + i));
    // The evicted keys are truly gone.
    expect(peekRoutingMarker(session, "key-0")).toBeUndefined();
    expect(peekRoutingMarker(session, `key-${total - 1}`)).toEqual({ id: total - 1 });
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (5) Expiry — 300s window, swept on both stash and peek, no stale-peek
// ══════════════════════════════════════════════════════════════════════════
describe("routingMarkerStore — 300s expiry (Date.now, as the store reads it)", () => {
  it("an entry older than ROUTING_MARKER_STALE_MS is unreachable, even by its OWN key", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "stale" }, "key-a");
    // Age it past the window by rewriting its stashed timestamp directly.
    entries(session)[0].ts -= ROUTING_MARKER_STALE_MS + 1;

    expect(peekRoutingMarker(session, "key-a")).toBeUndefined();
    // Swept off the list entirely (not just skipped).
    expect(entries(session)).toEqual([]);
  });

  it("an entry just under the window is still reachable", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "fresh" }, "key-a");
    entries(session)[0].ts -= ROUTING_MARKER_STALE_MS - 1000;

    expect(peekRoutingMarker(session, "key-a")).toEqual({ id: "fresh" });
  });

  it("stash sweeps an already-expired sibling entry", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "stale" }, "key-a");
    entries(session)[0].ts -= ROUTING_MARKER_STALE_MS + 1;

    stashRoutingMarker(session, { id: "fresh" }, "key-b");

    const ids = entries(session).map((e) => (e.routing as { id: string }).id);
    expect(ids).toEqual(["fresh"]);
  });

  it("an expired untagged entry never falls back to a keyless peek", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { id: "stale" }, null);
    entries(session)[0].ts -= ROUTING_MARKER_STALE_MS + 1;

    expect(peekRoutingMarker(session, null)).toBeUndefined();
    expect(entries(session)).toEqual([]);
  });

  it("using a mocked Date.now: entry expires exactly as the clock advances past 300s", () => {
    const session: Record<string, unknown> = {};
    let now = 1_000_000;
    const spy = vi.spyOn(Date, "now").mockImplementation(() => now);
    try {
      stashRoutingMarker(session, { id: "a" }, "key-a");
      now += ROUTING_MARKER_STALE_MS - 1;
      expect(peekRoutingMarker(session, "key-a")).toEqual({ id: "a" });
      now += 2; // now strictly past the window
      expect(peekRoutingMarker(session, "key-a")).toBeUndefined();
    } finally {
      spy.mockRestore();
    }
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (6) Golden rule: every helper is fail-open — never throws, on anything
// ══════════════════════════════════════════════════════════════════════════
describe("routingMarkerStore — fail-open on hostile inputs", () => {
  it("stashRoutingMarker never throws on a missing/null/undefined session", () => {
    expect(() => stashRoutingMarker(null, { id: 1 }, "k")).not.toThrow();
    expect(() => stashRoutingMarker(undefined, { id: 1 }, "k")).not.toThrow();
  });

  it("peekRoutingMarker never throws and returns undefined on a missing/null/undefined session", () => {
    expect(peekRoutingMarker(null, "k")).toBeUndefined();
    expect(peekRoutingMarker(undefined, "k")).toBeUndefined();
    const s: Record<string, unknown> = {};
    expect(peekRoutingMarker(s, "k")).toBeUndefined();
  });

  it("a corrupt (non-array) marker list degrades to no-op on peek and self-heals on stash", () => {
    const s: Record<string, unknown> = { _tp_routing_markers: "not-an-array" };
    expect(peekRoutingMarker(s, "k")).toBeUndefined();
    stashRoutingMarker(s, { id: 1 }, "k");
    expect(Array.isArray(s._tp_routing_markers)).toBe(true);
    expect(peekRoutingMarker(s, "k")).toEqual({ id: 1 });
  });

  it("a falsy routing value (null/undefined) is never stashed", () => {
    const s: Record<string, unknown> = {};
    stashRoutingMarker(s, null as unknown as RoutingMarker, "k");
    stashRoutingMarker(s, undefined as unknown as RoutingMarker, "k");
    expect(s._tp_routing_markers).toBeUndefined();
  });

  it("a session whose _tp_routing_markers getter throws degrades to undefined, never a crash", () => {
    const s: Record<string, unknown> = {};
    Object.defineProperty(s, "_tp_routing_markers", {
      get() {
        throw new Error("hostile getter");
      },
      configurable: true,
    });
    expect(() => peekRoutingMarker(s, "k")).not.toThrow();
    expect(peekRoutingMarker(s, "k")).toBeUndefined();
    expect(() => stashRoutingMarker(s, { id: 1 }, "k")).not.toThrow();
  });

  it("a Date.now() that throws still leaves stash/peek functional (sweep just no-ops)", () => {
    const spy = vi.spyOn(Date, "now").mockImplementation(() => {
      throw new Error("clock down");
    });
    try {
      const s: Record<string, unknown> = {};
      expect(() => stashRoutingMarker(s, { id: 1 }, "k")).not.toThrow();
      // ts falls back to 0 on a dead clock; the entry is still stashed.
      expect(entries(s)).toHaveLength(1);
      expect(() => peekRoutingMarker(s, "k")).not.toThrow();
    } finally {
      spy.mockRestore();
    }
  });
});

// ══════════════════════════════════════════════════════════════════════════
// (7) session.metadata is never mutated by any store/row helper
// ══════════════════════════════════════════════════════════════════════════
describe("routingMarkerStore — session.metadata is never touched", () => {
  it("stashRoutingMarker/peekRoutingMarker only ever write session._tp_routing_markers", () => {
    const session: Record<string, unknown> = {
      metadata: { _tp_routing: { rule_id: "r1" }, customer_key: "keep" },
    };
    const before = session.metadata;
    stashRoutingMarker(session, { rule_id: "r1", actual_model: "m2" }, "key-a");
    peekRoutingMarker(session, "key-a");
    peekRoutingMarker(session, "nope");

    expect(session.metadata).toBe(before); // same reference, untouched
    expect(session.metadata).toEqual({ _tp_routing: { rule_id: "r1" }, customer_key: "keep" });
  });

  it("copySessionMetadata never mutates the source session metadata object", () => {
    const src = { _tp_routing: { rule_id: "r1" }, customer_key: "keep" };
    const dest: Record<string, unknown> = {};
    copySessionMetadata(dest, src);

    expect(src).toEqual({ _tp_routing: { rule_id: "r1" }, customer_key: "keep" });
    expect(dest).toEqual({ customer_key: "keep" }); // _tp_routing stripped from the COPY
    expect(dest).not.toHaveProperty(TP_ROUTING_KEY);
  });

  it("rowMetadataFromSession builds a fresh object and never mutates session.metadata", () => {
    const session: Record<string, unknown> = {
      metadata: { _tp_routing: { rule_id: "r1" }, customer_key: "keep" },
    };
    const before = session.metadata;
    const out = rowMetadataFromSession(session, null);

    expect(session.metadata).toBe(before);
    expect(out).not.toBe(session.metadata); // fresh object, not the raw session ref
    expect(out).toEqual({ customer_key: "keep" }); // no key => no marker re-added
  });
});

// ══════════════════════════════════════════════════════════════════════════
// Row-side helpers: copySessionMetadata / stampRoutingMarker / rowMetadataFromSession
// ══════════════════════════════════════════════════════════════════════════
describe("copySessionMetadata", () => {
  it("copies every key except _tp_routing", () => {
    const dest: Record<string, unknown> = {};
    copySessionMetadata(dest, { a: 1, b: "x", _tp_routing: { rule_id: "r" } });
    expect(dest).toEqual({ a: 1, b: "x" });
  });

  it("no-op on a null/undefined dest (never throws, dest stays whatever it was)", () => {
    expect(() => copySessionMetadata(null as unknown as Record<string, unknown>, { a: 1 })).not.toThrow();
    expect(() => copySessionMetadata(undefined as unknown as Record<string, unknown>, { a: 1 })).not.toThrow();
  });

  it("no-op on missing sessionMetadata, dest left as-is", () => {
    const dest: Record<string, unknown> = { existing: true };
    copySessionMetadata(dest, null);
    copySessionMetadata(dest, undefined);
    expect(dest).toEqual({ existing: true });
  });

  it("an empty (but present) dest object IS written to — null-check, not truthiness", () => {
    // The store's own doc comment: `dest` is normally `{}` (a real object, not
    // falsy) — the guard must be `== null`, never `!dest`, or every row would
    // silently drop all customer metadata.
    const dest: Record<string, unknown> = {};
    copySessionMetadata(dest, { a: 1 });
    expect(dest).toEqual({ a: 1 });
  });
});

describe("stampRoutingMarker", () => {
  it("no-op when the store has nothing for this key", () => {
    const session: Record<string, unknown> = {};
    const dest: Record<string, unknown> = { keep: true };
    stampRoutingMarker(dest, session, "key-a");
    expect(dest).toEqual({ keep: true });
  });

  it("serialize=false (manual emitters): re-adds the RAW object", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { rule_id: "r1", actual_model: "m2" }, "key-a");
    const dest: Record<string, unknown> = {};
    stampRoutingMarker(dest, session, "key-a", false);
    expect(dest._tp_routing).toEqual({ rule_id: "r1", actual_model: "m2" });
    expect(typeof dest._tp_routing).toBe("object");
  });

  it("serialize=true (OTel paths): re-adds a JSON STRING", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { rule_id: "r1", actual_model: "m2" }, "key-a");
    const dest: Record<string, unknown> = {};
    stampRoutingMarker(dest, session, "key-a", true);
    expect(typeof dest._tp_routing).toBe("string");
    expect(JSON.parse(dest._tp_routing as string)).toEqual({ rule_id: "r1", actual_model: "m2" });
  });

  it("a wrong key never stamps — dest stays unmarked", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { rule_id: "r1" }, "key-a");
    const dest: Record<string, unknown> = {};
    stampRoutingMarker(dest, session, "key-b");
    expect(dest).not.toHaveProperty(TP_ROUTING_KEY);
  });

  it("an un-serializable routing object with serialize=true leaves the row unmarked, never throws", () => {
    const session: Record<string, unknown> = {};
    const circular: Record<string, unknown> = { rule_id: "r1" };
    circular.self = circular; // JSON.stringify throws on cycles
    stashRoutingMarker(session, circular, "key-a");
    const dest: Record<string, unknown> = {};
    expect(() => stampRoutingMarker(dest, session, "key-a", true)).not.toThrow();
    expect(dest).not.toHaveProperty(TP_ROUTING_KEY);
  });

  it("a null dest never throws", () => {
    const session: Record<string, unknown> = {};
    stashRoutingMarker(session, { rule_id: "r1" }, "key-a");
    expect(() => stampRoutingMarker(null as unknown as Record<string, unknown>, session, "key-a")).not.toThrow();
  });
});

describe("rowMetadataFromSession", () => {
  it("keyed row: builds metadata WITH the marker when the key matches", () => {
    const session: Record<string, unknown> = {
      metadata: { customer_key: "keep" },
    };
    stashRoutingMarker(session, { rule_id: "r1", actual_model: "m2" }, "key-a");
    const out = rowMetadataFromSession(session, "key-a");
    expect(out).toEqual({
      customer_key: "keep",
      _tp_routing: { rule_id: "r1", actual_model: "m2" },
    });
  });

  it("keyed row: builds metadata WITHOUT the marker when the key does not match", () => {
    const session: Record<string, unknown> = {
      metadata: { customer_key: "keep" },
    };
    stashRoutingMarker(session, { rule_id: "r1" }, "key-a");
    const out = rowMetadataFromSession(session, "key-b");
    expect(out).toEqual({ customer_key: "keep" });
  });

  it("undefined when session.metadata is missing/not an object", () => {
    expect(rowMetadataFromSession({}, "k")).toBeUndefined();
    expect(rowMetadataFromSession({ metadata: null }, "k")).toBeUndefined();
    expect(rowMetadataFromSession(null, "k")).toBeUndefined();
  });
});

describe("exported constants sanity", () => {
  it("TP_ROUTING_ATTR mirrors TP_ROUTING_KEY under the tp.meta. prefix", () => {
    expect(TP_ROUTING_ATTR).toBe(`tp.meta.${TP_ROUTING_KEY}`);
    expect(TP_ROUTING_KEY).toBe("_tp_routing");
  });

  it("ROUTING_MARKER_STALE_MS is 300s", () => {
    expect(ROUTING_MARKER_STALE_MS).toBe(300_000);
  });
});
