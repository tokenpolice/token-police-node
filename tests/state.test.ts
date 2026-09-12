import { describe, it, expect, beforeEach, vi, afterEach } from "vitest";
import { readFileSync } from "fs";
import {
  resetPack,
  applySnapshot,
  applyDeltas,
  getPack,
  getPackVersion,
  isCacheHealthy,
  isPackExpired,
  setClient,
  getClient,
} from "../src/state";
import type { TokenPolice } from "../src/client";

// `directive_removed` index-miss must self-heal, not silently advance.
// The delta reducer indexes directives by `directive.id` but the
// `directive_removed` handler looks up by `op.rule_id`. A truthy rule_id that
// misses the index used to be a silent no-op WHILE the version still advanced —
// a removed rule would keep enforcing until the next full snapshot. The fix
// poisons the cache and returns false on that miss so the SSE caller
// re-snapshots. Falsy/empty rid stays a benign no-op (version advances).
// Sibling: token-police-python/tests/test_state.py pins the same scenarios.

const snapshotAB = () => ({
  version: 1,
  tenant_id: "t1",
  project_id: "p1",
  directives: [
    { id: "A", kind: "UNCONDITIONAL_BLOCK", mode: "enforce" },
    { id: "B", kind: "UNCONDITIONAL_BLOCK", mode: "enforce" },
  ],
  loop_blocks: [],
});

describe("Directive_removed index-miss self-heal (Node)", () => {
  beforeEach(() => {
    resetPack();
  });

  it("directive_removed truthy-rid MISS returns false and poisons the cache (#1,#4)", () => {
    expect(applySnapshot(snapshotAB())).toBe(true);
    const ok = applyDeltas([{ op: "directive_removed", rule_id: "ZZZ" }], 2);
    expect(ok).toBe(false);
    // cache poisoned → enforcer falls back to inline /check
    expect(getPack()).toBeNull();
    expect(isCacheHealthy()).toBe(false);
  });

  it("directive_removed MISS does NOT advance the version (#3)", () => {
    applySnapshot(snapshotAB());
    applyDeltas([{ op: "directive_removed", rule_id: "ZZZ" }], 2);
    expect(getPackVersion()).toBe(1); // NOT 2 — no silent advance
  });

  it("after a MISS a fresh snapshot rebuilds the pack whole (#5)", () => {
    applySnapshot(snapshotAB());
    applyDeltas([{ op: "directive_removed", rule_id: "ZZZ" }], 2);
    expect(getPack()).toBeNull();
    // self-heal: a fresh snapshot restores a healthy, whole pack
    expect(applySnapshot(snapshotAB())).toBe(true);
    const pack = getPack();
    expect(pack).not.toBeNull();
    expect(pack.directives.map((d: { id: string }) => d.id)).toEqual(["A", "B"]);
    expect(isCacheHealthy()).toBe(true);
  });

  it("directive_removed HIT path is byte-identical: removes A, advances, rebuilds index (#6)", () => {
    applySnapshot(snapshotAB());
    const ok = applyDeltas([{ op: "directive_removed", rule_id: "A" }], 2);
    expect(ok).toBe(true);
    expect(getPackVersion()).toBe(2);
    const pack = getPack();
    expect(pack.directives.map((d: { id: string }) => d.id)).toEqual(["B"]);
    expect(isCacheHealthy()).toBe(true);
    // index rebuilt correctly: a subsequent remove of the shifted entry works
    const ok2 = applyDeltas([{ op: "directive_removed", rule_id: "B" }], 3);
    expect(ok2).toBe(true);
    expect(getPack().directives).toEqual([]);
  });

  it("falsy/empty rid stays a benign no-op: version advances, cache healthy (#7)", () => {
    for (const badRid of ["", null, undefined]) {
      resetPack();
      applySnapshot(snapshotAB());
      const op: Record<string, unknown> = { op: "directive_removed" };
      if (badRid !== undefined) op.rule_id = badRid;
      const ok = applyDeltas([op], 2);
      expect(ok).toBe(true);
      expect(getPackVersion()).toBe(2);
      expect(getPack().directives.map((d: { id: string }) => d.id)).toEqual(["A", "B"]);
      expect(isCacheHealthy()).toBe(true);
    }
  });

  it("dup/late delta short-circuits BEFORE the loop — never hits the new branch (#9)", () => {
    applySnapshot(snapshotAB());
    // new_version <= _packVersion → idempotent discard, returns true, no mutation
    const ok = applyDeltas([{ op: "directive_removed", rule_id: "ZZZ" }], 1);
    expect(ok).toBe(true);
    expect(getPackVersion()).toBe(1);
    expect(isCacheHealthy()).toBe(true);
    expect(getPack().directives.map((d: { id: string }) => d.id)).toEqual(["A", "B"]);
  });
});

// Delta entity-apply must clone-before-mutate (COW) so a reader that
// captured a directive object via getPack() is never mutated under it. The
// reducer shallow-copies the directive LIST but used to mutate the object
// ELEMENTS in place (`const d = directives[...]; d.entities = ...`) — those are
// the SAME objects the live _pack references. Fix: spread-copy the object
// before touching `entities`, then write the clone back into the list. Node has
// no live race (single event loop) — this is parity + defensive isolation.
// Sibling: token-police-python/tests/test_state.py pins the same scenarios.
const snapshotR1 = (entities: unknown[] = []) => ({
  version: 1,
  tenant_id: "t1",
  project_id: "p1",
  directives: [
    { id: "R1", kind: "ENTITY_BLOCK", mode: "enforce", entities: [...entities] },
  ],
  loop_blocks: [],
});

describe("Delta entity-apply clone-before-mutate (Node)", () => {
  beforeEach(() => {
    resetPack();
  });

  it("entity_blocked reader isolation — captured dict untouched, live dict differs (#6)", () => {
    expect(applySnapshot(snapshotR1([]))).toBe(true);
    const dBefore = getPack().directives[0];
    expect(applyDeltas([{ op: "entity_blocked", rule_id: "R1", entity: "user:42" }], 2)).toBe(true);
    // captured object not mutated by the writer
    expect(dBefore.entities).toEqual([]);
    // live pack now holds a fresh clone, not the object the reader captured
    expect(dBefore).not.toBe(getPack().directives[0]);
  });

  it("entity_unblocked reader isolation — captured dict keeps entity, live dict differs (#6)", () => {
    expect(applySnapshot(snapshotR1(["user:42"]))).toBe(true);
    const dBefore = getPack().directives[0];
    expect(applyDeltas([{ op: "entity_unblocked", rule_id: "R1", entity: "user:42" }], 2)).toBe(true);
    expect(dBefore.entities).toEqual(["user:42"]);
    expect(dBefore).not.toBe(getPack().directives[0]);
  });

  it("entity_rerouted reader isolation (reroute arm shares the branch) (#6)", () => {
    expect(applySnapshot({
      version: 1, tenant_id: "t1", project_id: "p1",
      directives: [{ id: "R1", kind: "REROUTE", mode: "enforce", entities: [] }],
      loop_blocks: [],
    })).toBe(true);
    const dBefore = getPack().directives[0];
    expect(applyDeltas([{ op: "entity_rerouted", rule_id: "R1", entity: "user:9" }], 2)).toBe(true);
    expect(dBefore.entities).toEqual([]);
    expect(dBefore).not.toBe(getPack().directives[0]);
  });

  it("entity-apply OUTCOME unchanged: arm present, de-dup, disarm gone, version advances (#7)", () => {
    applySnapshot(snapshotR1([]));
    expect(applyDeltas([{ op: "entity_blocked", rule_id: "R1", entity: "user:42" }], 2)).toBe(true);
    expect(getPack().directives[0].entities).toEqual(["user:42"]);
    expect(getPackVersion()).toBe(2);
    // de-dup: arming an already-armed entity yields no duplicate
    expect(applyDeltas([{ op: "entity_blocked", rule_id: "R1", entity: "user:42" }], 3)).toBe(true);
    expect(getPack().directives[0].entities).toEqual(["user:42"]);
    // disarm removes it
    expect(applyDeltas([{ op: "entity_unblocked", rule_id: "R1", entity: "user:42" }], 4)).toBe(true);
    expect(getPack().directives[0].entities).toEqual([]);
  });

  it("entity op for an absent rule is a no-op but still advances the version (#8)", () => {
    applySnapshot(snapshotR1([]));
    expect(applyDeltas([{ op: "entity_blocked", rule_id: "NOPE", entity: "user:1" }], 2)).toBe(true);
    expect(getPackVersion()).toBe(2);
    expect(getPack().directives[0].entities).toEqual([]);
  });

  it("falsy rid / null entity still short-circuits (continue guard), pack unchanged (#8)", () => {
    applySnapshot(snapshotR1([]));
    const ok = applyDeltas([
      { op: "entity_blocked", rule_id: "", entity: "user:1" },
      { op: "entity_blocked", rule_id: "R1", entity: null },
      { op: "entity_unblocked", rule_id: "R1" },
    ], 2);
    expect(ok).toBe(true);
    expect(getPackVersion()).toBe(2);
    expect(getPack().directives[0].entities).toEqual([]);
    expect(isCacheHealthy()).toBe(true);
  });

  it("upsert carry-forward NOT cloned and prior object uncorrupted (#9)", () => {
    applySnapshot(snapshotR1(["user:7"]));
    const ok = applyDeltas([
      { op: "directive_upserted", directive: { id: "R1", kind: "ENTITY_BLOCK", mode: "enforce" } },
    ], 2);
    expect(ok).toBe(true);
    expect(getPack().directives[0].entities).toEqual(["user:7"]);
  });

  it("entity branches spread-clone before mutate and write the clone back (static pin) (#3,#4)", () => {
    const src = readFileSync(new URL("../src/state.ts", import.meta.url), "utf8");
    const start = src.indexOf("export function applyDeltas");
    const end = src.indexOf("\n// ", start + 1);
    const fn = src.slice(start, end);
    // both entity branches clone via object spread and write back into the list
    const cloneCount = (fn.match(/const d = \{ \.\.\.directives\[idx\.get\(rid\)!\] \};/g) || []).length;
    const writeBackCount = (fn.match(/directives\[idx\.get\(rid\)!\] = d;/g) || []).length;
    expect(cloneCount).toBe(2);
    expect(writeBackCount).toBe(2);
    // the old in-place grab (no spread) must be gone from the entity branches
    expect(fn).not.toContain("const d = directives[idx.get(rid)!];");
  });
});

// Singleton re-init swaps the client. Python guards the swap with a
// threading.Lock (_instance_lock); Node's swap is atomic by virtue of the
// single-threaded event loop, so this side is a comment-only parity note — the
// executable swap logic must stay byte-identical (no Lock/mutex/await added).
// Sibling: token-police-python/tests/test_state.py pins the concurrency case.
describe("SetClient swap (Node event-loop-atomicity parity note)", () => {
  it("swap is comment-only: no Lock/mutex/await introduced (#8)", () => {
    const src = readFileSync(new URL("../src/state.ts", import.meta.url), "utf8");
    const start = src.indexOf("export function setClient");
    const end = src.indexOf("export function", start + 1);
    const fn = src.slice(start, end);
    // Strip comment lines so the "no synchronization primitive" check inspects
    // only executable code (the parity comment itself mentions await/Lock).
    const code = fn
      .split("\n")
      .filter((l) => !l.trim().startsWith("//"))
      .join("\n");
    // no runtime synchronization primitive added to the executable swap
    expect(code).not.toMatch(/\bawait\b/);
    expect(code).not.toMatch(/\bLock\b/);
    expect(code).not.toMatch(/mutex/i);
    // the three executable statements are byte-identical to the original
    expect(code).toContain("_instance.closeSync()");
    expect(code).toContain("_instance = client;");
    expect(code).toContain("resetPack();");
    // the comment documents the deliberate divergence from Python's _instance_lock
    expect(fn.toLowerCase()).toContain("single-threaded event loop");
  });

  it("setClient closes the old client exactly once and installs the new (runtime)", () => {
    let closed = 0;
    const c1 = { closeSync: () => { closed++; } } as unknown as TokenPolice;
    const c2 = { closeSync: () => { closed++; } } as unknown as TokenPolice;
    setClient(c1);
    expect(getClient()).toBe(c1);
    setClient(c2);
    expect(getClient()).toBe(c2);
    expect(closed).toBe(1); // only c1 closed; the c1 install closed nothing
    setClient(null as unknown as TokenPolice); // cleanup
  });
});

// The server stamps every /stream snapshot with a TTL (`ttl_seconds`).
// Neither SDK enforced it — an aged-out Decision Pack stayed "healthy" and kept
// enforcing while heartbeats arrived but no deltas/snapshots. Fix: store the TTL
// on snapshot, add a pure receipt-based `isPackExpired()` predicate (ttl<=0 ⇒
// never expires), and gate BOTH read accessors on it so an expired pack drops
// the enforcer to inline /check. Receipt refreshes on a successful delta too, so
// an actively-updated pack never falsely expires. RED-without-fix: on the
// pre-fix accessors an aged pack still returns non-null getPack() +
// isCacheHealthy() true. No 24h sleep — vi.setSystemTime drives the clock.
// Sibling: token-police-python/tests/test_state.py pins the same scenarios.
const ttlSnapshot = (ttl?: number) => ({
  version: 1,
  tenant_id: "t1",
  project_id: "p1",
  ttl_seconds: ttl,
  directives: [{ id: "A", kind: "UNCONDITIONAL_BLOCK", mode: "enforce" }],
  loop_blocks: [],
});

describe("Snapshot TTL expiry (Node)", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(1_000_000_000_000); // deterministic wall clock
    resetPack();
  });
  afterEach(() => {
    vi.useRealTimers();
    resetPack();
  });

  // Assertion 3 (predicate + ttl<=0 GATE + boundary).
  it("predicate: absent/<=0 TTL never expires; boundary at ttl*1000 (#3,#12)", () => {
    // absent ttl_seconds ⇒ never expires
    applySnapshot(ttlSnapshot(undefined));
    vi.advanceTimersByTime(10 ** 9); // ~11.5 days
    expect(isPackExpired()).toBe(false);

    // ttl_seconds <= 0 ⇒ never expires
    resetPack();
    applySnapshot(ttlSnapshot(0));
    vi.advanceTimersByTime(10 ** 9);
    expect(isPackExpired()).toBe(false);
    resetPack();
    applySnapshot(ttlSnapshot(-5));
    vi.advanceTimersByTime(10 ** 9);
    expect(isPackExpired()).toBe(false);

    // boundary: age exactly ttl*1000 ⇒ NOT expired; +1ms ⇒ expired
    resetPack();
    applySnapshot(ttlSnapshot(100));
    vi.advanceTimersByTime(100 * 1000); // exactly TTL
    expect(isPackExpired()).toBe(false);
    vi.advanceTimersByTime(1); // one ms past
    expect(isPackExpired()).toBe(true);
  });

  // Assertion 3 (null pack ⇒ false).
  it("predicate: null pack is never expired (#3)", () => {
    resetPack();
    expect(isPackExpired()).toBe(false);
  });

  // Assertion 15 (RED-without-fix / GREEN-post-fix) — pins the read accessors.
  it("expired pack: getPack() null + isCacheHealthy() false (#6,#7,#15)", () => {
    applySnapshot(ttlSnapshot(100));
    // pre-expiry: healthy
    expect(getPack()).not.toBeNull();
    expect(isCacheHealthy()).toBe(true);
    // age past TTL — RED on pre-fix accessors, GREEN post-fix
    vi.advanceTimersByTime(100 * 1000 + 1);
    expect(getPack()).toBeNull();
    expect(isCacheHealthy()).toBe(false);
    expect(isPackExpired()).toBe(true);
  });

  // Assertion 12 (no-TTL / ttl<=0 byte-identical read path).
  it("no-TTL snapshot behaves exactly as today after arbitrary time (#12)", () => {
    applySnapshot(ttlSnapshot(undefined));
    vi.advanceTimersByTime(10 ** 10); // ~115 days
    expect(getPack()).not.toBeNull();
    expect(isCacheHealthy()).toBe(true);
    expect(isPackExpired()).toBe(false);
  });

  // Assertion 5 (receipt refreshes on successful delta-apply).
  it("a successful delta resets the receipt clock (no false expiry) (#5)", () => {
    applySnapshot(ttlSnapshot(100));
    vi.advanceTimersByTime(90 * 1000); // 90s of the 100s window elapsed
    // valid contiguous delta refreshes the clock
    expect(applyDeltas([], 2)).toBe(true);
    // advance another 90s: 180s since snapshot (> TTL) but only 90s since delta
    vi.advanceTimersByTime(90 * 1000);
    expect(isPackExpired()).toBe(false); // delta reset the clock
    expect(getPack()).not.toBeNull();
    expect(isCacheHealthy()).toBe(true);
    // now cross the TTL relative to the delta
    vi.advanceTimersByTime(11 * 1000); // total 101s since delta
    expect(isPackExpired()).toBe(true);
  });

  // Assertion 1 (TTL stored on snapshot; non-numeric ⇒ 0 ⇒ never expires).
  it("non-numeric ttl_seconds coerces to 0 (never expires) (#1)", () => {
    applySnapshot({
      version: 1, tenant_id: "t1", project_id: "p1",
      ttl_seconds: "not-a-number",
      directives: [], loop_blocks: [],
    });
    vi.advanceTimersByTime(10 ** 9);
    expect(isPackExpired()).toBe(false);
    expect(getPack()).not.toBeNull();
  });

  // Assertion 2 (reset clears the TTL).
  it("resetPack clears TTL so a stale window can't leak into a TTL-less snapshot (#2)", () => {
    applySnapshot(ttlSnapshot(50));
    vi.advanceTimersByTime(60 * 1000); // expired
    expect(isPackExpired()).toBe(true);
    resetPack();
    // a fresh TTL-less snapshot must NOT inherit the old 50s TTL
    applySnapshot(ttlSnapshot(undefined));
    vi.advanceTimersByTime(10 ** 6);
    expect(isPackExpired()).toBe(false);
  });

  // Assertion 18 (golden-rule static pin): no throw added; predicate is pure
  // arithmetic; no issued_at parse anywhere in state.ts.
  it("state.ts has no throw and no issued_at parse (static pin) (#18)", () => {
    const src = readFileSync(new URL("../src/state.ts", import.meta.url), "utf8");
    expect(src).not.toContain("issued_at");
    // isPackExpired body is a pure comparison — no throw/parse
    const start = src.indexOf("export function isPackExpired");
    const end = src.indexOf("\n}", start);
    const fn = src.slice(start, end);
    expect(fn).not.toContain("throw");
    expect(fn).toContain("Date.now() - _packReceivedAt");
    expect(fn).toContain("_packTtlSeconds * 1000");
  });
});
