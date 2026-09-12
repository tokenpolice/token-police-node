/**
 * Global singleton state for the TokenPolice client.
 *
 * In addition to the legacy client singleton, holds the v2 Decision Pack
 * cache: directives, monotonic version, plus an observations queue with
 * per-call attribution: each entry is tagged (at push time) with the minted
 * "obs key" of the LLM call whose pre-flight produced it, and a /log drain
 * claims only its own call's entries — plus untagged entries (no call scope
 * was active at push) and stale orphans (whose owning /log never fired).
 * Legacy no-arg drains and the flush/shutdown greedy mode still drain
 * everything.
 *
 * Behavior is kept in parity with the TokenPolice Python SDK.
 */

import type { TokenPolice } from "./client";
import { randomUUID } from "crypto";
import { AsyncLocalStorage } from "node:async_hooks";

let _instance: TokenPolice | null = null;

export function getClient(): TokenPolice | null {
  return _instance;
}

export function setClient(client: TokenPolice): void {
  // Parity note (the Python SDK guards the equivalent swap with an instance
  // lock): the whole close-old → install-new → reset-pack swap below is atomic
  // by virtue of the single-threaded event loop. There is no `await` in this
  // function, so it runs to completion before any other task (including a
  // concurrent re-init()) can observe or mutate `_instance` — a runtime mutex
  // would be a no-op here. This deliberate divergence is why the Python SDK
  // needs a lock and Node does not.
  if (_instance && typeof _instance.closeSync === "function") {
    try { _instance.closeSync(); } catch { /* ignored */ }
  }
  _instance = client;
  resetPack();
}

// ── v2: Decision Pack cache ───────────────────────────────────────────
// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

let _pack: Any | null = null;
let _packVersion = 0;
let _packReceivedAt = 0;
// Snapshot TTL in seconds as stamped by the server (`ttl_seconds` field).
// 0 / absent / non-numeric ⇒ never expires (legacy-snapshot path stays
// byte-identical). Read only by isPackExpired().
let _packTtlSeconds = 0;
let _cacheInvalid = false;
let _packTenantId: string | null = null;
let _packProjectId: string | null = null;

/**
 * Internal observation wrapper: `obs` is the raw observation object (the ONLY
 * thing that ever reaches a /log payload), `key` is the owning call's obs key
 * (null = untagged), `ts` is Date.now() at push (staleness clock).
 */
type ObsEntry = { obs: Any; key: string | null; ts: number };

let _observationsQueue: ObsEntry[] = [];
// Greedy drain mode: while the DEPTH COUNTER is > 0, EVERY drain returns the
// whole queue regardless of keys. Engaged (begin/end) around client flush()/
// close() and SpanProcessor.shutdown — final flush must ship everything (the
// old behavior); a filtered drain there would STRAND tagged orphans = silent
// audit loss, the worst regression class. A counter, NOT a boolean: the
// beforeExit fire-and-forget flush can overlap a user's awaited close() (and
// two concurrent close() calls both enter before either seals), and a
// boolean's first finally would clear the flag mid-way through the other
// call's window. NOT engaged by SpanProcessor.forceFlush — that is a PUBLIC
// per-request OTel API (platform integrations call it on live traffic) and a
// greedy window there would reopen cross-call stealing; its orphans ride the
// staleness fallback instead.
let _obsDrainAllDepth = 0;
let _clientId = "";

export function resetPack(): void {
  _pack = null;
  _packVersion = 0;
  _packReceivedAt = 0;
  _packTtlSeconds = 0;
  _cacheInvalid = false;
  _packTenantId = null;
  _packProjectId = null;
  _observationsQueue = [];
  // Zero (not decrement) the greedy depth: a re-init deliberately discards
  // any in-flight flush window — the old client's queue was just cleared, so
  // its window has nothing left to protect. The orphaned windows' end() calls
  // clamp at 0, so no underflow can arm greedy mode for the new client.
  _obsDrainAllDepth = 0;
}

export function getPack(): Any | null {
  if (_cacheInvalid || _pack === null || isPackExpired()) return null;
  return _pack;
}

// Pure, throw-free receipt-based expiry predicate. An expired pack makes both
// read accessors go null/false → the existing enforcer null-pack bail
// (enforcer.ts) drops to inline /check (fail-open). Gate: ttl<=0/absent ⇒
// never expires (legacy path unchanged). Unit bridge: `_packReceivedAt` is
// Date.now() ms, `ttl_seconds` is seconds → compare against
// `_packTtlSeconds * 1000`.
export function isPackExpired(): boolean {
  return (
    _pack !== null &&
    _packTtlSeconds > 0 &&
    Date.now() - _packReceivedAt > _packTtlSeconds * 1000
  );
}

export function getPackVersion(): number {
  return _packVersion;
}

export function getClientId(): string {
  return _clientId;
}

export function assignClientId(): string {
  _clientId = randomUUID().replace(/-/g, "");
  return _clientId;
}

export function isCacheHealthy(): boolean {
  return _pack !== null && !_cacheInvalid && !isPackExpired();
}

export function invalidatePack(_reason = ""): void {
  _cacheInvalid = true;
}

// ── SSE stream connection freshness ───────────────────────────────────
// Tracked SEPARATELY from pack health on purpose: a stream disconnect must
// never invalidate the pack — last-known-good blocking during an outage is
// design intent. These two fields only let the enforcer decide that a locally
// ALLOWED call whose decision hinged on a streamed entity list has gone too
// stale to trust (entity arming reaches the SDK over SSE and nowhere else), and
// re-verify it with the inline /check. Written by the stream pump, read by the
// enforcer. Behavior is kept in parity with the Python SDK.
let _streamConnected = false;
let _streamDisconnectedAt: number | null = null;
// Monotonic connection generation. These fields are module-global but every
// connection is owned by ONE pump, and a replaced client's pump can outlive the
// swap (it exits only when its parked read unblocks). Without an owner check
// that zombie's exit path would mark the LIVE stream disconnected forever — it
// never re-marks connected. Each connect takes the next generation and only its
// owner may retire it.
let _streamGeneration = 0;

/**
 * Stream established (HTTP 200, body open). Returns the generation the caller
 * must hand back to markStreamDisconnected() when this connection ends.
 */
export function markStreamConnected(): number {
  _streamGeneration += 1;
  _streamConnected = true;
  _streamDisconnectedAt = null;
  return _streamGeneration;
}

/**
 * Stream lost/closed, retiring `generation` — a stale generation (a pump whose
 * connection was already superseded) is ignored, so it can never mark a live
 * stream down. IDEMPOTENT within a generation by design: the grace window is
 * measured from the EARLIEST disconnect since the last successful connect, so
 * the repeated calls a failing reconnect ladder produces must NOT refresh the
 * stamp (that would keep extending the stale-allow window through a sustained
 * outage).
 */
export function markStreamDisconnected(generation: number): void {
  if (generation !== _streamGeneration) return;
  if (!_streamConnected) return;
  _streamConnected = false;
  _streamDisconnectedAt = Date.now();
}

/**
 * True while streamed entity lists can still be trusted: connected, or
 * disconnected less than `graceMs` ago. Never-connected ⇒ false (defensive —
 * the pack is null then, so the enforcer is already on the inline /check path).
 */
export function isStreamFresh(graceMs: number): boolean {
  if (_streamConnected) return true;
  if (_streamDisconnectedAt === null) return false;
  const grace =
    typeof graceMs === "number" && Number.isFinite(graceMs) && graceMs > 0 ? graceMs : 0;
  return Date.now() - _streamDisconnectedAt <= grace;
}

export function applySnapshot(snapshot: Any): boolean {
  try {
    const version = Number(snapshot?.version ?? 0);
    const t = snapshot?.tenant_id;
    const p = snapshot?.project_id;
    // A snapshot without a well-formed tenant/project identity is refused and the
    // cache is poisoned (→ inline /check) rather than cached — so the cross-tenant
    // pin only ever arms from a snapshot that actually carries both ids. Without
    // this, a first snapshot missing an id would either leave the pin disarmed
    // forever or arm it half-formed and reject every later good snapshot. Covered
    // by "snapshot missing ids is refused and pin arms after heal".
    if (typeof t !== "string" || !t || typeof p !== "string" || !p) {
      _cacheInvalid = true;
      return false;
    }
    if (_packTenantId !== null && (_packTenantId !== t || _packProjectId !== p)) {
      _cacheInvalid = true;
      return false;
    }
    _pack = snapshot;
    _packVersion = version;
    _packReceivedAt = Date.now();
    // Stamp the server TTL; non-numeric/absent ⇒ 0 ⇒ never expires.
    _packTtlSeconds = Number(snapshot?.ttl_seconds ?? 0) || 0;
    _cacheInvalid = false;
    // Guaranteed non-empty strings by the identity guard above.
    _packTenantId = t;
    _packProjectId = p;
    return true;
  } catch {
    _cacheInvalid = true;
    return false;
  }
}

export function applyDeltas(ops: Any[], newVersion: number): boolean {
  try {
    if (_pack === null || _cacheInvalid) return false;
    if (newVersion <= _packVersion) return true; // duplicate/late
    if (newVersion !== _packVersion + 1) {
      _cacheInvalid = true;
      return false;
    }

    const directives: Any[] = Array.isArray(_pack.directives) ? [..._pack.directives] : [];
    let loopBlocks: Any[] = Array.isArray(_pack.loop_blocks) ? [..._pack.loop_blocks] : [];
    const idx = new Map<string, number>();
    directives.forEach((d, i) => idx.set(d?.id, i));

    for (const op of ops || []) {
      const kind = op?.op;
      if (kind === "directive_upserted") {
        const d = op.directive || {};
        if (!d.id) continue;
        if (idx.has(d.id)) {
          const prevIdx = idx.get(d.id)!;
          const prev = directives[prevIdx];
          // `directive_upserted` ops arrive without the runtime `entities`
          // list; a full replace would transiently disarm already-breached
          // entities on every rule edit. Carry the armed set forward when the
          // kind is unchanged and entity-bearing (block vs. reroute entity
          // sets differ, so a kind change must NOT inherit them). The
          // directive's own `mode` still gates enforcement, so this can never
          // arm a dry_run rule.
          if (
            d.entities === undefined && prev && prev.kind === d.kind &&
            (d.kind === "ENTITY_BLOCK" || d.kind === "REROUTE") &&
            Array.isArray(prev.entities)
          ) {
            d.entities = prev.entities;
          }
          directives[prevIdx] = d;
        } else { idx.set(d.id, directives.length); directives.push(d); }
      } else if (kind === "directive_removed") {
        const rid = op.rule_id;
        if (rid && idx.has(rid)) {
          const i = idx.get(rid)!;
          directives.splice(i, 1);
          idx.clear();
          directives.forEach((d, j) => idx.set(d?.id, j));
        } else if (rid) {
          // A truthy rule_id that misses the index means the removal op
          // referenced a directive we don't have. Rather than silently advance
          // the version (leaving a removed rule enforcing until the next full
          // snapshot), poison the cache and bail so the SSE caller re-snapshots.
          // Mirrors the catch below; Node is single-threaded so no lock concern.
          _cacheInvalid = true;
          return false;
        }
        // falsy rid: benign no-op, version still advances (unchanged).
      } else if (kind === "entity_blocked" || kind === "entity_rerouted") {
        const rid = op.rule_id; const entity = op.entity;
        if (!rid || entity == null) continue;
        if (idx.has(rid)) {
          // Clone-before-mutate (copy-on-write), parity with the Python SDK.
          // The directive object is still the one the live _pack references;
          // copy it before touching `entities`, then write the clone back, so a
          // reader that captured the old object is never mutated under it. (Node
          // has no live race — single event loop — this is parity + defensive.)
          const d = { ...directives[idx.get(rid)!] };
          const ents: Any[] = Array.isArray(d.entities) ? [...d.entities] : [];
          if (!ents.includes(entity)) ents.push(entity);
          d.entities = ents;
          directives[idx.get(rid)!] = d;
        }
      } else if (kind === "entity_unblocked" || kind === "entity_unrerouted") {
        const rid = op.rule_id; const entity = op.entity;
        if (!rid || entity == null) continue;
        if (idx.has(rid)) {
          // Clone-before-mutate (copy-on-write) — same reasoning as the arm branch.
          const d = { ...directives[idx.get(rid)!] };
          d.entities = (Array.isArray(d.entities) ? d.entities : []).filter((e: Any) => e !== entity);
          directives[idx.get(rid)!] = d;
        }
      } else if (kind === "loop_blocked") {
        if (op.trace_id && !loopBlocks.includes(op.trace_id)) loopBlocks.push(op.trace_id);
      } else if (kind === "loop_unblocked") {
        if (op.trace_id) loopBlocks = loopBlocks.filter((x) => x !== op.trace_id);
      }
      // unknown op: skip, version still advances
    }

    _pack = { ..._pack, directives, loop_blocks: loopBlocks, version: newVersion };
    _packVersion = newVersion;
    // A successful delta is a fresh "update" — reset the receipt clock so a
    // pack kept current by a steady delta stream never falsely expires (TTL
    // means "no snapshot OR delta for ttl_seconds").
    _packReceivedAt = Date.now();
    return true;
  } catch {
    _cacheInvalid = true;
    return false;
  }
}

// ── Observations ──────────────────────────────────────────────────────

/**
 * Staleness window for tagged-but-unclaimed observations. The window must
 * exceed the longest plausible LLM call duration — a short window would let a
 * concurrent /log re-steal a slow/streaming call's observations, recreating
 * the cross-trace-attribution bug this queue design fixes. A stale entry is
 * an orphan whose owning /log never fired (crashed wrapper, abandoned
 * stream); shipping it late on an arbitrary /log is deliberate — loss is
 * strictly worse than late/misattributed delivery.
 */
export const OBS_STALE_MS = 300_000;

/**
 * Per-call async-context scope carrying the minted obs key that tags every
 * observation pushed during that call's pre-flight. Lives here (state.ts is
 * imported by enforcer/telemetry, keeping imports acyclic). Mirrors the
 * enforcer's anthropic stream-latch stub pattern: on runtimes without
 * AsyncLocalStorage this degrades to run⇒fn() / getStore⇒undefined —
 * pushes go untagged and drains claim null ≈ pre-keying behavior, never a
 * throw.
 */
type ObsScope = { key: string };
const _callObsStorage: {
  run<T>(store: ObsScope, fn: () => T): T;
  getStore(): ObsScope | undefined;
} = (() => {
  try {
    return new AsyncLocalStorage<ObsScope>();
  } catch {
    return {
      run<T>(_store: ObsScope, fn: () => T): T {
        return fn();
      },
      getStore(): ObsScope | undefined {
        return undefined;
      },
    };
  }
})();

/** Mint a fresh obs key. Guarded — null when the runtime cannot mint. */
export function newObsKey(): string | null {
  try {
    return randomUUID();
  } catch {
    return null;
  }
}

/**
 * Run `fn` inside a per-call obs scope. REUSE-IF-EXISTS is deliberate:
 * nested wrapper invocations inside another wrapper's body are SDK-internal
 * delegation of the SAME logical customer call (vercel→provider,
 * `.stream()`→internal create, langchain→provider) and must share one key so
 * their pushes and drains pair up. App-level calls can never nest — the app
 * awaits outside the run scope — so a fresh key is minted exactly once per
 * customer call. Fail-open: any scope error runs `fn` bare (untagged pushes,
 * null-claim drains ≈ pre-keying behavior).
 */
export function runWithCallObsScope<T>(fn: () => T): T {
  // Scope RESOLUTION is guarded separately from fn INVOCATION: `fn` runs
  // exactly once on every path, and its own errors (TokenPoliceBlockedError,
  // customer construction errors) propagate untouched. A catch spanning
  // `run(scope, fn)` would re-invoke fn after fn's OWN throw — double-firing
  // the provider call — so only the storage reads are try/caught.
  let scope: ObsScope | null = null;
  try {
    const existing = _callObsStorage.getStore();
    if (!existing) {
      const key = newObsKey();
      if (key) scope = { key };
    }
  } catch {
    scope = null;
  }
  if (scope) return _callObsStorage.run(scope, fn);
  return fn();
}

/**
 * Re-enter a previously captured obs scope around `fn`. Used by stream
 * wrappers whose drain callbacks run in the CONSUMER's async context (ALS
 * does not survive external async-generator resumption) — they capture the
 * key at wrapper entry (inside the scope) and restore it here for the
 * drain-time log call. Falsy key ⇒ plain `fn()` (drains then claim null).
 */
export function runWithObsKey<T>(key: string | null | undefined, fn: () => T): T {
  // No catch around run(): AsyncLocalStorage.run only throws when fn itself
  // throws, and re-invoking fn from a catch would double-run the wrapped
  // drain/log logic. fn runs exactly once; its errors propagate untouched.
  if (key) return _callObsStorage.run({ key }, fn);
  return fn();
}

/** The current call's obs key, or undefined outside any call scope. */
export function getCurrentObsKey(): string | undefined {
  try {
    return _callObsStorage.getStore()?.key;
  } catch {
    return undefined;
  }
}

/**
 * Queue a firewall observation. PUBLIC SIGNATURE UNCHANGED — the entry is
 * internally tagged with the current call's obs key (null when no scope is
 * active / any scope read fails: fail-open to the untagged pool, which every
 * drain claims).
 */
export function pushObservation(obs: Any): void {
  if (!obs || typeof obs !== "object") return;
  let key: string | null = null;
  try {
    key = getCurrentObsKey() ?? null;
  } catch {
    key = null;
  }
  let ts = 0;
  try {
    ts = Date.now();
  } catch {
    ts = 0;
  }
  _observationsQueue.push({ obs, key, ts });
}

// No-arg drain sentinel — distinguishes legacy `drainObservations()` (drain
// ALL: shutdown paths, tests) from an explicit `drainObservations(null)`
// (unknown claimant: untagged + stale only).
const _DRAIN_ALL: unique symbol = Symbol("tp.drainAll");

/**
 * Drain observations for a /log POST, returning RAW observation objects
 * (the internal {obs, key, ts} wrapper never reaches a payload).
 *
 * - No argument → drain EVERYTHING (exact legacy semantics — shutdown
 *   callers and existing tests rely on this).
 * - Explicit `null` (unknown claimant) → drain untagged entries + stale
 *   orphans (age > OBS_STALE_MS) only.
 * - A string key → drain that call's entries + untagged + stale; everything
 *   else stays queued for its own call's drain (order preserved on both the
 *   returned and remaining sides).
 *
 * While a greedy flush window is open (see beginObservationsDrainAll) every
 * drain returns everything — a terminal flush must never strand an orphan.
 */
export function drainObservations(
  claimKey: string | null | typeof _DRAIN_ALL = _DRAIN_ALL,
): Any[] {
  if (claimKey === _DRAIN_ALL || _obsDrainAllDepth > 0) {
    const out = _observationsQueue;
    _observationsQueue = [];
    return out.map((e) => e.obs);
  }
  const now = Date.now();
  const claimed: Any[] = [];
  const kept: ObsEntry[] = [];
  for (const e of _observationsQueue) {
    if (
      e.key === null ||
      (typeof claimKey === "string" && e.key === claimKey) ||
      now - e.ts > OBS_STALE_MS
    ) {
      claimed.push(e.obs);
    } else {
      kept.push(e);
    }
  }
  _observationsQueue = kept;
  return claimed;
}

/**
 * Open a greedy drain window: while at least one window is open, ANY drain
 * call returns the whole queue. forceFlushTokenPoliceSpans (greedy mode)
 * begins on entry / ends in a finally so client flush()/close()/processor
 * shutdown ship every queued observation (the old drain-all behavior)
 * instead of stranding tagged orphans. Depth-counted so overlapping windows
 * (beforeExit fire-and-forget flush racing an awaited close(), concurrent
 * close() calls) never clear each other early. NOT used by the per-request
 * SpanProcessor.forceFlush path — see the counter comment above.
 */
export function beginObservationsDrainAll(): void {
  _obsDrainAllDepth += 1;
}

/** Close a greedy drain window. Clamped at 0 — resetPack may have discarded
 * in-flight windows, and an orphaned end() must never underflow into a
 * negative depth (which would break the `> 0` gate's meaning). */
export function endObservationsDrainAll(): void {
  if (_obsDrainAllDepth > 0) _obsDrainAllDepth -= 1;
}

/**
 * @internal Test-only compatibility shim over the depth counter: true forces
 * one open window (depth=1), false closes all (depth=0). Production code
 * uses begin/end pairs — do not call this outside tests.
 */
export function setObservationsDrainAll(value: boolean): void {
  _obsDrainAllDepth = value === true ? 1 : 0;
}

/**
 * @internal Test seam (parity with the Python SDK's `_observations_set`):
 * replace the queue with explicit {obs, key, ts} entries so tests can inject
 * keys and ages without reaching into module privates.
 */
export function _observationsSet(entries: Array<{ obs: Any; key: string | null; ts: number }>): void {
  _observationsQueue = entries.map((e) => ({
    obs: e.obs,
    key: e.key ?? null,
    ts: typeof e.ts === "number" ? e.ts : Date.now(),
  }));
}
