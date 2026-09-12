/**
 * SSE client for /v1/guard/stream.
 *
 * Maintains a long-lived fetch ReadableStream (Node 18+), parses SSE
 * frames, and pumps snapshots + deltas into state.applySnapshot /
 * state.applyDeltas. Daemon-only; opened when firewall is 'enforce' or
 * 'dry_run'.
 *
 * Behavior is kept in parity with the TokenPolice Python SDK.
 */

import * as state from "./state";

const HEARTBEAT_TIMEOUT_MS = 45_000;
const STALE_DATA_THRESHOLD_MS = 600_000;

export interface StreamOptions {
  baseUrl: string;
  apiKey: string;
  sdkVersion: string;
  deployment: string;
  clientId: string;
  firewall: string;
  reconnectCapSeconds?: number;
}

export class StreamClient {
  private opts: Required<StreamOptions>;
  private stopped = false;
  private controller: AbortController | null = null;
  private lastEventId: number | null = null;
  // Per-connection "force immediate reconnect" flag. Set on the unhealthy-cache
  // delta branch (dispatch, below) and on snapshot-TTL expiry (connectAndPump);
  // consumed by the loop-top guard in connectAndPump so the existing backoff
  // reconnects without Last-Event-ID (→ fresh snapshot). Reset per-connection
  // at pump-entry so a flag set on connection N can never kill connection N+1.
  // Never sets `stopped`.
  private forceReconnect = false;
  // A terminal connect status (401/403/404) does NOT stop the reader. It records
  // the status here so the outer loop re-probes at the reconnect cap (ladder top)
  // instead of the ~1s floor; cleared once consumed. Never sets `stopped`.
  private terminalStatus: number | null = null;
  // Per-connection "traffic evidence" flag: true once the current connection
  // has delivered any dispatched frame or any comment other than the
  // `: stream-open` banner. Gates the attempt=0 reset in runForever — a 200
  // that EOFs before any traffic is a failed connect in disguise, and
  // resetting backoff for it collapses the ladder into a ~1 Hz reconnect
  // storm. stream-open doesn't count as traffic: the server writes it before
  // subscribe/snapshot on every connection, including ones about to fail, so
  // it proves nothing. Parity with the Python SDK's _saw_stream_traffic.
  private sawStreamTraffic = false;
  // Liveness generation owned by the CURRENT connection (0 = none yet). Set at
  // connect, handed back on exit so a superseded pump can't retire a live
  // stream. Stays set after an exit: a reconnect attempt that fails before
  // connecting re-marks the same (already-retired) generation, which no-ops.
  private streamGeneration = 0;
  private lastEventAt = 0;
  private lastDataEventAt = 0;
  private loopPromise: Promise<void> | null = null;
  private wakeResolve: (() => void) | null = null;
  private wakeTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(opts: StreamOptions) {
    this.opts = {
      reconnectCapSeconds: 300,
      ...opts,
    };
    // Clamp the reconnect cap to a 60s floor at construction, parity with the
    // Python SDK's `max(60, int(reconnect_cap_seconds or 300))`. undefined/0/NaN
    // → 300, any value < 60 → 60, ≥60 unchanged. Without this, a caller-supplied
    // cap < 60 lets Node reconnect to /v1/guard/stream faster than 60s during a
    // sustained outage — a reconnect storm the Python SDK cannot produce. The
    // clamped value feeds both the backoff ladder's terminal step and the
    // per-sleep ceiling in runForever, so clamping once here is sufficient.
    // (JS keeps float residuals ≥60, e.g. 120.9 vs Python's int-truncated 120 —
    // negligible for a reconnect ceiling.) Covered by streamReconnectCap.test.ts.
    this.opts.reconnectCapSeconds = Math.max(60, this.opts.reconnectCapSeconds || 300);
  }

  start(): void {
    if (this.loopPromise) return;
    this.stopped = false;
    this.loopPromise = this.runForever()
      .catch(() => {
        // Silent fail — caller already handles via fallback to inline /check.
      })
      // Null loopPromise on the loop's true exit so a later start() on the same
      // instance restarts cleanly (JS analog of the Python SDK's is_alive()
      // guard). loopPromise is non-null iff the loop is running, so the
      // early-return guard at the top of start() still no-ops while running
      // (running path unchanged) and a sync stop()+start() still no-ops (loop
      // not yet exited) — no double-loop race. Covered by streamRestart.test.ts.
      .finally(() => { this.loopPromise = null; });
  }

  stop(): void {
    this.stopped = true;
    // Only signals. Retiring the connection is left to the pump's own exit path
    // below, which owns the generation — stop() marking here would have to guess
    // one, and a wrong guess is exactly the zombie-marks-live-stream bug the
    // generation check exists to prevent.
    try { this.controller?.abort(); } catch { /* ignored */ }
    // Wake an in-progress reconnect backoff so teardown is immediate instead
    // of blocking up to reconnectCap seconds on a pending setTimeout (which
    // would also keep the event loop pinned after shutdown()).
    if (this.wakeTimer) { clearTimeout(this.wakeTimer); this.wakeTimer = null; }
    if (this.wakeResolve) { this.wakeResolve(); this.wakeResolve = null; }
  }

  private async runForever(): Promise<void> {
    const steps = [1, 2, 4, 8, 16, 32, 64, 128, 256, this.opts.reconnectCapSeconds];
    let attempt = 0;
    while (!this.stopped) {
      try {
        await this.connectAndPump();
        if (this.terminalStatus !== null) {
          // The last connect returned a terminal status (401/403/404) without
          // stopping. Re-probe at the reconnect cap (ladder top) rather than the
          // ~1s floor — a persistently terminal endpoint is polled at most once
          // per cap period, never hammered. A later non-terminal connect resets
          // attempt=0.
          this.terminalStatus = null;
          attempt = steps.length - 1;
        } else if (this.sawStreamTraffic) {
          // Reset backoff only when the disconnect showed traffic evidence
          // (a frame or a keepalive). A clean 200 EOF with no traffic ever is
          // a failed connect in disguise — preserve `attempt` so the ladder
          // climbs (1→2→4→8…) like the transient path, instead of storming at
          // the ~1s floor.
          attempt = 0;
        }
      } catch {
        // ignored — reconnect with backoff below
      } finally {
        // EVERY pump exit is a disconnect: transient throw, heartbeat/stale
        // watchdog return, clean EOF, forced-reconnect + snapshot-TTL branch,
        // abort/stop. Retires only THIS pump's generation (0 = never connected,
        // which no generation ever equals), so a superseded pump exiting late
        // cannot mark the replacement's live stream down. Idempotent +
        // earliest-stamp within a generation (see state.ts), so a failing
        // reconnect ladder can never refresh the staleness clock. Pack state is
        // deliberately untouched — a disconnect never invalidates the cache.
        try { state.markStreamDisconnected(this.streamGeneration); } catch { /* ignored */ }
      }
      if (this.stopped) return;
      const base = steps[Math.min(attempt, steps.length - 1)];
      const jitter = base * 0.2 * (Math.random() * 2 - 1);
      const sleepMs = Math.max(1000, Math.min(this.opts.reconnectCapSeconds * 1000, (base + jitter) * 1000));
      attempt += 1;
      await new Promise<void>((resolve) => {
        this.wakeResolve = resolve;
        this.wakeTimer = setTimeout(resolve, sleepMs);
        // A backoff sleep must never keep the customer's process alive: unref so
        // a pending reconnect wait can't pin process exit during an outage (up to
        // the reconnect cap per cycle). stop() still clears and resolves it —
        // unref changes neither. Mirrors the idle timer above. Covered by
        // streamIdleTimeout.test.ts.
        this.wakeTimer.unref?.();
      });
      this.wakeResolve = null;
      this.wakeTimer = null;
    }
  }

  private async connectAndPump(): Promise<void> {
    const url = `${this.opts.baseUrl}/v1/guard/stream`;
    const headers: Record<string, string> = {
      Authorization: `Bearer ${this.opts.apiKey}`,
      Accept: "text/event-stream",
      "Cache-Control": "no-cache",
      "X-TP-Sdk-Version": this.opts.sdkVersion,
      "X-TP-Sdk-Schema-Version": "1",
      "X-TP-Client-Id": this.opts.clientId,
      "X-TP-Deployment-Mode": this.opts.deployment,
      "X-TP-Firewall-Mode": this.opts.firewall,
    };
    if (this.lastEventId !== null) headers["Last-Event-ID"] = String(this.lastEventId);

    const ac = new AbortController();
    this.controller = ac;
    let reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
    let idleTimer: ReturnType<typeof setTimeout> | null = null;
    try {
      const resp = await fetch(url, { headers, signal: ac.signal });
      if (!resp.ok || !resp.body) {
        // A terminal connect status (revoked/invalid key → 401/403, or a
        // misconfigured base_url/route → 404) will not self-heal on a fast retry.
        // Rather than hammer the endpoint at the ~1s backoff floor (a tight
        // reconnect loop), invalidate the pack FIRST — the TTL watchdog lives
        // inside the pump loop, so leaving a still-healthy pack in place would
        // enforce a stale pack forever; unhealthy → enforcer bails to inline
        // /check (fail-open) — then mark this connect terminal and return WITHOUT
        // setting the stop signal. The outer loop re-probes at the reconnect cap
        // (~one probe per cap period); a later non-terminal connect resumes normal
        // streaming. stop() still interrupts the cap-length wait instantly.
        // Drop Last-Event-ID here too (every other poison path already does).
        // Collector skip-when-equal withholds the snapshot that clears the
        // poison; a leftover cursor on an idle project never heals (SSE-12).
        if (resp.status === 401 || resp.status === 403 || resp.status === 404) {
          state.invalidatePack("stream_terminal_" + resp.status);
          this.lastEventId = null;
          this.terminalStatus = resp.status;
          return;
        }
        // A TRANSIENT connect failure (429/5xx/other-4xx/bodyless-200) must
        // NOT bare-return — a clean return resets runForever's attempt=0, collapsing
        // the exponential backoff ladder to its ~1s floor (a reconnect storm on
        // /v1/guard/stream during a sustained outage). Throw instead so the existing
        // runForever catch preserves the climbed `attempt` and the next reconnect
        // climbs the ladder (1→2→4→8…). Terminal statuses already returned above
        // (no throw); the forced-reconnect and snapshot-TTL-expiry returns inside
        // the pump loop below are unaffected.
        throw new Error("stream connect " + resp.status);
      }
      // Connection actually established (200 + body) — every terminal/transient
      // status check above has passed. Capture the generation this connection
      // owns; only it may retire the liveness state (see runForever's finally).
      this.streamGeneration = state.markStreamConnected();
      // No traffic evidence yet for this connection — reset here (not inside
      // the pump loop) so a 200 whose body dies before the reader loop runs
      // (including the no-reader early return below) still counts as
      // traffic-less and climbs the backoff ladder.
      this.sawStreamTraffic = false;
      const now = Date.now();
      this.lastEventAt = now;
      this.lastDataEventAt = now;

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      reader = (resp.body as any).getReader
        ? (resp.body as ReadableStream<Uint8Array>).getReader()
        : null;
      if (!reader) return;
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let eventType = "message";
      let eventId: string | null = null;
      let dataLines: string[] = [];

      // Idle read timeout (parity with the Python SDK's httpx read=45s timeout).
      // Web Streams / undici fetch have no built-in idle timeout, so on a silent
      // half-open socket reader.read() (below) never resolves *or* rejects and the
      // inline watchdog above can never re-evaluate. This timer aborts captured
      // `ac` after HEARTBEAT_TIMEOUT_MS of byte-silence; the in-flight read() then
      // rejects AbortError, connectAndPump() throws, and the existing runForever()
      // catch + backoff reconnects. Armed only AFTER the reader is obtained (a
      // pre-reader early return never arms it), reset on ANY received bytes incl.
      // comment keepalives (an *idle* timeout, not a wall-clock ceiling). finally
      // still clears the timer first, then aborts `ac`.
      const timer = setTimeout(() => { try { ac.abort(); } catch { /* ignored */ } }, HEARTBEAT_TIMEOUT_MS);
      timer.unref();
      idleTimer = timer;

      // Fresh connection starts with a clear reconnect flag.
      this.forceReconnect = false;
      while (!this.stopped) {
        // An unhealthy-cache delta asked for an immediate reconnect.
        // Return normally (no throw) → finally clears the idle timer and
        // aborts this pump's fetch; return stays a return. Then runForever
        // resets attempt=0 and the existing backoff reconnects with
        // no Last-Event-ID so the server sends a fresh snapshot.
        if (this.forceReconnect) return;
        const now2 = Date.now();
        if (now2 - this.lastEventAt > HEARTBEAT_TIMEOUT_MS) return; // force reconnect
        if (now2 - this.lastDataEventAt > STALE_DATA_THRESHOLD_MS) return;
        // The pack aged past its server-stamped TTL while heartbeats kept
        // arriving but no snapshot/delta refreshed it. Invalidate (→ inline
        // /check) and reuse the same forced-reconnect flag: null the cursor and
        // set forceReconnect so the loop-top guard returns → the existing
        // backoff reconnects without Last-Event-ID → fresh snapshot → pack
        // un-expires. No new backoff/socket/timer code.
        if (state.isPackExpired()) {
          state.invalidatePack("snapshot_ttl_expired");
          this.lastEventId = null;
          this.forceReconnect = true;
        }

        const { value, done } = await reader.read();
        if (done) return;
        timer.refresh(); // reset idle window on any received bytes
        buffer += decoder.decode(value, { stream: true });
        let nlIdx;
        // eslint-disable-next-line no-cond-assign
        while ((nlIdx = buffer.indexOf("\n")) >= 0) {
          const line = buffer.slice(0, nlIdx).replace(/\r$/, "");
          buffer = buffer.slice(nlIdx + 1);
          this.lastEventAt = Date.now();
          if (line === "") {
            if (dataLines.length > 0) {
              // Traffic evidence at dispatch time regardless of whether apply
              // succeeds — a well-formed frame proves the server pipeline is
              // alive.
              this.sawStreamTraffic = true;
              this.dispatch(eventType, eventId, dataLines.join("\n"));
              this.lastDataEventAt = Date.now();
            }
            eventType = "message"; eventId = null; dataLines = [];
            continue;
          }
          if (line.startsWith(":")) {
            // comment / keepalive. Any comment except the stream-open banner
            // is traffic evidence (the banner precedes subscribe/snapshot on
            // every connection, including ones about to fail).
            if (line.slice(1).trim() !== "stream-open") this.sawStreamTraffic = true;
            continue;
          }
          if (line.startsWith("event:")) eventType = line.slice(6).trim();
          else if (line.startsWith("id:")) eventId = line.slice(3).trim();
          else if (line.startsWith("data:")) dataLines.push(line.slice(5).replace(/^ /, ""));
        }
      }
    } finally {
      if (idleTimer) clearTimeout(idleTimer);
      // Abort MUST precede cancel: cancel on an already-errored stream rejects immediately;
      // cancel-first can block on the source. Do not await cancel — a hang stalls markStreamDisconnected.
      try { ac.abort(); } catch { /* ignored */ }
      if (reader) {
        try {
          const p = reader.cancel?.();
          if (p != null && typeof (p as Promise<unknown>).then === "function") {
            (p as Promise<unknown>).catch(() => {});
          }
        } catch { /* ignored */ }
      }
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private dispatch(eventType: string, eventId: string | null, data: string): void {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let payload: any;
    try { payload = JSON.parse(data); } catch { return; }

    if (eventType === "snapshot") {
      const ok = state.applySnapshot(payload);
      if (ok && eventId && /^\d+$/.test(eventId)) this.lastEventId = parseInt(eventId, 10);
      else if (!ok) {
        // Apply FAILED → drop cache and cursor so the next reconnect pulls a
        // fresh snapshot. A successful apply that merely lacks a numeric id keeps
        // the pack healthy and leaves the cursor as-is (mirrors the delta branch
        // below) — a missing id must never discard a snapshot that applied cleanly.
        state.invalidatePack("snapshot_apply_failed");
        this.lastEventId = null;
      }
      return;
    }

    if (eventType === "delta") {
      const version = Number(payload?.version || 0);
      const ops = Array.isArray(payload?.ops) ? payload.ops : [];
      if (!version) return;
      if (!state.isCacheHealthy()) {
        // Delta arrived while the cache is unhealthy (no snapshot landed yet,
        // or a prior apply/TTL failure poisoned it). Dropping the cursor is
        // inert without a reconnect (Last-Event-ID is only sent in the connect
        // headers), so also force an immediate reconnect → the server pushes a
        // fresh snapshot and the cache self-heals.
        this.lastEventId = null;
        this.forceReconnect = true;
        return;
      }
      const ok = state.applyDeltas(ops, version);
      if (ok && eventId && /^\d+$/.test(eventId)) this.lastEventId = parseInt(eventId, 10);
      else if (!ok) {
        this.lastEventId = null;
        state.invalidatePack("delta_apply_failed");
      }
    }
  }
}
