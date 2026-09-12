/**
 * OpenTelemetry coexistence — setupOpenTelemetry must piggyback onto an existing
 * customer TracerProvider instead of clobbering it.
 *
 * The JS OTel API always registers a singleton ProxyTracerProvider globally and
 * hides the real provider behind it, so `trace.getTracerProvider()` returns a
 * ProxyTracerProvider in EVERY case (unset AND customer-set). The old detection
 * compared `constructor.name === "ProxyTracerProvider"` and so ALWAYS took the
 * register-own branch — dead piggyback code. Concretely, when a customer already
 * had a provider, our `provider.register()` still ran: the duplicate
 * `setGlobalTracerProvider` failed silently (a diag error) but the trailing
 * `propagation.setGlobalPropagator(...)` inside register() overwrote the
 * customer's global propagator — an unwanted side effect.
 *
 * The fix resolves the REAL provider via the proxy's `getDelegate()`:
 * - no real provider (Noop sentinel delegate) → register() ours (happy path);
 * - a real customer provider → do NOT register; attach our span processor to it;
 * - a customer provider without addSpanProcessor → skip silently;
 * - a hostile probe (getDelegate throws) → fail-open to register-own.
 *
 * PRE-FIX EVIDENCE (test 2 "customer provider present"): under the old block
 * `provider.register()` fired → `trace.setGlobalTracerProvider` WAS called during
 * setup (spy asserts 0) AND our TokenPoliceSpanProcessor was NEVER attached to the
 * customer's provider → both assertions in test 2 FAIL pre-fix.
 *
 * OTel global state is process-global; each test resets it via
 * unsetupOpenTelemetry() + trace.disable() (fresh ProxyTracerProvider).
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { trace, context } from "@opentelemetry/api";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import {
  setupOpenTelemetry,
  unsetupOpenTelemetry,
  TokenPoliceSpanProcessor,
} from "../src/telemetry";

function hasOurProcessor(provider: any): boolean {
  // SDK 1.x kept `_registeredSpanProcessors`; 2.x holds a MultiSpanProcessor
  // at `_activeSpanProcessor` with the list in `_spanProcessors`.
  const procs =
    provider?._registeredSpanProcessors ??
    provider?._activeSpanProcessor?._spanProcessors;
  return (
    Array.isArray(procs) &&
    procs.some((p: any) => p instanceof TokenPoliceSpanProcessor)
  );
}

function resetOtelState(): void {
  try {
    unsetupOpenTelemetry();
  } catch {
    /* ignore */
  }
  try {
    // Fresh ProxyTracerProvider + unregister global — full OTel reset.
    trace.disable();
  } catch {
    /* ignore */
  }
  try {
    context.disable();
  } catch {
    /* ignore */
  }
}

beforeEach(() => {
  resetOtelState();
});

afterEach(() => {
  resetOtelState();
  vi.restoreAllMocks();
});

describe("OTel coexistence: setupOpenTelemetry provider registration", () => {
  // Test 1 — no customer provider (fresh proxy state): setup registers our
  // provider globally, so the proxy delegate becomes ours (a BasicTracerProvider
  // carrying our TokenPoliceSpanProcessor). Pins today's happy path.
  it("no customer provider: registers our provider globally", () => {
    expect(() => setupOpenTelemetry()).not.toThrow();

    const proxy: any = trace.getTracerProvider();
    const delegate: any = proxy.getDelegate();
    expect(delegate.constructor.name).toBe("BasicTracerProvider");
    // The delegate is OURS — it carries the TokenPoliceSpanProcessor.
    expect(hasOurProcessor(delegate)).toBe(true);
  });

  // Test 2 — customer provider present AND exposing addSpanProcessor (an OTel
  // SDK 1.x-shaped provider): setup must NOT attempt registration (no
  // duplicate-registration side effect), the customer's provider REMAINS the
  // delegate, and our TokenPoliceSpanProcessor is attached to it.
  it("customer provider present (addSpanProcessor): piggybacks without re-registering", () => {
    const attached: any[] = [];
    const customer: any = {
      getTracer: () => trace.getTracer("customer"),
      addSpanProcessor: (p: any) => attached.push(p),
    };
    trace.setGlobalTracerProvider(customer);

    // Spy AFTER the customer's own registration — asserts our setup does not
    // call setGlobalTracerProvider (i.e. no register-own path runs).
    const regSpy = vi.spyOn(trace, "setGlobalTracerProvider");

    expect(() => setupOpenTelemetry()).not.toThrow();

    // No re-registration attempt during our setup.
    expect(regSpy).not.toHaveBeenCalled();
    // Customer's provider is still the delegate (untouched).
    expect((trace.getTracerProvider() as any).getDelegate()).toBe(customer);
    // Our processor was attached to the customer's provider.
    expect(attached.some((p) => p instanceof TokenPoliceSpanProcessor)).toBe(true);
  });

  // Test 2b — customer provider from OTel SDK 2.x: processors are constructor-
  // only (no addSpanProcessor), so setup takes the documented skip branch —
  // no throw, no registration attempt, the customer's provider and its
  // processor list are left exactly as they were.
  it("customer 2.x BasicTracerProvider (constructor-only): skipped, globals untouched", () => {
    const customer = new BasicTracerProvider();
    trace.setGlobalTracerProvider(customer);
    const before = (customer as any)._activeSpanProcessor?._spanProcessors?.length;

    const regSpy = vi.spyOn(trace, "setGlobalTracerProvider");

    expect(() => setupOpenTelemetry()).not.toThrow();

    expect(regSpy).not.toHaveBeenCalled();
    expect((trace.getTracerProvider() as any).getDelegate()).toBe(customer);
    expect(hasOurProcessor(customer)).toBe(false);
    expect((customer as any)._activeSpanProcessor?._spanProcessors?.length).toBe(before);
  });

  // Test 3 — customer provider WITHOUT addSpanProcessor (minimal delegate object):
  // no throw, no registration attempt, silently skipped (delegate untouched).
  it("customer provider without addSpanProcessor: skipped silently", () => {
    const minimal: any = { getTracer: () => trace.getTracer("noop") };
    trace.setGlobalTracerProvider(minimal);

    const regSpy = vi.spyOn(trace, "setGlobalTracerProvider");

    expect(() => setupOpenTelemetry()).not.toThrow();

    expect(regSpy).not.toHaveBeenCalled();
    expect((trace.getTracerProvider() as any).getDelegate()).toBe(minimal);
  });

  // Test 4 — hostile probe (getDelegate throws): detection is fail-open and
  // degrades to register-own without throwing.
  it("hostile getDelegate that throws: degrades to register-own, no throw", () => {
    const hostile: any = {
      getDelegate() {
        throw new Error("boom: hostile getDelegate");
      },
    };
    const getSpy = vi
      .spyOn(trace, "getTracerProvider")
      .mockReturnValue(hostile);

    expect(() => setupOpenTelemetry()).not.toThrow();
    expect(getSpy).toHaveBeenCalled();

    // Restore so we can observe the real global state: register-own happened.
    getSpy.mockRestore();
    const delegate: any = (trace.getTracerProvider() as any).getDelegate();
    expect(delegate.constructor.name).toBe("BasicTracerProvider");
    expect(hasOurProcessor(delegate)).toBe(true);
  });
});
