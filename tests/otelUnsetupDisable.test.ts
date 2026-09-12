/**
 * unsetupOpenTelemetry must undo the global registrations setup performed — and
 * ONLY those.
 *
 * - When we registered our own provider (no customer provider present),
 * `provider.register()` installed our global tracer provider + global
 * propagator, and we installed our global context manager. unsetup must
 * disable all three so a later re-init starts clean and no stale delegate
 * points at our shut-down provider.
 * - When a customer provider already exists we piggyback and never call
 * register(); unsetup must NOT disable the customer's globals. Likewise a
 * customer-owned context manager (our setGlobalContextManager returned false)
 * must be left intact.
 *
 * OTel global state is process-global; reset between tests.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";
import { trace, context, propagation, ROOT_CONTEXT } from "@opentelemetry/api";
import { BasicTracerProvider } from "@opentelemetry/sdk-trace-base";
import { setupOpenTelemetry, unsetupOpenTelemetry } from "../src/telemetry";

function resetOtelState(): void {
  try {
    unsetupOpenTelemetry();
  } catch {
    /* ignore */
  }
  try {
    trace.disable();
  } catch {
    /* ignore */
  }
  try {
    context.disable();
  } catch {
    /* ignore */
  }
  try {
    propagation.disable();
  } catch {
    /* ignore */
  }
}

// A minimal but valid ContextManager so a "customer" can own the global one.
function makeNoopContextManager() {
  return {
    active: () => ROOT_CONTEXT,
    with: (_ctx: any, fn: any, thisArg?: any, ...args: any[]) =>
      fn.call(thisArg, ...args),
    bind: (_ctx: any, target: any) => target,
    enable() {
      return this;
    },
    disable() {
      return this;
    },
  };
}

beforeEach(() => resetOtelState());
afterEach(() => {
  resetOtelState();
  vi.restoreAllMocks();
});

describe("unsetupOpenTelemetry — disable only what we registered (fix 4)", () => {
  it("we registered globals: unsetup disables tracer provider, propagator, context manager", () => {
    setupOpenTelemetry();

    // Sanity: our provider is the live delegate after setup.
    const delegateAfterSetup: any = (trace.getTracerProvider() as any).getDelegate();
    expect(delegateAfterSetup.constructor.name).toBe("BasicTracerProvider");

    // Call-through spies (record + still run real disable so state stays clean).
    const traceSpy = vi.spyOn(trace, "disable");
    const propSpy = vi.spyOn(propagation, "disable");
    const ctxSpy = vi.spyOn(context, "disable");

    unsetupOpenTelemetry();

    expect(traceSpy).toHaveBeenCalledTimes(1);
    expect(propSpy).toHaveBeenCalledTimes(1);
    expect(ctxSpy).toHaveBeenCalledTimes(1);

    // Observable outcome: the proxy delegate is no longer our provider.
    const delegateAfterUnsetup: any = (trace.getTracerProvider() as any).getDelegate();
    expect(delegateAfterUnsetup.constructor.name).not.toBe("BasicTracerProvider");
  });

  it("piggyback mode (customer provider present): unsetup does NOT disable the customer's globals", () => {
    const customer = new BasicTracerProvider();
    trace.setGlobalTracerProvider(customer);

    setupOpenTelemetry(); // must piggyback, not register

    const traceSpy = vi.spyOn(trace, "disable");
    const propSpy = vi.spyOn(propagation, "disable");

    unsetupOpenTelemetry();

    // Customer's tracer provider + propagator were never registered by us, so
    // never disabled by us.
    expect(traceSpy).not.toHaveBeenCalled();
    expect(propSpy).not.toHaveBeenCalled();
    // Customer's provider is still the live delegate.
    expect((trace.getTracerProvider() as any).getDelegate()).toBe(customer);
  });

  it("customer-owned context manager: unsetup does NOT disable it", () => {
    const customer = new BasicTracerProvider();
    trace.setGlobalTracerProvider(customer);
    // Customer owns the global context manager BEFORE our setup, so our
    // setGlobalContextManager returns false and we record no ownership.
    context.setGlobalContextManager(makeNoopContextManager() as any);

    setupOpenTelemetry();

    const ctxSpy = vi.spyOn(context, "disable");
    unsetupOpenTelemetry();

    expect(ctxSpy).not.toHaveBeenCalled();
  });

  it("re-init after unsetup works cleanly (globals re-registered)", () => {
    setupOpenTelemetry();
    unsetupOpenTelemetry();
    // Second cycle must re-register our provider (flags were reset).
    expect(() => setupOpenTelemetry()).not.toThrow();
    const delegate: any = (trace.getTracerProvider() as any).getDelegate();
    expect(delegate.constructor.name).toBe("BasicTracerProvider");
  });
});
