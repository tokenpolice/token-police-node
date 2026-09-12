/**
 * Fail-open wrappers.
 *
 * Higher-order functions that wrap sync/async functions, catching any exception
 * so the underlying LLM call proceeds — an SDK failure must never propagate
 * into the caller. Does not swallow TokenPoliceBlockedError — that is a
 * legitimate, explicitly-configured block.
 *
 * Behavior is kept in parity with the TokenPolice Python SDK.
 */

import { TokenPoliceBlockedError } from "./exceptions";
import { getClient } from "./state";

/**
 * Best-effort error report from a fail-open catch block. Fully guarded:
 * these catch blocks are the SDK's outermost fail-open boundary, so a throw
 * from the client lookup, a hostile `message` getter / `toString` on the
 * caught value, or a customer-replaced `console.error` must never escape
 * into the caller's LLM call. Mirrors the Python SDK's guarded handler.
 */
function reportFailOpen(prefix: string, e: unknown): void {
  try {
    const tp = getClient();
    if (tp && tp.logErrors) {
      console.error(`${prefix}: ${e instanceof Error ? e.message : String(e)}`);
    }
  } catch {
    // Swallow — reporting must never replace the fail-open contract.
  }
}

/**
 * Wraps a synchronous function with fail-open protection.
 * On any error (except TokenPoliceBlockedError), silently allows the call to proceed.
 */
export function failSafeSync<T extends (...args: any[]) => any>(
  fn: T,
): (...args: Parameters<T>) => ReturnType<T> | undefined {
  return (...args: Parameters<T>): ReturnType<T> | undefined => {
    try {
      return fn(...args);
    } catch (e) {
      if (e instanceof TokenPoliceBlockedError) {
        throw e;
      }
      reportFailOpen("TokenPolice instrumentation failed (fail-open)", e);
      return undefined;
    }
  };
}

/**
 * Wraps an async function with fail-open protection.
 * On any error (except TokenPoliceBlockedError), silently allows the call to proceed.
 */
export function failSafeAsync<T extends (...args: any[]) => Promise<any>>(
  fn: T,
): (...args: Parameters<T>) => Promise<Awaited<ReturnType<T>> | undefined> {
  return async (
    ...args: Parameters<T>
  ): Promise<Awaited<ReturnType<T>> | undefined> => {
    try {
      return await fn(...args);
    } catch (e) {
      if (e instanceof TokenPoliceBlockedError) {
        throw e;
      }
      reportFailOpen("TokenPolice async instrumentation failed (fail-open)", e);
      return undefined;
    }
  };
}
