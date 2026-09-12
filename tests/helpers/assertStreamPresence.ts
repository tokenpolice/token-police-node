/**
 * G4 presence invariants (sdk_test_hardening_design.md T-N3).
 *
 * Fail-open code converts failures into silence. Asserting only properties of
 * rows that exist cannot catch "row never created". After driving a
 * usage-bearing mock stream, call this on the log spy.
 */
import { expect } from "vitest";

/**
 * Extract /log call tuples for a provider. Node client.log signature:
 * log(userId, paidPlan, workflowName, sessionId, model, provider, input, output, …)
 * provider is args[5]; input tokens args[6]; output tokens args[7].
 */
export function logRowsForProvider(logSpy: { mock: { calls: any[][] } }, provider: string): any[][] {
  return logSpy.mock.calls.filter((c) => c[5] === provider);
}

/**
 * A wrapper driven with a usage-bearing mock stream MUST dispatch exactly
 * `expectedCount` log payload(s) for `provider` with tokens > 0.
 */
export function assertStreamedLogPresent(
  logSpy: { mock: { calls: any[][] } },
  provider: string,
  opts: { expectedCount?: number; minTokens?: number } = {},
): any[][] {
  const expectedCount = opts.expectedCount ?? 1;
  const minTokens = opts.minTokens ?? 1;
  const rows = logRowsForProvider(logSpy, provider);
  expect(
    rows.length,
    `G4 presence: expected ${expectedCount} ${provider} /log row(s), got ${rows.length}`,
  ).toBe(expectedCount);
  for (const r of rows) {
    const input = Number(r[6] ?? 0) || 0;
    const output = Number(r[7] ?? 0) || 0;
    expect(
      input + output,
      `G4 presence: ${provider} row must have tokens > 0 (input=${input} output=${output})`,
    ).toBeGreaterThanOrEqual(minTokens);
  }
  return rows;
}
