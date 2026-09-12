/**
 * Raised when an LLM call is blocked by TokenPolice firewall policy — budget
 * exhaustion, loop-detection, or any other firewall rule. This is the ONLY
 * exception that may propagate from TokenPolice into customer code, and only
 * when the firewall is in enforce mode on an explicit block decision.
 *
 * Structured fields let callers branch on the block without parsing `message`
 * (all optional — populated best-effort from the server / local decision):
 * - reason: human-readable block reason from the server.
 * - ruleId: ID of the firewall rule that blocked the call.
 * - kind: opaque block-type discriminator forwarded verbatim from the
 * server (e.g. a specific loop-detection signal, or "budget"
 * for a budget block). Treat as an opaque string, not a closed enum.
 * - traceId: trace ID associated with the block.
 */
export class TokenPoliceBlockedError extends Error {
  readonly reason?: string;
  readonly ruleId?: string;
  readonly kind?: string;
  readonly traceId?: string;

  constructor(
    message: string,
    details?: { reason?: string; ruleId?: string; kind?: string; traceId?: string },
  ) {
    super(message);
    this.name = "TokenPoliceBlockedError";
    this.reason = details?.reason;
    this.ruleId = details?.ruleId;
    this.kind = details?.kind;
    this.traceId = details?.traceId;
    // Restore prototype chain for instanceof checks in TypeScript
    Object.setPrototypeOf(this, TokenPoliceBlockedError.prototype);
  }
}
