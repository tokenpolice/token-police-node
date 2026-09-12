/**
 * TokenPolice Node.js SDK
 *
 * Real-time budget enforcement and token usage monitoring for LLM applications.
 *
 * Architecture:
 * 1. Passive Token Extraction — Local OpenTelemetry via OpenLLMetry-JS instrumentors.
 * No data ever leaves via OpenTelemetry exporters.
 * 2. Active Budget Enforcement — Pre-flight check() before every LLM request.
 * Fails open on any error. Only blocks when explicitly configured.
 *
 * @example
 * ```typescript
 * import * as tp from 'token-police';
 *
 * tp.init({
 * apiKey: 'tp_sk_your_api_key',
 * baseUrl: 'http://localhost:3001',
 * firewall: 'enforce',
 * });
 *
 * // All OpenAI/Anthropic calls are now automatically tracked and enforced
 * ```
 *
 * @module token-police
 */

// Core
export { init, flush, flushSync, shutdown, TokenPolice } from "./client";
export type {
  TokenPoliceOptions,
  CheckResult,
  FirewallMode,
  RerouteDirective,
  ErrorDetailMode,
} from "./client";
export type { InstrumentModules } from "./telemetry";

// Context
export {
  session,
  agent,
  chain,
  workflow,
  serverless,
  TPSession,
  getCurrentSession,
  setSpanName,
  tool,
  toolSpan,
} from "./context";
export type { SessionOptions, WorkflowOptions, ToolSpanOptions } from "./context";

// Enforcer
export { protect, uninstrument, tokenPoliceAiSdkMiddleware } from "./enforcer";

// State
export { getClient } from "./state";

// Exceptions
export { TokenPoliceBlockedError } from "./exceptions";
