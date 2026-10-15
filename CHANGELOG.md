# Changelog

All notable changes to `token-police` (Node.js) are documented here.

## 1.0.1 — 2026-10-15

### Fixed

- Every Anthropic call was recorded twice, doubling its reported cost and the
  amount it counted against your budgets, once `@anthropic-ai/sdk` 0.134.0 or
  later was installed. That release emits its own OpenTelemetry spans by
  default, and TokenPolice counted them alongside its own. TokenPolice now
  ignores the Anthropic SDK's own spans, so each call is recorded once. Nothing
  to change in your app; earlier `@anthropic-ai/sdk` versions are unaffected.

## 1.0.0 — 2026-09

Initial public release.
