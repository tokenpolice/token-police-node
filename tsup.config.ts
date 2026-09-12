import { defineConfig } from "tsup";

export default defineConfig({
  entry: ["src/index.ts"],
  format: ["esm", "cjs"],
  dts: true,
  sourcemap: false,
  clean: true,
  splitting: false,
  external: [
    "@opentelemetry/api",
    "@opentelemetry/sdk-trace-base",
    "@traceloop/instrumentation-openai",
    "@traceloop/instrumentation-anthropic",
    "@traceloop/instrumentation-langchain",
    // Provider SDKs are runtime-resolved (app-first) — they are devDependencies
    // here for tests only and must NEVER be inlined into the bundle. A bundled
    // copy would be a third module record no app ever calls (see the
    // resolveProviderModule block in src/enforcer.ts).
    "@huggingface/inference",
    "@mistralai/mistralai",
  ],
});
