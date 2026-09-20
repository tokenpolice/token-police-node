import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    include: ["tests/**/*.test.ts"],
    // Keeps every test off the real collector; see tests/setup.ts.
    setupFiles: ["tests/setup.ts"],
    testTimeout: 10000,
  },
});
