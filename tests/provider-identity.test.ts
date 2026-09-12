/**
 * Locks the SDK side of provider-aware pricing: the enforcer must extract the
 * bound client's base URL as host+path (NO credentials, NO query) so the
 * server can identify the serving provider. All mapping logic lives in the
 * server; the SDK only forwards the sanitized string.
 *
 * Mirrors the Python tests/test_provider_identity.py. Covers the gap that the
 * server-side e2e spec can't (it POSTs api_base straight to /log, bypassing
 * the SDK).
 */
import { describe, it, expect } from "vitest";
import { _extractBaseURL } from "../src/enforcer";

const withClient = (baseURL: string) => ({ _client: { baseURL } });

describe("_extractBaseURL", () => {
  it("extracts scheme + host + path from the bound client's base URL", () => {
    expect(_extractBaseURL(withClient("https://api.minimax.io/anthropic"))).toBe(
      "https://api.minimax.io/anthropic",
    );
    expect(_extractBaseURL(withClient("https://api.together.xyz/v1"))).toBe(
      "https://api.together.xyz/v1",
    );
  });

  it("also reads a top-level baseURL (not just ._client.baseURL)", () => {
    expect(_extractBaseURL({ baseURL: "https://api.fireworks.ai/inference/v1" })).toBe(
      "https://api.fireworks.ai/inference/v1",
    );
  });

  it("strips embedded credentials and the query string (only host metadata leaves)", () => {
    expect(_extractBaseURL(withClient("https://user:secret@api.minimax.io/x?k=v"))).toBe(
      "https://api.minimax.io/x",
    );
  });

  it("keeps a non-default port", () => {
    expect(_extractBaseURL(withClient("http://localhost:8000/v1"))).toBe(
      "http://localhost:8000/v1",
    );
  });

  it("is fail-safe — returns '' for missing/odd input rather than throwing", () => {
    expect(_extractBaseURL(undefined)).toBe("");
    expect(_extractBaseURL({})).toBe("");
    expect(_extractBaseURL(withClient(""))).toBe("");
    expect(_extractBaseURL(withClient("not a url"))).toBe("");
  });
});
