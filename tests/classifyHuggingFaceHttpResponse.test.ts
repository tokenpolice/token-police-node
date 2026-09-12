/**
 * F-17-C — `_directStatusOf` reads the `@huggingface/inference` v4
 * `httpResponse.status`.
 *
 * `@huggingface/inference` v4 errors (`InferenceClientProviderApiError` /
 * `InferenceClientHubApiError`, both extending `InferenceClientHttpRequestError`)
 * carry the HTTP status ONLY on the `httpResponse` bag
 * (`{ requestId, status, body }`): no top-level `status` / `statusCode`, no
 * `.response`, and a message ("Failed to perform inference: Invalid credentials
 * in Authorization header") with no parsable three-digit code for the message
 * regex to recover. Every such failure therefore classified unknown/0 — an HF
 * 401 shipped a failure row with no error_kind and no http_status.
 *
 * The read is last in `_directStatusOf` (so a real top-level status, a
 * `.response` status, and the AWS `$metadata` read all still win) and guarded
 * `typeof === "number"` + `Number.isInteger` + 100..600, so the non-status
 * shapes the bag can hold (`requestId`, a string status, a body payload) can
 * never be mistaken for one. Only `.status` is read — `httpResponse.body` is
 * the raw provider error payload and stays untouched (token-metadata-only).
 *
 * `_classify` is pure telemetry and must stay TOTAL / NO-THROW — it runs inside
 * the SDK's failure path, never into customer code.
 */
import { describe, it, expect } from "vitest";
import { classifyException } from "../src/_classify";

/**
 * `@huggingface/inference` v4 error stand-in: status lives only on
 * `httpResponse`. Note the real class sets `.name` to the SHORT name
 * ("ProviderApiError") while `constructor.name` is the long
 * "InferenceClientProviderApiError" — neither is in any classifier name set,
 * so the status read is the only signal available.
 */
class InferenceClientProviderApiError extends Error {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  httpResponse: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(message: string, httpResponse: any) {
    super(message);
    this.name = "ProviderApiError";
    this.httpResponse = httpResponse;
  }
}

/** Sibling shape thrown for huggingface.co (hub) calls rather than provider calls. */
class InferenceClientHubApiError extends Error {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  httpResponse: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(message: string, httpResponse: any) {
    super(message);
    this.name = "HubApiError";
    this.httpResponse = httpResponse;
  }
}

describe("classifyException — @huggingface/inference httpResponse.status", () => {
  it("classifies a ProviderApiError 401 as auth_error/401", () => {
    const err = new InferenceClientProviderApiError(
      "Failed to perform inference: Invalid credentials in Authorization header",
      {
        requestId: "x",
        status: 401,
        body: { error: "Invalid credentials in Authorization header" },
      },
    );
    expect(classifyException(err)).toEqual({ error_kind: "auth_error", http_status: 401 });
  });

  it("classifies a HubApiError 403 as auth_error/403", () => {
    const err = new InferenceClientHubApiError(
      "Failed to perform inference: insufficient permissions for this repo",
      { requestId: "y", status: 403, body: { error: "insufficient permissions" } },
    );
    expect(classifyException(err)).toEqual({ error_kind: "auth_error", http_status: 403 });
  });

  it("classifies a 429 as rate_limited", () => {
    const err = new InferenceClientProviderApiError("Failed to perform inference: rate limited", {
      requestId: "z",
      status: 429,
      body: { error: "Rate limit reached" },
    });
    expect(classifyException(err)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  it("classifies a 500 as server_error and a 400 as client_error", () => {
    const server = new InferenceClientProviderApiError("Failed to perform inference: server error", {
      requestId: "a",
      status: 500,
      body: "internal error",
    });
    expect(classifyException(server)).toEqual({ error_kind: "server_error", http_status: 500 });

    const client = new InferenceClientProviderApiError("Failed to perform inference: bad input", {
      requestId: "b",
      status: 400,
      body: { error: "Input validation error" },
    });
    expect(classifyException(client)).toEqual({ error_kind: "client_error", http_status: 400 });
  });

  it("ignores a STRING status (never coerced)", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const err: any = new Error("Failed to perform inference");
    err.httpResponse = { requestId: "x", status: "401", body: {} };
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("ignores a BOOLEAN status", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const err: any = new Error("Failed to perform inference");
    err.httpResponse = { requestId: "x", status: true, body: {} };
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("ignores out-of-range statuses", () => {
    for (const v of [99, 600]) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const err: any = new Error("Failed to perform inference");
      err.httpResponse = { requestId: "x", status: v, body: {} };
      expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
    }
  });

  it("ignores a NaN status", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const err: any = new Error("Failed to perform inference");
    err.httpResponse = { requestId: "x", status: NaN, body: {} };
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("ignores a non-integer status", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const err: any = new Error("Failed to perform inference");
    err.httpResponse = { requestId: "x", status: 401.5, body: {} };
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("stays unknown/0 when httpResponse is null, undefined, or not an object", () => {
    for (const bag of [null, undefined, "401"]) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const err: any = new Error("Failed to perform inference");
      err.httpResponse = bag;
      expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
    }
  });

  it("is TOTAL on a hostile httpResponse getter (never throws)", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const err: any = new Error("Failed to perform inference");
    Object.defineProperty(err, "httpResponse", {
      get() {
        throw new Error("nope");
      },
    });
    expect(() => classifyException(err)).not.toThrow();
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("is TOTAL on a hostile httpResponse.status getter (never throws)", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const bag: any = { requestId: "x", body: {} };
    Object.defineProperty(bag, "status", {
      get() {
        throw new Error("nope");
      },
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const err: any = new Error("Failed to perform inference");
    err.httpResponse = bag;
    expect(() => classifyException(err)).not.toThrow();
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("lets a top-level status outrank httpResponse (read is last)", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const err: any = new Error("Failed to perform inference");
    err.status = 500;
    err.httpResponse = { requestId: "x", status: 401, body: {} };
    expect(classifyException(err)).toEqual({ error_kind: "server_error", http_status: 500 });
  });

  it("lets a .response status outrank httpResponse", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const err: any = new Error("Failed to perform inference");
    err.response = { status: 429 };
    err.httpResponse = { requestId: "x", status: 401, body: {} };
    expect(classifyException(err)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  it("lets the AWS $metadata status outrank httpResponse", () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const err: any = new Error("Failed to perform inference");
    err.$metadata = { httpStatusCode: 403 };
    err.httpResponse = { requestId: "x", status: 500, body: {} };
    expect(classifyException(err)).toEqual({ error_kind: "auth_error", http_status: 403 });
  });

  it("recovers httpResponse from a WRAPPED error (chain pass re-runs the direct read)", () => {
    const inner = new InferenceClientProviderApiError(
      "Failed to perform inference: Invalid credentials in Authorization header",
      { requestId: "x", status: 401, body: { error: "Invalid credentials" } },
    );
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const top: any = new Error("wrapped");
    top.cause = inner;
    expect(classifyException(top)).toEqual({ error_kind: "auth_error", http_status: 401 });
  });

  it("is TOTAL on null/undefined input", () => {
    expect(classifyException(null)).toEqual({ error_kind: "unknown", http_status: 0 });
    expect(classifyException(undefined)).toEqual({ error_kind: "unknown", http_status: 0 });
  });
});
