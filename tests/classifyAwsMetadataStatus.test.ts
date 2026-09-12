/**
 * F-16-3c — `_directStatusOf` reads the AWS SDK v3 `$metadata.httpStatusCode`.
 *
 * AWS SDK v3 ServiceExceptions (`@aws-sdk/client-bedrock-runtime` and friends)
 * carry the HTTP status ONLY on the `$metadata` bag: no top-level `status` /
 * `statusCode`, no `.response`. Every such failure therefore classified
 * unknown/0 — a Bedrock `AccessDeniedException` shipped a failure row with no
 * error_kind and no http_status.
 *
 * The read is last in `_directStatusOf` (so a real top-level status still
 * wins) and guarded `typeof === "number"` + `Number.isInteger` + 100..600, so
 * the non-status shapes `$metadata` can hold (string codes, `attempts`,
 * `totalRetryDelay`) can never be mistaken for one.
 *
 * `_classify` is pure telemetry and must stay TOTAL / NO-THROW — it runs
 * inside the SDK's failure path, never into customer code.
 */
import { describe, it, expect } from "vitest";
import { classifyException } from "../src/_classify";

/** AWS SDK v3 ServiceException stand-in: status lives only on `$metadata`. */
class AccessDeniedException extends Error {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  $metadata: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  constructor(message: string, metadata: any) {
    super(message);
    this.name = "AccessDeniedException";
    this.$metadata = metadata;
  }
}

describe("classifyException — AWS SDK v3 $metadata.httpStatusCode", () => {
  it("classifies a bedrock AccessDeniedException as auth_error/403", () => {
    const err = new AccessDeniedException(
      "User is not authorized to perform bedrock:InvokeModel",
      { httpStatusCode: 403, requestId: "abc-123", attempts: 1, totalRetryDelay: 0 },
    );
    expect(classifyException(err)).toEqual({ error_kind: "auth_error", http_status: 403 });
  });

  it("classifies a throttling 429 as rate_limited", () => {
    const err: any = new Error("Too many requests");
    err.$metadata = { httpStatusCode: 429 };
    expect(classifyException(err)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  it("classifies a 500 as server_error and a 400 as client_error", () => {
    const server: any = new Error("internal");
    server.$metadata = { httpStatusCode: 500 };
    expect(classifyException(server)).toEqual({ error_kind: "server_error", http_status: 500 });

    const client: any = new Error("validation");
    client.$metadata = { httpStatusCode: 400 };
    expect(classifyException(client)).toEqual({ error_kind: "client_error", http_status: 400 });
  });

  it("ignores a STRING httpStatusCode (never coerced)", () => {
    const err: any = new Error("denied");
    err.$metadata = { httpStatusCode: "403" };
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("ignores a non-integer httpStatusCode", () => {
    const err: any = new Error("denied");
    err.$metadata = { httpStatusCode: 403.5 };
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("ignores an out-of-range httpStatusCode", () => {
    for (const v of [0, 99, 600, 1006]) {
      const err: any = new Error("denied");
      err.$metadata = { httpStatusCode: v };
      expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
    }
  });

  it("stays unknown/0 when $metadata is missing entirely", () => {
    expect(classifyException(new Error("some failure"))).toEqual({
      error_kind: "unknown",
      http_status: 0,
    });
  });

  it("stays unknown/0 when $metadata carries no httpStatusCode", () => {
    const err: any = new Error("denied");
    err.$metadata = { requestId: "abc-123", attempts: 3, totalRetryDelay: 120 };
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("stays unknown/0 when $metadata is not an object", () => {
    const err: any = new Error("denied");
    err.$metadata = 403;
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("lets a top-level status outrank $metadata (read is last)", () => {
    const err: any = new Error("denied");
    err.status = 500;
    err.$metadata = { httpStatusCode: 403 };
    expect(classifyException(err)).toEqual({ error_kind: "server_error", http_status: 500 });
  });

  it("lets a .response status outrank $metadata", () => {
    const err: any = new Error("denied");
    err.response = { status: 400 };
    err.$metadata = { httpStatusCode: 403 };
    expect(classifyException(err)).toEqual({ error_kind: "client_error", http_status: 400 });
  });

  it("recovers $metadata from a WRAPPED error (chain pass re-runs the direct read)", () => {
    const inner: any = new Error("Too many requests");
    inner.$metadata = { httpStatusCode: 429 };
    const top: any = new Error("stream iteration failed");
    top.cause = inner;
    expect(classifyException(top)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });

  it("is TOTAL on a hostile $metadata getter (never throws)", () => {
    const err: any = new Error("denied");
    Object.defineProperty(err, "$metadata", {
      get() {
        throw new Error("nope");
      },
    });
    expect(() => classifyException(err)).not.toThrow();
    expect(classifyException(err)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("is TOTAL on null/undefined input", () => {
    expect(classifyException(null)).toEqual({ error_kind: "unknown", http_status: 0 });
    expect(classifyException(undefined)).toEqual({ error_kind: "unknown", http_status: 0 });
  });

  it("leaves existing non-AWS classification unchanged", () => {
    const err: any = new Error("provider rate limited");
    err.status_code = 429;
    expect(classifyException(err)).toEqual({ error_kind: "rate_limited", http_status: 429 });
  });
});
