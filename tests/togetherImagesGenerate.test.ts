/**
 * Together-AI image seam — `Images.prototype.create` (together-ai <0.30) vs
 * `Images.prototype.generate` (together-ai ≥0.30, current 0.50).
 *
 * The registry used to target only `create`. On a current together-ai the
 * prototype carries `generate` alone, so the walker found nothing to wrap:
 * no pre-flight /check, no image_gen row, no cost — silent and fail-open.
 * Both method names are now registered; exactly one resolves on any given
 * install and _wrapMethod skips the other (method-not-found early return).
 *
 * HARNESS NOTES — mirrors tests/instrumentModulesClassNormalization.test.ts:
 * `init({ instrumentModules })` installs the wrapper, `client.check` /
 * `client.log` are spied so nothing hits the network, and the seam is proven
 * via the `check` spy (pre-flight fired) plus the `log` spy (image_gen row
 * with the together_image shape). Every fixture here is a hand-built fake
 * reproducing only the vendor CONTRACT the enforcer depends on:
 * `Together.Images.prototype.<method>` returning the
 * `{ id, model, object, data: [...] }` ImageFile.
 *
 * Fakes are the RIGHT tool for this file specifically: it has to drive the
 * `create`-only (<0.30) shape, which no installable together-ai provides any
 * more. `together-ai` did become a devDependency alongside the
 * realProviderSeams.* family, and `tests/realProviderSeams.together.test.ts`
 * covers the pair contract against the REAL client — the two are complements.
 */
import { describe, test, expect, afterEach, vi } from "vitest";
import { uninstrument } from "../src/enforcer";
import { init } from "../src/client";
import { setClient } from "../src/state";

const IMAGE_BODY = {
  model: "stabilityai/stable-diffusion-xl-base-1.0",
  prompt: "a red origami crane",
  n: 1,
};

function imageFile(body: any) {
  return {
    id: "img_1",
    model: body?.model,
    object: "list",
    data: [{ index: 0, type: "url", url: "https://example.invalid/1.png" }],
  };
}

/** together-ai ≥0.30 shape: Images.prototype has `generate`, no `create`. */
function makeTogetherWithGenerate() {
  class Images {
    async generate(body: any): Promise<any> {
      return imageFile(body);
    }
  }
  function Together(this: any) {
    this.images = new Images();
  }
  (Together as any).Images = Images;
  return { Together, Images };
}

/** together-ai <0.30 shape: Images.prototype has `create`, no `generate`. */
function makeTogetherWithCreate() {
  class Images {
    async create(body: any): Promise<any> {
      return imageFile(body);
    }
  }
  function Together(this: any) {
    this.images = new Images();
  }
  (Together as any).Images = Images;
  return { Together, Images };
}

/** Install the wrapper with the pre-flight stubbed ALLOWED, spying `check` + `log`. */
function initAllowed(instrumentModules: Record<string, any>) {
  const client = init({
    apiKey: "tp_sk_test",
    deployment: "serverless",
    firewall: "dry_run",
    instrumentModules,
  } as any);
  const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
  const checkSpy = vi
    .spyOn(client, "check")
    .mockResolvedValue({ status: "allowed", fail_open: false } as any);
  return { client, checkSpy, logSpy };
}

function expectImageGenLog(logSpy: ReturnType<typeof vi.spyOn>) {
  expect(logSpy).toHaveBeenCalledTimes(1);
  const args = logSpy.mock.calls[0] as any[];
  // Positional tp.log(userId, paidPlan, workflowName, sessionId, model, provider, …, opts)
  expect(args[4]).toBe(IMAGE_BODY.model);
  expect(args[5]).toBe("together");
  const opts = args[13];
  expect(opts.operation).toBe("image_gen");
  expect(opts.usage.shape).toBe("together_image");
  expect(opts.usage.items.images_generated).toBe(1);
  expect(opts.usage.items.image_model).toBe(IMAGE_BODY.model);
}

afterEach(() => {
  vi.restoreAllMocks();
  try {
    uninstrument();
  } catch {
    /* ignore */
  }
  try {
    setClient(undefined as any);
  } catch {
    /* ignore */
  }
});

describe("together-ai images seam: create (<0.30) and generate (≥0.30)", () => {
  test("only `generate` exists (together-ai ≥0.30) → wrapped: one /check, one image_gen log", async () => {
    const { Together, Images } = makeTogetherWithGenerate();
    expect(Object.getOwnPropertyNames(Images.prototype)).toEqual(["constructor", "generate"]);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});

    const { checkSpy, logSpy } = initAllowed({ together: { Together } });
    const resp = await new (Together as any)().images.generate(IMAGE_BODY);

    expect(resp.data).toHaveLength(1);
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expectImageGenLog(logSpy);
    // The absent `create` sibling is skipped silently (logErrors is off by default).
    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
    // Nothing was invented onto the prototype for the missing sibling.
    expect(Object.getOwnPropertyNames(Images.prototype)).toEqual(["constructor", "generate"]);
  });

  test("only `create` exists (together-ai <0.30) → still wrapped: one /check, one image_gen log", async () => {
    const { Together, Images } = makeTogetherWithCreate();
    expect(Object.getOwnPropertyNames(Images.prototype)).toEqual(["constructor", "create"]);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});

    const { checkSpy, logSpy } = initAllowed({ together: { Together } });
    const resp = await new (Together as any)().images.create(IMAGE_BODY);

    expect(resp.data).toHaveLength(1);
    expect(checkSpy).toHaveBeenCalledTimes(1);
    expectImageGenLog(logSpy);
    expect(warn).not.toHaveBeenCalled();
    expect(error).not.toHaveBeenCalled();
    expect(Object.getOwnPropertyNames(Images.prototype)).toEqual(["constructor", "create"]);
  });

  test("absent sibling never throws out of init(), even with logErrors on", () => {
    const { Together } = makeTogetherWithGenerate();
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() =>
      init({
        apiKey: "tp_sk_test",
        deployment: "serverless",
        firewall: "dry_run",
        logErrors: true,
        instrumentModules: { together: { Together } },
      } as any),
    ).not.toThrow();
    // Diagnostic-only under logErrors: at most a console.warn, never an error.
    expect(error).not.toHaveBeenCalled();
    void warn;
  });
});
