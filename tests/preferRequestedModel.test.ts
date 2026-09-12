/**
 * One rerouted model, one model string.
 *
 * The auto-instrumented span path reports `gen_ai.request.model` (the
 * post-rewrite alias, e.g. `claude-haiku-4-5`) while every manual/streaming tap
 * goes through `_logManual` and reads the provider's ECHO (the dated snapshot
 * the alias resolves to, `claude-haiku-4-5-20251001`). Cost-by-model panels saw
 * two models for one. `preferRequestedModel` collapses the echo back onto the
 * request — family-gated, so a provider that genuinely served something else
 * (gateway auto-routing) keeps its echo.
 */
import { describe, it, expect, vi, afterEach } from "vitest";
import { init } from "../src/client";
import { getClient } from "../src/state";
import { protect, uninstrument } from "../src/enforcer";
import { preferRequestedModel } from "../src/rerouteNoop";

const API_KEY = "tp_sk_f5_test";

afterEach(() => {
  try {
    uninstrument();
  } catch {
    /* ignore */
  }
  const c = getClient();
  if (c) c.closeSync();
  vi.restoreAllMocks();
});

describe("PreferRequestedModel", () => {
  it("dated-snapshot echo of the requested alias → the alias wins", () => {
    expect(preferRequestedModel("claude-haiku-4-5", "claude-haiku-4-5-20251001")).toBe(
      "claude-haiku-4-5",
    );
  });

  it("openai date form", () => {
    expect(preferRequestedModel("gpt-4o-mini", "gpt-4o-mini-2024-07-18")).toBe("gpt-4o-mini");
  });

  it("vertex @-date form", () => {
    expect(preferRequestedModel("claude-3-sonnet", "claude-3-sonnet@20240229")).toBe(
      "claude-3-sonnet",
    );
  });

  it("identical strings → the extracted model, unchanged", () => {
    expect(preferRequestedModel("gpt-4o-mini", "gpt-4o-mini")).toBe("gpt-4o-mini");
  });

  it("a genuinely different served model keeps its echo (gateway auto-routing)", () => {
    expect(preferRequestedModel("openrouter/auto", "anthropic/claude-3-haiku")).toBe(
      "anthropic/claude-3-haiku",
    );
    expect(preferRequestedModel("gpt-4o-mini", "gpt-4o-2024-08-06")).toBe("gpt-4o-2024-08-06");
  });

  it("a pinned snapshot request is never rewritten", () => {
    expect(
      preferRequestedModel("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
    ).toBe("claude-haiku-4-5-20251001");
  });

  it("non-date suffixes are not a family match", () => {
    expect(preferRequestedModel("gemini-1.5-pro", "gemini-1.5-pro-002")).toBe(
      "gemini-1.5-pro-002",
    );
    expect(preferRequestedModel("claude-3-5-sonnet", "claude-3-5-sonnet-latest")).toBe(
      "claude-3-5-sonnet-latest",
    );
  });

  it("case/whitespace-insensitive match returns the ORIGINAL requested string", () => {
    expect(preferRequestedModel(" Claude-Haiku-4-5 ", "CLAUDE-HAIKU-4-5-20251001")).toBe(
      " Claude-Haiku-4-5 ",
    );
  });

  it("garbage requested input → the extracted model, unchanged (never throws)", () => {
    const extracted = "claude-haiku-4-5-20251001";
    expect(preferRequestedModel(undefined, extracted)).toBe(extracted);
    expect(preferRequestedModel(null, extracted)).toBe(extracted);
    expect(preferRequestedModel("", extracted)).toBe(extracted);
    expect(preferRequestedModel("   ", extracted)).toBe(extracted);
    expect(preferRequestedModel(42, extracted)).toBe(extracted);
    expect(preferRequestedModel({}, extracted)).toBe(extracted);
    expect(preferRequestedModel(["claude-haiku-4-5"], extracted)).toBe(extracted);
  });

  it("empty / non-string extracted model is returned as-is", () => {
    expect(preferRequestedModel("claude-haiku-4-5", "")).toBe("");
    expect(preferRequestedModel("claude-haiku-4-5", undefined as any)).toBe(undefined);
  });
});

// NEW-1: on Bedrock the model id IS the price — a cross-region inference
// profile (`us.anthropic.claude-haiku-4-5-20251001-v1:0`) bills at a different
// rate than its bare echo (`claude-haiku-4-5-20251001`). The gate must widen to
// treat the echo as same-family (so the profile id is reported verbatim), but
// ONLY when the requested id is unmistakably Bedrock-shaped — a genuinely
// different served model, or a look-alike that isn't Bedrock-shaped, must not
// widen.
describe("PreferRequestedModel — Bedrock family gate", () => {
  it("CRIS profile id, bare-model echo → the profile id wins (the prod bug)", () => {
    expect(
      preferRequestedModel(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-haiku-4-5-20251001",
      ),
    ).toBe("us.anthropic.claude-haiku-4-5-20251001-v1:0");
  });

  it("non-CRIS (no region prefix) Bedrock id, bare-model echo → the requested id wins", () => {
    expect(
      preferRequestedModel(
        "anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-haiku-4-5-20251001",
      ),
    ).toBe("anthropic.claude-haiku-4-5-20251001-v1:0");
  });

  it("echo is the region-less form (vendor + version kept) → the profile id wins", () => {
    expect(
      preferRequestedModel(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic.claude-haiku-4-5-20251001-v1:0",
      ),
    ).toBe("us.anthropic.claude-haiku-4-5-20251001-v1:0");
  });

  it("amazon vendor, versioned echo dropped → the profile id wins", () => {
    expect(preferRequestedModel("amazon.nova-lite-v1:0", "nova-lite-v1:0")).toBe(
      "amazon.nova-lite-v1:0",
    );
  });

  it("amazon CRIS + versioned echo dropped → the profile id wins", () => {
    expect(preferRequestedModel("us.amazon.nova-lite-v1:0", "nova-lite-v1:0")).toBe(
      "us.amazon.nova-lite-v1:0",
    );
  });

  it("eu region, meta vendor → the profile id wins", () => {
    expect(
      preferRequestedModel("eu.meta.llama3-2-90b-instruct-v1:0", "llama3-2-90b-instruct-v1:0"),
    ).toBe("eu.meta.llama3-2-90b-instruct-v1:0");
  });

  it("global region + dated-snapshot echo (both wrappers peeled) → the profile id wins", () => {
    expect(
      preferRequestedModel(
        "global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-sonnet-4-5-20250929",
      ),
    ).toBe("global.anthropic.claude-sonnet-4-5-20250929-v1:0");
  });

  it("case/whitespace-insensitive match returns the ORIGINAL requested string verbatim", () => {
    expect(
      preferRequestedModel(
        " US.Anthropic.Claude-Haiku-4-5-20251001-v1:0 ",
        "claude-haiku-4-5-20251001",
      ),
    ).toBe(" US.Anthropic.Claude-Haiku-4-5-20251001-v1:0 ");
  });

  it("a genuinely different served model on Bedrock keeps its echo (no over-widening)", () => {
    expect(
      preferRequestedModel("us.anthropic.claude-haiku-4-5-20251001-v1:0", "claude-sonnet-4-6"),
    ).toBe("claude-sonnet-4-6");
    expect(preferRequestedModel("us.anthropic.claude-haiku-4-5-20251001-v1:0", "gpt-4o")).toBe(
      "gpt-4o",
    );
  });

  it("unknown vendor namespace → not Bedrock-shaped, no widening", () => {
    expect(preferRequestedModel("us.notavendor.some-model-v1:0", "some-model")).toBe(
      "some-model",
    );
    expect(preferRequestedModel("notavendor.some-model-v1:0", "some-model")).toBe("some-model");
  });

  it("region-shaped-but-not-a-CRIS-region token → no widening", () => {
    expect(preferRequestedModel("us-east-1.anthropic.claude-haiku-4-5", "claude-haiku-4-5")).toBe(
      "claude-haiku-4-5",
    );
  });

  // Every CRIS region token in BEDROCK_REGION_PREFIX, not just `us.` — regex
  // alternation backtracks, so listing order between `us-gov` and `us` is not
  // load-bearing; what matters is that the FULL region token is consumed
  // (never a partial match like `us` alone in front of `-gov.…`).
  it("every CRIS region token widens correctly (apac, global, us-gov)", () => {
    expect(
      preferRequestedModel(
        "apac.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-sonnet-4-5-20250929",
      ),
    ).toBe("apac.anthropic.claude-sonnet-4-5-20250929-v1:0");
    expect(
      preferRequestedModel(
        "global.amazon.nova-lite-v1:0",
        "nova-lite-v1:0",
      ),
    ).toBe("global.amazon.nova-lite-v1:0");
    expect(
      preferRequestedModel(
        "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-sonnet-4-5-20250929",
      ),
    ).toBe("us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0");
  });

  it("us-gov strips the WHOLE region token, not just \"us-\"", () => {
    // Guards against a future implementation that stops short of the full
    // `us-gov.` token (e.g. a naive split on the first `-`): if only `us`
    // were peeled, the vendor-namespace gate would see `gov.anthropic....`
    // (not a real vendor prefix) and refuse to widen — the echo would stay
    // unchanged. Widening here proves the full `us-gov.` token was consumed.
    expect(
      preferRequestedModel(
        "us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-sonnet-4-5-20250929",
      ),
    ).toBe("us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0");
  });

  it("only ONE region prefix is peeled — a doubled prefix does not widen", () => {
    // `bedrockFamilyCandidates` strips a single region layer, then requires a
    // real vendor namespace immediately after it. A malformed doubled prefix
    // (`us.eu.…`) leaves `eu.anthropic….` after the one strip, which is not a
    // recognized vendor namespace, so NO candidates are built — the echo must
    // be kept verbatim, not the malformed requested string.
    expect(
      preferRequestedModel(
        "us.eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "claude-sonnet-4-5-20250929",
      ),
    ).toBe("claude-sonnet-4-5-20250929");
  });

  it("case-insensitive region token still widens (US.ANTHROPIC…)", () => {
    expect(
      preferRequestedModel(
        "US.ANTHROPIC.claude-haiku-4-5-20251001-v1:0",
        "claude-haiku-4-5-20251001",
      ),
    ).toBe("US.ANTHROPIC.claude-haiku-4-5-20251001-v1:0");
  });

  // T3: BEDROCK_VERSION_SUFFIX widened to accept a BARE numeric version tag
  // (no `v`) — `us.openai.gpt-oss-120b-1:0` is a real catalog id. The `v` is
  // optional only when a `:<n>` part follows; that's what keeps a dated
  // snapshot suffix (`-20251001`, no colon) from being swallowed as a version.
  it("bare-numeric version tag (no \"v\") widens correctly", () => {
    expect(preferRequestedModel("us.openai.gpt-oss-120b-1:0", "gpt-oss-120b")).toBe(
      "us.openai.gpt-oss-120b-1:0",
    );
  });

  it("explicit v + colon-minor version tag widens correctly", () => {
    expect(
      preferRequestedModel("us.mistral.mistral-7b-instruct-v0:2", "mistral-7b-instruct"),
    ).toBe("us.mistral.mistral-7b-instruct-v0:2");
  });

  it("a dated snapshot suffix is NEVER treated as a version tag (no over-widening)", () => {
    // No colon and no leading "v" — must fail BOTH branches of
    // BEDROCK_VERSION_SUFFIX. If it were ever swallowed, `claude-haiku-4-5`
    // would land in the candidate set and this would incorrectly widen.
    expect(
      preferRequestedModel("anthropic.claude-haiku-4-5-20251001", "claude-haiku-4-5"),
    ).toBe("claude-haiku-4-5");
  });
});

// ── _logManual level: the payload model the server actually receives ──
class FakeAnthropicClient {
  async create(params: any): Promise<any> {
    return {
      // The provider echoes the dated snapshot the alias resolves to.
      model: (this as any)._echo ?? params.model,
      usage: { input_tokens: 10, output_tokens: 5 },
    };
  }
}

async function loggedModelFor(requested: string, echo: string): Promise<string> {
  const client = init({ apiKey: API_KEY, baseUrl: "http://localhost:9", firewall: "off" });
  const logSpy = vi.spyOn(client, "log").mockImplementation(() => {});
  protect("anthropic-fake", ["prototype"], "create", true, {
    manual: true,
    provider: "anthropic",
    module: FakeAnthropicClient,
  });
  const instance = new FakeAnthropicClient();
  (instance as any)._echo = echo;
  await instance.create({ model: requested, messages: [{ role: "user", content: "hi" }] });
  expect(logSpy).toHaveBeenCalledTimes(1);
  return logSpy.mock.calls[0][4] as string;
}

describe("_logManual reports the requested model", () => {
  it("alias requested, dated snapshot echoed → the alias is logged", async () => {
    expect(await loggedModelFor("claude-haiku-4-5", "claude-haiku-4-5-20251001")).toBe(
      "claude-haiku-4-5",
    );
  });

  it("customer pinned the snapshot → logged unchanged", async () => {
    expect(
      await loggedModelFor("claude-haiku-4-5-20251001", "claude-haiku-4-5-20251001"),
    ).toBe("claude-haiku-4-5-20251001");
  });

  it("provider served a different model → the echo is logged", async () => {
    expect(await loggedModelFor("claude-haiku-4-5", "claude-sonnet-4-5-20250929")).toBe(
      "claude-sonnet-4-5-20250929",
    );
  });

  // NEW-1: proves the fix at the wire level, not just in the helper — a
  // Bedrock cross-region inference profile request whose echo is the bare
  // model id must reach the /log payload as the PROFILE ID, since that's
  // what AWS actually bills.
  it("Bedrock CRIS profile requested, bare-model echo → the profile id is logged", async () => {
    expect(
      await loggedModelFor(
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "claude-haiku-4-5-20251001",
      ),
    ).toBe("us.anthropic.claude-haiku-4-5-20251001-v1:0");
  });
});
