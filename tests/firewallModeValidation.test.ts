import { describe, it, expect, vi, afterEach } from "vitest";
import { TokenPolice, type TokenPoliceOptions, type FirewallMode } from "../src/client";

// ── firewall mode string validation ───────────────────────
//
// An unrecognized `firewall` string must fall back to the SAFE default
// "dry_run" (never "enforce"), byte-for-byte and case-sensitively, so a typo
// can never silently enable live blocking. Valid modes pass through unchanged.

const API_KEY = "tp_sk_test123";

// Build options while bypassing the FirewallMode literal union so we can
// simulate a JS consumer passing an arbitrary runtime string. This mirrors
// real-world misuse (the union only guards TS callers).
function makeClient(opts: Partial<TokenPoliceOptions> & { firewall?: string }): TokenPolice {
  return new TokenPolice({ apiKey: API_KEY, ...opts } as TokenPoliceOptions);
}

describe("Firewall mode validation", () => {
  const created: TokenPolice[] = [];
  function track(c: TokenPolice): TokenPolice {
    created.push(c);
    return c;
  }

  afterEach(() => {
    while (created.length) created.pop()?.closeSync();
    vi.restoreAllMocks();
  });

  it("unknown mode falls back to dry_run", () => {
    // Items 1-4: unknown / wrong-case / other non-member / empty → dry_run.
    for (const bad of ["dryrun", "Enforce", "shadow", ""]) {
      const c = track(makeClient({ firewall: bad }));
      expect(c.firewall).toBe("dry_run");
    }
  });

  it("does not throw when an unknown mode is given", () => {
    // Item 11: init must complete (fail-open), never throw on a bad mode.
    expect(() => track(makeClient({ firewall: "totally-bogus" }))).not.toThrow();
    expect(() =>
      track(makeClient({ firewall: "totally-bogus", logErrors: true })),
    ).not.toThrow();
  });

  it("valid modes pass through byte-for-byte unchanged", () => {
    // Items 5-7: each canonical literal is preserved exactly.
    for (const good of ["enforce", "off", "dry_run"] as FirewallMode[]) {
      const c = track(makeClient({ firewall: good }));
      expect(c.firewall).toBe(good);
    }
  });

  it("default (no firewall, no enforce) stays dry_run", () => {
    // Item 8.
    const c = track(makeClient({}));
    expect(c.firewall).toBe("dry_run");
  });

  it("legacy enforce boolean alias is unchanged", () => {
    // Item 9: true→enforce, false→off, explicit firewall wins over enforce.
    expect(track(makeClient({ enforce: true })).firewall).toBe("enforce");
    expect(track(makeClient({ enforce: false })).firewall).toBe("off");
    expect(track(makeClient({ enforce: true, firewall: "off" })).firewall).toBe("off");
    // A bogus enforce-bool alias is impossible (boolean), but a bogus explicit
    // firewall still wins → falls back to dry_run, never the enforce alias.
    expect(track(makeClient({ enforce: true, firewall: "nope" })).firewall).toBe("dry_run");
  });

  it("warns UNCONDITIONALLY on an unknown mode (not gated behind logErrors)", () => {
    // An unknown firewall mode is a serious misconfiguration the developer must
    // always see — the warning fires regardless of logErrors (mirroring the
    // apiKey-format warning). It names the bad value + the dry_run fallback.
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    // logErrors omitted → still warns.
    track(makeClient({ firewall: "dryrun" }));
    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(String(warnSpy.mock.calls[0][0])).toContain("dryrun");
    expect(String(warnSpy.mock.calls[0][0])).toContain("dry_run");

    // logErrors: false → still warns.
    warnSpy.mockClear();
    track(makeClient({ firewall: "dryrun", logErrors: false }));
    expect(warnSpy).toHaveBeenCalledTimes(1);

    // logErrors: true → still exactly one firewall warn.
    warnSpy.mockClear();
    track(makeClient({ firewall: "dryrun", logErrors: true }));
    expect(warnSpy).toHaveBeenCalledTimes(1);

    // Valid mode → no firewall warn regardless of logErrors.
    warnSpy.mockClear();
    track(makeClient({ firewall: "enforce", logErrors: true }));
    expect(warnSpy).not.toHaveBeenCalled();
  });

  it("whitespace variant of a valid mode falls back to dry_run", () => {
    // Items 21: NO trim / NO case-fold — whitespace/newline variants of a
    // valid mode must NOT reach enforce.
    expect(track(makeClient({ firewall: " enforce " })).firewall).toBe("dry_run");
    expect(track(makeClient({ firewall: "enforce\n" })).firewall).toBe("dry_run");
    expect(track(makeClient({ firewall: " off " })).firewall).toBe("dry_run");
  });
});
