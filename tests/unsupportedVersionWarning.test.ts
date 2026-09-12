/**
 * B5 — silent version-cap skip.
 *
 * @traceloop/instrumentation-openai declares supportedVersions ">=4 <7" for the
 * `openai` module. On openai 7 the OTel loader hook matches the module name,
 * fails `isSupported(...)`, and `return exports;` — with no diag call at ANY
 * level. The enforcer's pre-flight check still runs, but usage extraction for
 * Chat/Completions relies on the traceloop span, so metering silently goes to
 * zero for the provider.
 *
 * warnIfInstrumentorSkippedModule() detects this WITHOUT semver parsing:
 * `_onRequire` assigns `def.moduleExports` iff the name matched AND the version
 * gate passed, so a matching def with no `moduleExports` after the module has
 * been required means "gate rejected the installed copy".
 *
 * The last describe pins that invariant against the REAL traceloop instrumentor
 * shape, so an upstream internals rename fails here rather than in production.
 */
import { describe, it, expect, vi } from "vitest";
import { createRequire } from "module";

import {
  resolveInstalledVersion,
  warnIfInstrumentorSkippedModule,
} from "../src/telemetry";

const requireCjs = createRequire(import.meta.url);

/** Dedupe state (`_warnedProviders`) is per-process, so every test uses its own name. */
const fakeInstrumentor = (def: any) => ({ _modules: [def] });

describe("warnIfInstrumentorSkippedModule — warns on a gate-rejected module", () => {
  it("warns once, naming the module, its version and the supported ranges", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = fakeInstrumentor({
      name: "fake-sdk-warns",
      supportedVersions: [">=4 <7"],
    });

    warnIfInstrumentorSkippedModule(inst, "fake-sdk-warns", () => "7.3.0");

    expect(warnSpy).toHaveBeenCalledTimes(1);
    const msg = String(warnSpy.mock.calls[0][0]);
    expect(msg).toContain("[TokenPolice Warning]");
    expect(msg).toContain("fake-sdk-warns");
    expect(msg).toContain("7.3.0");
    expect(msg).toContain(">=4 <7");
    expect(msg).toContain("will NOT be metered");
    warnSpy.mockRestore();
  });

  it("joins multiple supportedVersions ranges verbatim", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = fakeInstrumentor({
      name: "fake-sdk-multirange",
      supportedVersions: [">=4 <7", "^3.2.1"],
    });

    warnIfInstrumentorSkippedModule(inst, "fake-sdk-multirange", () => "7.0.0");

    const msg = String(warnSpy.mock.calls[0][0]);
    expect(msg).toContain(">=4 <7");
    expect(msg).toContain("^3.2.1");
    warnSpy.mockRestore();
  });

  it("omits the version when no getter is supplied", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = fakeInstrumentor({
      name: "fake-sdk-noversion",
      supportedVersions: [">=4 <7"],
    });

    warnIfInstrumentorSkippedModule(inst, "fake-sdk-noversion");

    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(String(warnSpy.mock.calls[0][0])).toContain("fake-sdk-noversion is disabled");
    warnSpy.mockRestore();
  });

  it("dedupes — a second call for the same module stays silent", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = fakeInstrumentor({
      name: "fake-sdk-dedupe",
      supportedVersions: [">=4 <7"],
    });

    warnIfInstrumentorSkippedModule(inst, "fake-sdk-dedupe", () => "7.0.0");
    warnIfInstrumentorSkippedModule(inst, "fake-sdk-dedupe", () => "7.0.0");

    expect(warnSpy).toHaveBeenCalledTimes(1);
    warnSpy.mockRestore();
  });
});

describe("warnIfInstrumentorSkippedModule — re-initialization vs version gate", () => {
  // unsetup() + a second init() re-constructs instrumentors, but
  // require-in-the-middle's per-module cache short-circuits before the new
  // instrumentor's hook: `moduleExports` stays unset even though the version is
  // fine, and the module really is left unpatched. Having seen the gate accept
  // this module earlier in the process is what tells the two causes apart.
  it("reports re-initialization, not an unsupported version, after an earlier accept", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    // init() #1 — the gate accepted the copy.
    warnIfInstrumentorSkippedModule(
      fakeInstrumentor({
        name: "fake-sdk-reinit",
        supportedVersions: [">=4 <7"],
        moduleExports: { ok: true },
      }),
      "fake-sdk-reinit",
      () => "6.1.0",
    );
    expect(warnSpy).not.toHaveBeenCalled();

    // init() #2 — freshly constructed instrumentor, hook never fires again.
    warnIfInstrumentorSkippedModule(
      fakeInstrumentor({ name: "fake-sdk-reinit", supportedVersions: [">=4 <7"] }),
      "fake-sdk-reinit",
      () => "6.1.0",
    );

    expect(warnSpy).toHaveBeenCalledTimes(1);
    const msg = String(warnSpy.mock.calls[0][0]);
    expect(msg).toContain("stopped after re-initialization");
    expect(msg).toContain("Restart the process to restore metering");
    expect(msg).not.toContain("not yet supported");
    warnSpy.mockRestore();
  });

  it("keeps the version-gate message when the gate never accepted this module", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    // A sibling module's acceptance must not leak into this one's diagnosis.
    warnIfInstrumentorSkippedModule(
      fakeInstrumentor({ name: "fake-sdk-sibling", moduleExports: {} }),
      "fake-sdk-sibling",
    );
    warnIfInstrumentorSkippedModule(
      fakeInstrumentor({
        name: "fake-sdk-never-accepted",
        supportedVersions: [">=4 <7"],
      }),
      "fake-sdk-never-accepted",
      () => "7.0.0",
    );

    expect(warnSpy).toHaveBeenCalledTimes(1);
    const msg = String(warnSpy.mock.calls[0][0]);
    expect(msg).toContain("the installed version 7.0.0 is not yet supported");
    expect(msg).not.toContain("re-initialization");
    warnSpy.mockRestore();
  });
});

describe("warnIfInstrumentorSkippedModule — stays silent when it cannot conclude", () => {
  it("does not warn when moduleExports is set (version gate accepted the copy)", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = fakeInstrumentor({
      name: "fake-sdk-accepted",
      supportedVersions: [">=4 <7"],
      moduleExports: { OpenAI: class {} },
    });

    warnIfInstrumentorSkippedModule(inst, "fake-sdk-accepted", () => "6.1.0");

    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it("does not warn when no def matches the target module name", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = fakeInstrumentor({
      name: "some-other-module",
      supportedVersions: [">=4 <7"],
    });

    warnIfInstrumentorSkippedModule(inst, "fake-sdk-nodef", () => "7.0.0");

    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it("treats one accepted def among several as instrumented", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst: any = {
      _modules: [
        { name: "fake-sdk-mixed", supportedVersions: [">=4 <7"] },
        { name: "fake-sdk-mixed", supportedVersions: [">=7"], moduleExports: {} },
      ],
    };

    warnIfInstrumentorSkippedModule(inst, "fake-sdk-mixed", () => "7.0.0");

    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});

describe("warnIfInstrumentorSkippedModule — fail-open on garbage", () => {
  it("never throws and never warns on malformed internals", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    const bad: any[] = [
      null,
      undefined,
      {},
      { _modules: "nope" },
      { _modules: [null, 42, {}] },
      { _modules: [{ name: 123 }] },
      { _modules: [{ name: { toString: () => "fake-sdk-garbage" } }] },
    ];
    for (const inst of bad) {
      expect(() =>
        warnIfInstrumentorSkippedModule(inst, "fake-sdk-garbage", () => "7.0.0"),
      ).not.toThrow();
    }

    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });

  it("warns without a version when the version getter throws", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = fakeInstrumentor({
      name: "fake-sdk-throwing-getter",
      supportedVersions: [">=4 <7"],
    });

    expect(() =>
      warnIfInstrumentorSkippedModule(inst, "fake-sdk-throwing-getter", () => {
        throw new Error("package.json is not exported");
      }),
    ).not.toThrow();

    expect(warnSpy).toHaveBeenCalledTimes(1);
    const msg = String(warnSpy.mock.calls[0][0]);
    expect(msg).toContain("fake-sdk-throwing-getter is disabled");
    expect(msg).toContain(">=4 <7");
    warnSpy.mockRestore();
  });

  it("warns without ranges when supportedVersions is missing or malformed", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    warnIfInstrumentorSkippedModule(
      fakeInstrumentor({ name: "fake-sdk-noranges" }),
      "fake-sdk-noranges",
      () => "7.0.0",
    );
    warnIfInstrumentorSkippedModule(
      fakeInstrumentor({ name: "fake-sdk-badranges", supportedVersions: "nope" }),
      "fake-sdk-badranges",
      () => "7.0.0",
    );

    expect(warnSpy).toHaveBeenCalledTimes(2);
    for (const call of warnSpy.mock.calls) {
      expect(String(call[0])).not.toContain("its bundled instrumentor supports");
    }
    warnSpy.mockRestore();
  });

  it("tolerates a non-string version from the getter", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = fakeInstrumentor({
      name: "fake-sdk-badversion",
      supportedVersions: [">=4 <7"],
    });

    expect(() =>
      warnIfInstrumentorSkippedModule(
        inst,
        "fake-sdk-badversion",
        () => ({ nope: true }) as any,
      ),
    ).not.toThrow();

    expect(warnSpy).toHaveBeenCalledTimes(1);
    expect(String(warnSpy.mock.calls[0][0])).toContain(
      "fake-sdk-badversion is disabled",
    );
    warnSpy.mockRestore();
  });
});

describe("warnIfInstrumentorSkippedModule — real OpenAIInstrumentation (B5 scenario)", () => {
  // Gated: the instrumentor is an optionalDependency and may be absent.
  let OpenAIInstrumentation: any;
  try {
    OpenAIInstrumentation = requireCjs(
      "@traceloop/instrumentation-openai",
    ).OpenAIInstrumentation;
  } catch {
    OpenAIInstrumentation = undefined;
  }

  const maybe = OpenAIInstrumentation ? it : it.skip;

  // Runs BEFORE the accepted-copy case below: that one would record "openai" in
  // `_versionGateAccepted`, flipping this case to the re-initialization message.
  // `enabled: false` keeps the constructor from registering real global loader
  // hooks from inside the test process.
  maybe("finds the real `openai` def and warns while moduleExports is unset", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = new OpenAIInstrumentation({ enabled: false });

    // Pins the structural type against upstream: a def named after the target
    // module, carrying supportedVersions, with moduleExports unset until the
    // loader hook accepts a required copy.
    const def = (inst as any)._modules.find((d: any) => d?.name === "openai");
    expect(def).toBeDefined();
    expect(Array.isArray(def.supportedVersions)).toBe(true);
    expect(def.moduleExports).toBeUndefined();

    warnIfInstrumentorSkippedModule(inst, "openai", () => "7.3.0");

    expect(warnSpy).toHaveBeenCalledTimes(1);
    const msg = String(warnSpy.mock.calls[0][0]);
    expect(msg).toContain("token capture for openai is disabled");
    expect(msg).toContain("the installed version 7.3.0");
    expect(msg).toContain(def.supportedVersions[0]);
    warnSpy.mockRestore();
  });

  maybe("stays silent once the gate has accepted a copy", () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
    const inst = new OpenAIInstrumentation({ enabled: false });
    const def = (inst as any)._modules.find((d: any) => d?.name === "openai");
    // What `_onRequire` does after `isSupported(...)` passes.
    def.moduleExports = { OpenAI: class {} };

    warnIfInstrumentorSkippedModule(inst, "openai", () => "6.9.0");

    expect(warnSpy).not.toHaveBeenCalled();
    warnSpy.mockRestore();
  });
});

describe("resolveInstalledVersion", () => {
  // Gated: `openai` is a devDependency and may be absent.
  let openAIInstalled: boolean;
  try {
    requireCjs.resolve("openai");
    openAIInstalled = true;
  } catch {
    openAIInstalled = false;
  }
  const maybe = openAIInstalled ? it : it.skip;

  // `openai` omits "./package.json" from its exports map, so the direct require
  // fails with ERR_PACKAGE_PATH_NOT_EXPORTED — this pins the directory-walk
  // fallback against that real-world packaging.
  maybe("reads openai's version despite ERR_PACKAGE_PATH_NOT_EXPORTED", () => {
    expect(() => requireCjs("openai/package.json")).toThrow();
    expect(resolveInstalledVersion(requireCjs, "openai")).toMatch(/^\d+\./);
  });

  it("returns undefined for a package that isn't installed", () => {
    expect(
      resolveInstalledVersion(requireCjs, "definitely-not-installed-xyz"),
    ).toBeUndefined();
  });

  it("returns undefined when both require and require.resolve throw", () => {
    const hostile: any = Object.assign(
      () => {
        throw new Error("require exploded");
      },
      {
        resolve: () => {
          throw new Error("resolve exploded");
        },
      },
    );

    let result: string | undefined;
    expect(() => {
      result = resolveInstalledVersion(hostile, "openai");
    }).not.toThrow();
    expect(result).toBeUndefined();
  });

  it("returns undefined for a require with no resolve at all", () => {
    const bare: any = () => {
      throw new Error("nope");
    };
    expect(() => resolveInstalledVersion(bare, "openai")).not.toThrow();
    expect(resolveInstalledVersion(bare, "openai")).toBeUndefined();
  });

  it("skips stub manifests whose name doesn't match and keeps walking", () => {
    // Mirrors a package that ships `dist/package.json` = {"type":"commonjs"}:
    // the entry file's own directory has a manifest, but not the right one.
    const fake: any = Object.assign(
      (id: string) => {
        if (id === "pkg/package.json") throw new Error("not exported");
        if (id.endsWith("/dist/package.json")) return { type: "commonjs" };
        if (id.endsWith("/pkg/package.json")) return { name: "pkg", version: "9.9.9" };
        throw new Error("ENOENT");
      },
      { resolve: () => "/tmp/node_modules/pkg/dist/index.js" },
    );

    expect(resolveInstalledVersion(fake, "pkg")).toBe("9.9.9");
  });

  it("returns undefined when the matching manifest has no usable version", () => {
    const fake: any = Object.assign(
      (id: string) => {
        if (id === "pkg/package.json") throw new Error("not exported");
        if (id.endsWith("/pkg/package.json")) return { name: "pkg", version: "" };
        throw new Error("ENOENT");
      },
      { resolve: () => "/tmp/node_modules/pkg/index.js" },
    );

    expect(resolveInstalledVersion(fake, "pkg")).toBeUndefined();
  });
});
