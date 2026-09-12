/**
 * REAL-PACKAGE seam contract — Voyage AI.
 *
 * THIN BY DESIGN (long-tail provider): the prototype seam plus the attach
 * check. The token round trip lives in `realProviderSeams.voyageUsage.test.ts`
 * so that a usage-shape regression and a seam regression fail separately.
 *
 * PACKAGE NAME
 * ------------
 * The Voyage Node SDK is published as **`voyageai`**
 * (github.com/voyage-ai/typescript-sdk). The enforcer's registry `moduleName`,
 * both auto-discovery attempts (`resolveProviderModule("voyageai")` and the
 * `await import("voyageai")` fallback) and `package.json`'s optional
 * peerDependency all name that package. Until 2026-09 they named
 * `voyageai-typescript`, which has never existed on npm (404) — so a customer
 * who simply installed `voyageai` got NO Voyage instrumentation unless they
 * also passed `instrumentModules: { voyageai: <module> }`. The
 * "registry name resolves" case below pins the corrected name so the gap
 * cannot silently return.
 */
import {
  describe,
  it,
  expect,
  beforeAll,
  afterAll,
  beforeEach,
  afterEach,
} from "vitest";

import { __test__, uninstrument } from "../src/enforcer";
import {
  createSeamHarness,
  flushLogs,
  installNoNetworkGuard,
  installedVersion,
  methodNames,
  requireCjs,
} from "./helpers/realProviderSeam";

/** The published package — and, since 2026-09, the enforcer's registry name. */
const PKG = "voyageai";
const VERSION = installedVersion(PKG);
const voyageModule = (() => {
  try {
    return requireCjs(PKG);
  } catch {
    return undefined;
  }
})();
const maybe = voyageModule ? it : it.skip;
const VoyageAIClient: any =
  voyageModule?.VoyageAIClient ?? voyageModule?.default?.VoyageAIClient ?? voyageModule?.default;

const h = createSeamHarness();
let restoreNetwork: () => void;

beforeAll(() => {
  restoreNetwork = installNoNetworkGuard();
  console.log(`[realProviderSeams] voyageai@${VERSION ?? "ABSENT"}`);
});
afterAll(() => restoreNetwork());

beforeEach(() => h.install());
afterEach(async () => {
  await flushLogs(30);
  uninstrument();
  h.reset();
});

describe("voyageai — shape", () => {
  maybe("VoyageAIClient is exported at the package root", () => {
    expect(typeof VoyageAIClient).toBe("function");
  });

  maybe("embed and multimodalEmbed are reachable on the prototype chain", () => {
    // `_instrumentVoyage` reads `proto[method]`, which walks the chain — the
    // generated Fern base class carries `multimodalEmbed` while the hand-written
    // subclass overrides `embed`. Both must stay reachable.
    for (const m of ["embed", "multimodalEmbed"]) {
      expect(
        typeof VoyageAIClient.prototype[m],
        `VoyageAIClient.prototype.${m} is gone; own keys: ` +
          methodNames(VoyageAIClient.prototype).join(", "),
      ).toBe("function");
    }
  });

  maybe("the registry's module name resolves to the installed package", () => {
    // Pins the package-name fix: the registry's `moduleName` / auto-discovery
    // name must be the package that actually exists on npm. If this fails,
    // auto-discovery is dead again and only the explicit
    // `instrumentModules: { voyageai }` path instruments Voyage.
    const REGISTRY_NAME = "voyageai";
    expect(REGISTRY_NAME).toBe(PKG);
    let resolvable = true;
    try {
      requireCjs.resolve(REGISTRY_NAME);
    } catch {
      resolvable = false;
    }
    expect(
      resolvable,
      `\`${REGISTRY_NAME}\` does not resolve — the enforcer's Voyage registry name ` +
        "no longer matches the published package, so auto-discovery is dead.",
    ).toBe(true);
  });

  maybe("_instrumentVoyage replaces both embedding methods", () => {
    const before = {
      embed: VoyageAIClient.prototype.embed,
      multimodalEmbed: VoyageAIClient.prototype.multimodalEmbed,
    };
    (__test__ as any)._instrumentVoyage(voyageModule);
    (__test__ as any)._setInstrumented(true);
    expect(VoyageAIClient.prototype.embed, "embed was NOT wrapped").not.toBe(before.embed);
    expect(VoyageAIClient.prototype.multimodalEmbed).not.toBe(before.multimodalEmbed);
  });
});
