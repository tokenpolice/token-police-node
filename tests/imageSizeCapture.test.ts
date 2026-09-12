/**
 * Image_size resolution cascade (request → response → binary → default).
 *
 * Mapper turns parseable WxH into image_output_pixels. Extractors must prefer
 * request dims, then response/binary, then conservative provider defaults.
 * Never throw (GOLDEN RULE). Never invent for xAI / ai_sdk without measured dims.
 */
import { describe, it, expect } from "vitest";
import { deflateSync } from "node:zlib";
import { __test__ } from "../src/enforcer";

const {
  MODALITY_HANDLERS,
  _imageSizeFromDims,
  _parseImageSizeStr,
  _imageDimsFromBinary,
  _resolveImageSize,
} = __test__ as any;

function makePng(w: number, h: number): Buffer {
  const crcTable: number[] = [];
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    crcTable[n] = c >>> 0;
  }
  const crc = (buf: Buffer) => {
    let c = 0xffffffff;
    for (let i = 0; i < buf.length; i++) c = crcTable[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
    const out = Buffer.alloc(4);
    out.writeUInt32BE((c ^ 0xffffffff) >>> 0, 0);
    return out;
  };
  const chunk = (tag: string, data: Buffer) => {
    const len = Buffer.alloc(4);
    len.writeUInt32BE(data.length, 0);
    const type = Buffer.from(tag, "ascii");
    return Buffer.concat([len, type, data, crc(Buffer.concat([type, data]))]);
  };
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // RGB
  const rows: Buffer[] = [];
  for (let y = 0; y < h; y++) {
    const row = Buffer.alloc(1 + w * 3);
    row[0] = 0;
    for (let x = 0; x < w; x++) {
      row[1 + x * 3] = 255;
      row[2 + x * 3] = 0;
      row[3 + x * 3] = 0;
    }
    rows.push(row);
  }
  const idat = deflateSync(Buffer.concat(rows));
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", idat),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

const PNG_2X3 = makePng(2, 3);
const PNG_2X3_B64 = PNG_2X3.toString("base64");
const PNG_512X256 = makePng(512, 256);
const PNG_512X256_B64 = PNG_512X256.toString("base64");

describe("Helpers", () => {
  it("imageSizeFromDims", () => {
    expect(_imageSizeFromDims(512, 768)).toBe("512x768");
    expect(_imageSizeFromDims(null, 768)).toBe("");
    expect(_imageSizeFromDims(512, 0)).toBe("");
  });

  it("parseImageSizeStr", () => {
    expect(_parseImageSizeStr("1024x1024")).toBe("1024x1024");
    expect(_parseImageSizeStr("512×768")).toBe("512x768");
    expect(_parseImageSizeStr("auto")).toBe("");
    expect(_parseImageSizeStr("16:9")).toBe("");
  });

  it("binary PNG header", () => {
    expect(_imageDimsFromBinary(PNG_2X3)).toBe("2x3");
    expect(_imageDimsFromBinary(PNG_512X256)).toBe("512x256");
    expect(_imageDimsFromBinary(Buffer.from("nope"))).toBe("");
    expect(_imageDimsFromBinary(null)).toBe("");
  });

  it("resolve: request wins over binary", () => {
    expect(
      _resolveImageSize({
        requestSize: "1024x1792",
        result: { data: [{ b64_json: PNG_2X3_B64 }] },
        provider: "openai",
        model: "dall-e-3",
      }),
    ).toBe("1024x1792");
  });

  it("resolve: binary before default", () => {
    expect(
      _resolveImageSize({
        requestSize: "auto",
        result: { data: [{ b64_json: PNG_512X256_B64 }] },
        provider: "openai",
        model: "gpt-image-1",
      }),
    ).toBe("512x256");
  });

  it("resolve: together default", () => {
    expect(_resolveImageSize({ provider: "together", model: "flux" })).toBe("1024x1024");
  });

  it("resolve: xai no default", () => {
    expect(
      _resolveImageSize({ provider: "xai", model: "grok", allowDefault: false }),
    ).toBe("");
  });
});

describe("Image_size capture (Together / HF / Google / OpenAI)", () => {
  it("together: captures width×height when both present", () => {
    const h = MODALITY_HANDLERS["together:image_gen"];
    const args = [{ model: "black-forest-labs/FLUX.1-schnell", prompt: "cat", n: 1, width: 512, height: 768 }];
    const ex = h.extract(args, { data: [{}] });
    expect(ex.items.images_generated).toBe(1);
    expect(ex.items.image_size).toBe("512x768");
  });

  it("together: default 1024x1024 when width/height omitted", () => {
    const h = MODALITY_HANDLERS["together:image_gen"];
    const ex = h.extract([{ model: "m", prompt: "x", n: 1 }], { data: [{}] });
    expect(ex.items.image_size).toBe("1024x1024");
  });

  it("together: binary overrides default", () => {
    const h = MODALITY_HANDLERS["together:image_gen"];
    const ex = h.extract([{ model: "m", n: 1 }], { data: [{ b64_json: PNG_2X3_B64 }] });
    expect(ex.items.image_size).toBe("2x3");
  });

  it("huggingface: captures parameters.width/height", () => {
    const h = MODALITY_HANDLERS["huggingface:image_gen"];
    const args = [
      {
        model: "black-forest-labs/FLUX.1-dev",
        inputs: "a cat",
        parameters: { width: 640, height: 480 },
      },
    ];
    const ex = h.extract(args, {});
    expect(ex.items.images_generated).toBe(1);
    expect(ex.items.image_size).toBe("640x480");
  });

  it("huggingface: also accepts top-level width/height", () => {
    const h = MODALITY_HANDLERS["huggingface:image_gen"];
    const ex = h.extract([{ model: "m", width: 256, height: 256 }], {});
    expect(ex.items.image_size).toBe("256x256");
  });

  it("huggingface: default when no dims", () => {
    const h = MODALITY_HANDLERS["huggingface:image_gen"];
    const ex = h.extract([{ model: "m", inputs: "a cat" }], {});
    expect(ex.items.image_size).toBe("1024x1024");
  });

  it("huggingface: Buffer result yields binary dims", () => {
    const h = MODALITY_HANDLERS["huggingface:image_gen"];
    const ex = h.extract([{ model: "m" }], PNG_512X256);
    expect(ex.items.image_size).toBe("512x256");
  });

  it("google: captures config width×height", () => {
    const h = MODALITY_HANDLERS["google:image_gen"];
    const args = [
      {
        model: "imagen-3.0-generate-002",
        config: { numberOfImages: 1, width: 1024, height: 1024 },
      },
    ];
    const ex = h.extract(args, { generatedImages: [{}] });
    expect(ex.items.images_generated).toBe(1);
    expect(ex.items.image_size).toBe("1024x1024");
  });

  it("google: default when only count", () => {
    const h = MODALITY_HANDLERS["google:image_gen"];
    const ex = h.extract(
      [{ model: "imagen-3.0-generate-002", config: { numberOfImages: 2 } }],
      { generatedImages: [{}, {}] },
    );
    expect(ex.items.images_generated).toBe(2);
    expect(ex.items.image_size).toBe("1024x1024");
  });

  it("google: aspect-only does not invent", () => {
    const h = MODALITY_HANDLERS["google:image_gen"];
    const ex = h.extract(
      [{ model: "imagen-3.0-generate-002", config: { numberOfImages: 1, aspectRatio: "16:9" } }],
      { generatedImages: [{}] },
    );
    expect(ex.items.image_size).toBe("");
  });

  it("openai: request size", () => {
    const h = MODALITY_HANDLERS["openai:image_gen"];
    const ex = h.extract(
      [{ model: "dall-e-3", size: "1792x1024", n: 1 }],
      { data: [{}] },
    );
    expect(ex.items.image_size).toBe("1792x1024");
  });

  it("openai: b64 when size=auto", () => {
    const h = MODALITY_HANDLERS["openai:image_gen"];
    const ex = h.extract(
      [{ model: "gpt-image-1", size: "auto", n: 1 }],
      { data: [{ b64_json: PNG_512X256_B64 }] },
    );
    expect(ex.items.image_size).toBe("512x256");
  });

  it("openai: default when url-only and size omitted", () => {
    const h = MODALITY_HANDLERS["openai:image_gen"];
    const ex = h.extract(
      [{ model: "dall-e-3", n: 1 }],
      { data: [{ url: "https://example.com/x.png" }] },
    );
    expect(ex.items.image_size).toBe("1024x1024");
  });

  it("ai_sdk: parseable size preserved; aspect alone stays empty", () => {
    const h = MODALITY_HANDLERS["ai_sdk:image_gen"];
    expect(
      h.extract([{ n: 1, size: "1024x1024", __tpXaiModel: "x" }], { images: [] }).items
        .image_size,
    ).toBe("1024x1024");
    expect(
      h.extract([{ n: 1, aspectRatio: "16:9", __tpXaiModel: "x" }], { images: [] }).items
        .image_size,
    ).toBe("");
  });

  // GOLDEN RULE: extract never throws on garbage input.
  it("handlers fail-open on null/malformed body", () => {
    for (const key of [
      "together:image_gen",
      "huggingface:image_gen",
      "google:image_gen",
      "openai:image_gen",
      "ai_sdk:image_gen",
    ]) {
      const h = MODALITY_HANDLERS[key];
      expect(() => h.extract([null as any], null)).not.toThrow();
      expect(() => h.extract([], undefined)).not.toThrow();
      expect(() => h.extract([{ parameters: "x" }], { data: "bad" })).not.toThrow();
    }
    expect(() => _imageDimsFromBinary(undefined)).not.toThrow();
    expect(() => _resolveImageSize({ result: { data: [{ b64_json: "!!!" }] } })).not.toThrow();
  });
});
