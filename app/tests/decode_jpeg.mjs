// Decode JPEGs with jpeg-js exactly as the vitest parity test does, and write raw RGB for Python.
// usage: node tests/decode_jpeg.mjs <out_dir> <file1.jpg> [file2.jpg ...]
// writes <out_dir>/<index>.rgb and <out_dir>/<index>.json ({width,height})
import fs from "node:fs";
import path from "node:path";
import jpeg from "jpeg-js";

export function decodeJpegRGB(buf) {
  const img = jpeg.decode(buf, { useTArray: true, formatAsRGBA: false, maxResolutionInMP: 200, maxMemoryUsageInMB: 2048 });
  return { width: img.width, height: img.height, data: img.data };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const [outDir, ...files] = process.argv.slice(2);
  fs.mkdirSync(outDir, { recursive: true });
  files.forEach((f, i) => {
    const { width, height, data } = decodeJpegRGB(fs.readFileSync(f));
    fs.writeFileSync(path.join(outDir, `${i}.rgb`), data);
    fs.writeFileSync(path.join(outDir, `${i}.json`), JSON.stringify({ width, height, file: f }));
  });
}
