/**
 * Pillow-compatible BILINEAR resize for 8-bit RGB images.
 *
 * Re-implements Pillow's libImaging/Resample.c (precompute_coeffs, normalize_coeffs_8bpc,
 * ImagingResampleHorizontal_8bpc / Vertical_8bpc): a separable, antialiased triangle filter whose
 * support is scaled by the downscale factor, fixed-point coefficients (PRECISION_BITS = 22),
 * horizontal pass first into an 8-bit intermediate, then the vertical pass. The goal is bit-exact
 * agreement with `PIL.Image.resize(size, Image.BILINEAR)` so the Python bean finder and this port
 * see the same pixels (verified by tests/beanfinder.parity.test.ts).
 *
 * Images are interleaved RGB (3 bytes per pixel) Uint8Arrays.
 */

export interface RGBImage {
  width: number;
  height: number;
  /** interleaved RGB, length = width * height * 3 */
  data: Uint8Array;
}

const PRECISION_BITS = 32 - 8 - 2; // 22, as in Pillow
const HALF = 1 << (PRECISION_BITS - 1);
const ONE = 1 << PRECISION_BITS;

function bilinearFilter(x: number): number {
  if (x < 0.0) x = -x;
  if (x < 1.0) return 1.0 - x;
  return 0.0;
}

interface Coeffs {
  ksize: number;
  bounds: Int32Array; // [xmin, xmax(count)] per output pixel
  kk: Int32Array; // fixed-point coefficients, outSize * ksize
}

function precomputeCoeffs(inSize: number, in0: number, in1: number, outSize: number): Coeffs {
  const support0 = 1.0; // bilinear
  const scale = (in1 - in0) / outSize;
  let filterscale = scale;
  if (filterscale < 1.0) filterscale = 1.0;
  const support = support0 * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const pre = new Float64Array(outSize * ksize);
  const bounds = new Int32Array(outSize * 2);
  for (let xx = 0; xx < outSize; xx++) {
    const center = in0 + (xx + 0.5) * scale;
    let ww = 0.0;
    const ss = 1.0 / filterscale;
    // (int) cast in C truncates toward zero
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    const base = xx * ksize;
    for (let x = 0; x < xmax; x++) {
      const w = bilinearFilter((x + xmin - center + 0.5) * ss);
      pre[base + x] = w;
      ww += w;
    }
    for (let x = 0; x < xmax; x++) {
      if (ww !== 0.0) pre[base + x] /= ww;
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  // normalize_coeffs_8bpc
  const kk = new Int32Array(outSize * ksize);
  for (let i = 0; i < pre.length; i++) {
    const v = pre[i];
    kk[i] = v < 0 ? Math.trunc(-0.5 + v * ONE) : Math.trunc(0.5 + v * ONE);
  }
  return { ksize, bounds, kk };
}

function clip8(ss: number): number {
  // Pillow: clip8_lookups[ss >> PRECISION_BITS]; arithmetic shift == floor division
  const v = Math.floor(ss / ONE);
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

/** Bit-exact port of PIL Image.resize((outW, outH), Image.BILINEAR) for RGB images. */
export function pillowResizeBilinear(img: RGBImage, outW: number, outH: number): RGBImage {
  const inW = img.width;
  const inH = img.height;
  if (outW === inW && outH === inH) {
    return { width: inW, height: inH, data: img.data.slice() };
  }
  const needH = outW !== inW;
  const needV = outH !== inH;
  const hc = precomputeCoeffs(inW, 0, inW, outW);
  const vc = precomputeCoeffs(inH, 0, inH, outH);

  let src = img.data;
  let srcW = inW;
  let rowOffset = 0;
  const vBounds = vc.bounds.slice();

  if (needH) {
    const yFirst = vc.bounds[0];
    const yLast = vc.bounds[outH * 2 - 2] + vc.bounds[outH * 2 - 1];
    for (let i = 0; i < outH; i++) vBounds[i * 2] -= yFirst;
    const tmpH = yLast - yFirst;
    const tmp = new Uint8Array(outW * tmpH * 3);
    const { ksize, bounds, kk } = hc;
    for (let yy = 0; yy < tmpH; yy++) {
      const rowIn = (yy + yFirst) * inW * 3;
      const rowOut = yy * outW * 3;
      for (let xx = 0; xx < outW; xx++) {
        const xmin = bounds[xx * 2];
        const xmax = bounds[xx * 2 + 1];
        const kb = xx * ksize;
        let s0 = HALF, s1 = HALF, s2 = HALF;
        let p = rowIn + xmin * 3;
        for (let x = 0; x < xmax; x++) {
          const k = kk[kb + x];
          s0 += src[p] * k;
          s1 += src[p + 1] * k;
          s2 += src[p + 2] * k;
          p += 3;
        }
        const o = rowOut + xx * 3;
        tmp[o] = clip8(s0);
        tmp[o + 1] = clip8(s1);
        tmp[o + 2] = clip8(s2);
      }
    }
    src = tmp;
    srcW = outW;
    rowOffset = 0;
    if (!needV) {
      // Pillow only skips the vertical pass when heights match and box is trivial; tmp is the answer.
      return { width: outW, height: tmpH, data: tmp };
    }
  }

  // vertical pass
  const outData = new Uint8Array(outW * outH * 3);
  const bnds = needH ? vBounds : vc.bounds;
  const { ksize, kk } = vc;
  for (let yy = 0; yy < outH; yy++) {
    const kb = yy * ksize;
    const ymin = bnds[yy * 2];
    const ymax = bnds[yy * 2 + 1];
    const rowOut = yy * outW * 3;
    for (let xx = 0; xx < outW; xx++) {
      let s0 = HALF, s1 = HALF, s2 = HALF;
      for (let y = 0; y < ymax; y++) {
        const k = kk[kb + y];
        const p = ((y + ymin + rowOffset) * srcW + xx) * 3;
        s0 += src[p] * k;
        s1 += src[p + 1] * k;
        s2 += src[p + 2] * k;
      }
      const o = rowOut + xx * 3;
      outData[o] = clip8(s0);
      outData[o + 1] = clip8(s1);
      outData[o + 2] = clip8(s2);
    }
  }
  return { width: outW, height: outH, data: outData };
}
