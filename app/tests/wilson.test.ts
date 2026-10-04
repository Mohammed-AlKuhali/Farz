import { describe, expect, it } from "vitest";
import { bandOf, bandOfRate, crossesBandEdge, wilson } from "../src/lib/wilson";

describe("wilson 95% interval", () => {
  it("matches the PLAN.md worked example: 12/100 -> 7.0%-19.8%", () => {
    const iv = wilson(12, 100);
    expect(iv.lo).toBeCloseTo(0.07, 3);
    expect(iv.hi).toBeCloseTo(0.198, 3);
    expect(bandOf(iv)).toBe("some");
  });
  it("matches an independent reference value (k=5, n=50)", () => {
    // scipy.stats.binomtest(5, 50).proportion_ci(method='wilson') = (0.043476, 0.213602), run in .venv 3 Oct 2026
    const iv = wilson(5, 50);
    expect(iv.lo).toBeCloseTo(0.04347, 4);
    expect(iv.hi).toBeCloseTo(0.2136, 4);
  });
  it("k=0 has lo=0 and is clean only with enough beans", () => {
    expect(wilson(0, 100).lo).toBe(0);
    expect(bandOf(wilson(0, 100))).toBe("clean"); // hi = 3.7%
    expect(bandOf(wilson(0, 60))).toBe("unsure"); // hi = 6.0% crosses 5%
  });
  it("k=n is many", () => {
    expect(bandOf(wilson(100, 100))).toBe("many");
    expect(wilson(100, 100).hi).toBeCloseTo(1, 12);
  });
  it("a band is given only if the whole interval is inside it", () => {
    expect(bandOf({ lo: 0.01, hi: 0.049 })).toBe("clean");
    expect(bandOf({ lo: 0.01, hi: 0.05 })).toBe("unsure");
    expect(bandOf({ lo: 0.05, hi: 0.2 })).toBe("some");
    expect(bandOf({ lo: 0.049, hi: 0.15 })).toBe("unsure");
    expect(bandOf({ lo: 0.15, hi: 0.21 })).toBe("unsure");
    expect(bandOf({ lo: 0.2, hi: 0.4 })).toBe("unsure");
    expect(bandOf({ lo: 0.2001, hi: 0.4 })).toBe("many");
  });
  it("invalid input never produces a band", () => {
    expect(bandOf(wilson(0, 0))).toBe("unsure");
    expect(bandOf(wilson(5, 3))).toBe("unsure");
  });
  it("interval always contains p and stays in [0,1]", () => {
    for (let n = 1; n <= 160; n += 7)
      for (let k = 0; k <= n; k++) {
        const iv = wilson(k, n);
        expect(iv.lo).toBeGreaterThanOrEqual(0);
        expect(iv.hi).toBeLessThanOrEqual(1);
        expect(iv.lo).toBeLessThanOrEqual(k / n + 1e-12);
        expect(iv.hi).toBeGreaterThanOrEqual(k / n - 1e-12);
      }
  });
});

describe("point-estimate band and the 'about' flag", () => {
  it("bandOfRate uses clean [0,5%), some [5%,20%], many (20%,100%]", () => {
    expect(bandOfRate(0)).toBe("clean");
    expect(bandOfRate(0.0499)).toBe("clean");
    expect(bandOfRate(0.05)).toBe("some");
    expect(bandOfRate(0.2)).toBe("some");
    expect(bandOfRate(0.2001)).toBe("many");
    expect(bandOfRate(NaN)).toBe("unsure");
  });
  it("crossesBandEdge is false only when the whole interval sits in the point's band", () => {
    expect(crossesBandEdge(wilson(12, 100))).toBe(false);
    expect(crossesBandEdge(wilson(0, 100))).toBe(false);
    expect(crossesBandEdge(wilson(3, 100))).toBe(true);
    expect(crossesBandEdge(wilson(25, 97))).toBe(true);
    expect(crossesBandEdge(wilson(40, 100))).toBe(false);
  });
});
