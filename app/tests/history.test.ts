import { describe, expect, it } from "vitest";
import { historyRecordFor, shownCounts, type HistoryRecord } from "../src/lib/history";
import { decide, type BeanCall } from "../src/lib/rules";

const calls = (o: Partial<Record<BeanCall, number>>): BeanCall[] =>
  (Object.entries(o) as [BeanCall, number][]).flatMap(([k, n]) => Array<BeanCall>(n).fill(k));

describe("history privacy (REGRESSION_A I-1)", () => {
  it("a not-sure result (implausible) keeps only date, total and band — no per-class counts, no range", () => {
    // cbd_aaa_2 as the app counts it: 39 broken, 8 unhulled, 3 unsure of 50 -> implausible
    const o = decide(calls({ broken: 39, unhulled: 8, unsure: 3 }));
    expect(o.band).toBe("unsure");
    expect(o.unsureReason).toBe("implausible");
    const rec = historyRecordFor(o, 1, "real");
    expect(rec).toEqual({ ts: 1, total: 50, band: "unsure", handfuls: 1, demo: "real" });
    expect(JSON.stringify(rec)).not.toMatch(/broken|unhulled|counts|"lo"|"hi"/);
    expect(shownCounts(rec)).toBeNull();
  });
  it("a not-sure result (too many unsure) is stored the same way", () => {
    const o = decide(calls({ good: 70, broken: 5, unsure: 25 }));
    expect(o.unsureReason).toBe("too_many_unsure");
    const rec = historyRecordFor(o, 2);
    expect(rec.counts).toBeUndefined();
    expect(rec.lo).toBeUndefined();
    expect(shownCounts(rec)).toBeNull();
  });
  it("records saved by older builds with counts are still shown WITHOUT counts when not sure", () => {
    const old: HistoryRecord = { ts: 3, total: 50, band: "unsure", lo: 0.8, hi: 0.95,
      counts: { good: 0, dark: 0, insect: 0, broken: 39, unhulled: 8, other_defect: 0, unsure: 3 } };
    expect(shownCounts(old)).toBeNull();
  });
  it("a banded result keeps its counts and range", () => {
    const o = decide(calls({ good: 88, dark: 4, insect: 3, broken: 2, unhulled: 2, unsure: 1 }));
    expect(o.band).toBe("some");
    const rec = historyRecordFor(o, 4, "synthetic");
    expect(rec.counts).toEqual(o.counts);
    expect(rec.lo).toBeCloseTo(o.interval.lo);
    expect(shownCounts(rec)).toEqual(o.counts);
  });
});
