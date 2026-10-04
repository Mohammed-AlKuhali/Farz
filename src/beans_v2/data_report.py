"""Print the markdown count tables for docs/DATA_SOURCES_V2.md from data/crops_v2/summary_v2.json and
the per-source build logs (so every number in the doc is reproducible).

  python src/beans_v2/data_report.py > /tmp/tables.md
"""
import csv, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from data_common import ROOT, FARZ_CLASSES
from data_build_crops import SOURCE_ORDER, OUT

S = json.load(open(os.path.join(OUT, "summary_v2.json")))
counts = S["counts_kept"]

print("| Source | " + " | ".join(FARZ_CLASSES) + " | Total kept | Distinct photos/groups | Source images in | Crops built | Dropped as duplicates |")
print("|---|" + "---|" * (len(FARZ_CLASSES) + 4))
for s in SOURCE_ORDER:
    if s not in counts: continue
    log = json.load(open(os.path.join(OUT, s, "build_log.json")))
    ndup = log["crops"] - sum(counts[s].values())
    row = [s] + [f"{counts[s][c]:,}" for c in FARZ_CLASSES] + [f"{sum(counts[s].values()):,}",
           f"{S['distinct_groups_kept'][s]:,}", f"{log['items']:,}", f"{log['crops']:,}", f"{ndup:,}"]
    print("| " + " | ".join(row) + " |")
tot = S["totals_by_class"]
print("| **Total** | " + " | ".join(f"**{tot[c]:,}**" for c in FARZ_CLASSES) + f" | **{S['total_kept']:,}** | | | | |")
print()
print("Per source class (kept crops):")
print()
rows = list(csv.DictReader(open(os.path.join(OUT, "crops_v2.csv"))))
by = {}
for r in rows:
    if int(r["keep"]):
        by.setdefault(r["source"], {}).setdefault((r["source_class"], r["farz_class"]), 0)
        by[r["source"]][(r["source_class"], r["farz_class"])] += 1
for s in SOURCE_ORDER:
    if s not in by: continue
    items = sorted(by[s].items(), key=lambda kv: (kv[0][1], -kv[1]))
    print(f"- **{s}**: " + "; ".join(f"{sc} -> {fc} {n:,}" for (sc, fc), n in items))
print()
print("Dedupe (exact):", json.dumps(S["dedupe"]))
print("Near-duplicates dropped (thumbnail r>=0.998):", json.dumps(S["near_dup_dropped"]))
print("Identical dHash across sources (bucket counts):", json.dumps(S["near_dup_identical_dhash_cross_source_buckets"]))
print("Identical dHash vs J4ckDev v1 crops:", json.dumps(S["identical_dhash_vs_j4ckdev_v1_crops"]))
print("Identical dHash within a source, different files:", json.dumps(S["identical_dhash_within_source_different_files"]))
