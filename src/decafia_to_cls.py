"""Convert DECAFIA/CoffeeLeaf-CO (YOLO detection) into clean single-label leaf images.
rust = roya only; miner = minador only; healthy = leaf outline with no disease boxes.
Mixed or weevil-only images are skipped rather than guessed. Crops to the leaf box when there is exactly one."""
from pathlib import Path
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "data/raw/decafia/CoffeeLeaf-CO"
out = ROOT / "data/raw/decafia_cls"
NAMES = {0: "roya", 1: "coco", 2: "hojas", 3: "minador"}
stats = {}
for img in base.glob("images/*/*.jpg"):
    lab = base / "labels" / img.parent.name / (img.stem + ".txt")
    boxes = []
    if lab.exists():
        for line in lab.read_text().split("\n"):
            p = line.split()
            if len(p) == 5:
                boxes.append((int(p[0]), *map(float, p[1:])))
    cls = {NAMES[b[0]] for b in boxes}
    disease = cls - {"hojas"}
    if disease == {"roya"}: y = "rust"
    elif disease == {"minador"}: y = "miner"
    elif not disease: y = "healthy"
    else: y = None
    stats[y] = stats.get(y, 0) + 1
    if y is None: continue
    im = Image.open(img).convert("RGB")
    leaves = [b for b in boxes if b[0] == 2]
    if len(leaves) == 1:
        _, xc, yc, w, h = leaves[0]; W, H = im.size; m = 0.05
        im = im.crop((max(0, (xc - w/2 - m) * W), max(0, (yc - h/2 - m) * H), min(W, (xc + w/2 + m) * W), min(H, (yc + h/2 + m) * H)))
    d = out / y; d.mkdir(parents=True, exist_ok=True)
    im.save(d / f"{img.parent.name}_{img.stem}.jpg", quality=92)
print({str(k): v for k, v in stats.items()})
