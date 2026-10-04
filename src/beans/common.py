"""Shared constants for the Farz per-bean classifier (no training code here)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["good", "dark", "insect", "broken", "unhulled"]
# J4ckDev photo stem -> Farz class (5-class map agreed in the task brief)
PHOTO_CLASS = {
    "Normales": "good",
    "Negros": "dark", "MarronAVinagre": "dark", "DXHongo": "dark",
    "BrocadoLeve": "insect", "BrocadoSevero": "insect",
    "PMordidoCortado": "broken", "Inmaduro": "broken", "Concha": "broken",
    "Pergamino": "unhulled", "CerezaSeca": "unhulled",
}
J4CK_DIR = ROOT / "data/raw/green/ImageDataset"
CBD_DIR = ROOT / "data/raw/cbd/CBD_Coffee Bean Dataset/CBD_Coffee Bean Dataset"
CROPS_DIR = ROOT / "data/crops"
CROPS_CSV = CROPS_DIR / "crops.csv"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats used by torchvision weights
