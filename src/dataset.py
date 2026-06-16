from pathlib import Path
from typing import List, Optional, Tuple
 
import numpy as np
from PIL import Image
from torch.utils.data import  Dataset

 
LABEL_MAP = {
    "cane":       "dog",
    "cavallo":    "horse",
    "elefante":   "elephant",
    "farfalla":   "butterfly",
    "gallina":    "chicken",
    "gatto":      "cat",
    "mucca":      "cow",
    "pecora":     "sheep",
    "ragno":      "spider",
    "scoiattolo": "squirrel",
}
 
CLASSES = sorted(LABEL_MAP.values())
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)
IMG_SIZE = 128
 
class Animals10Dataset(Dataset):
    """
    Dataset klasa za Animals-10.
 
    Args:
        root_dir      : putanja do 'raw-img' foldera
        transform     : torchvision transform pipeline
        max_per_class : maksimalan broj slika po klasi (None = sve)
    """
 
    def __init__(
        self,
        root_dir: str,
        transform=None,
        max_per_class: Optional[int] = None,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
 
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
 
        for it_name, en_name in LABEL_MAP.items():
            class_dir = self.root_dir / it_name
            if not class_dir.exists():
                print(f"  [UPOZORENJE] Folder nije pronadjen: {class_dir}")
                continue
            label = CLASS_TO_IDX[en_name]
            imgs = sorted([
                p for p in class_dir.iterdir()
                if p.suffix.lower() in valid_ext
            ])
            if max_per_class:
                imgs = imgs[:max_per_class]
            for img_path in imgs:
                self.samples.append((img_path, label))
 
        self.classes = CLASSES
        self.class_to_idx = CLASS_TO_IDX
        print(f"Dataset ucitan: {len(self.samples)} slika, {NUM_CLASSES} klasa")
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        if self.transform:
            img = self.transform(img)
        return img, label
 
    def get_labels(self) -> np.ndarray:
        return np.array([s[1] for s in self.samples])
 
    def class_counts(self) -> dict:
        from collections import Counter
        counts = Counter(s[1] for s in self.samples)
        return {CLASSES[k]: v for k, v in sorted(counts.items())}
 
 
