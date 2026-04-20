import cv2
import os
from pathlib import Path

folder = Path("/Users/ryanrakestraw/Library/Application Support/Cryptomator/mnt/cyptomator_vault/local_vids/juju-b/")
files = sorted([f for f in folder.iterdir() if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')])[:10]

for f in files:
    img = cv2.imread(str(f))
    if img is not None:
        h, w = img.shape[:2]
        print(f"{f.name}: {w}x{h}")
    else:
        print(f"{f.name}: Failed to load")
