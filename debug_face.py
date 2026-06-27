from pathlib import Path

import cv2

img_path = Path(
    "/Users/ryanrakestraw/Library/Application Support/Cryptomator/mnt/"
    "cyptomator_vault/local_vids/juju-b/20230828_112434.jpg"
)
img = cv2.imread(str(img_path))
if img is None:
    print("Could not load image")
    exit()

h, w = img.shape[:2]
scale_factor = 1.0
max_dim = 4000
img_for_detection = img
if max(h, w) > max_dim:
    scale_factor = max_dim / max(h, w)
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    img_for_detection = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)
gray = cv2.cvtColor(img_for_detection, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))

print(f"Found {len(faces)} faces")
for x, y, w_f, h_f in faces:
    print(f"  Face at {x}, {y}, size {w_f}x{h_f}")
