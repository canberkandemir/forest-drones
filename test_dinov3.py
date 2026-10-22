
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel

PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_PATH = PROJECT_ROOT / "assets" / "sample_images" / "forest.jpeg"
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "yolo11s.pt"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "dinov3_exploration" / "dinov3_upper_body_similarity.png"

# ---------------------------------------
# 1. Load original image
# ---------------------------------------

image = Image.open(IMAGE_PATH).convert("RGB")
image_w, image_h = image.size

# ---------------------------------------
# 2. Detect the person with YOLO
# ---------------------------------------

yolo = YOLO(YOLO_MODEL_PATH)

results = yolo.predict(
    source=IMAGE_PATH,
    classes=[0],
    conf=0.25,
    imgsz=960,
    verbose=False
)

boxes = results[0].boxes

if len(boxes) == 0:
    raise RuntimeError("No person detected by YOLO.")

best_idx = int(boxes.conf.argmax())
x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()

print("YOLO confidence:", float(boxes.conf[best_idx]))

# ---------------------------------------
# 3. Select head and upper body
# ---------------------------------------

# Use the upper 55% of YOLO's person box.
# This is a heuristic, not an exact head annotation.

upper_y2 = y1 + 0.55 * (y2 - y1)

# ---------------------------------------
# 4. Load DINOv3
# ---------------------------------------

model_id = "facebook/dinov3-vits16-pretrain-lvd1689m"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

# Apply the processor to the original image
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# ---------------------------------------
# 5. Get patch features and grid size
# ---------------------------------------

features = outputs.last_hidden_state

input_h, input_w = inputs["pixel_values"].shape[-2:]
patch_size = 16

grid_h = input_h // patch_size
grid_w = input_w // patch_size
num_patches = grid_h * grid_w

patch_features = features[0, -num_patches:, :]
patch_features = F.normalize(patch_features, dim=-1)

print("DINOv3 patch grid:", grid_h, "x", grid_w)

# ---------------------------------------
# 6. Map YOLO box to processed image
# ---------------------------------------

# Hugging Face DINOv3 processor commonly resizes
# the image and then applies a center crop.

if processor.do_resize:
    resize_size = processor.size

    if "shortest_edge" in resize_size:
        target_short = resize_size["shortest_edge"]
        scale = target_short / min(image_w, image_h)
        resized_w = int(image_w * scale)
        resized_h = int(image_h * scale)
    else:
        resized_h = resize_size["height"]
        resized_w = resize_size["width"]

else:
    resized_w, resized_h = image_w, image_h

scale_x = resized_w / image_w
scale_y = resized_h / image_h

if processor.do_center_crop:
    crop_left = (resized_w - input_w) // 2
    crop_top = (resized_h - input_h) // 2
else:
    crop_left = 0
    crop_top = 0

# Transform upper-body coordinates
px1 = x1 * scale_x - crop_left
py1 = y1 * scale_y - crop_top
px2 = x2 * scale_x - crop_left
py2 = upper_y2 * scale_y - crop_top

gx1 = max(0, int(np.floor(px1 / patch_size)))
gy1 = max(0, int(np.floor(py1 / patch_size)))
gx2 = min(grid_w, int(np.ceil(px2 / patch_size)))
gy2 = min(grid_h, int(np.ceil(py2 / patch_size)))

if gx2 <= gx1 or gy2 <= gy1:
    raise RuntimeError(
        "Upper body falls outside DINOv3's processed image."
    )

# ---------------------------------------
# 7. Create upper-body reference
# ---------------------------------------

mask = np.zeros((grid_h, grid_w), dtype=bool)
mask[gy1:gy2, gx1:gx2] = True

selected = patch_features[
    torch.from_numpy(mask.flatten())
]

reference = F.normalize(
    selected.mean(dim=0),
    dim=0
)

# ---------------------------------------
# 8. Compute cosine similarity
# ---------------------------------------

similarity = patch_features @ reference
heatmap = similarity.reshape(grid_h, grid_w).cpu().numpy()

heatmap_norm = (
    (heatmap - heatmap.min())
    / (heatmap.max() - heatmap.min() + 1e-8)
)

# Create the image DINOv3 actually processed,
# without applying normalization.
processed_image = image.resize(
    (resized_w, resized_h),
    Image.Resampling.BICUBIC
)

if processor.do_center_crop:
    processed_image = processed_image.crop((
        crop_left,
        crop_top,
        crop_left + input_w,
        crop_top + input_h
    ))

heatmap_resized = Image.fromarray(
    np.uint8(heatmap_norm * 255)
).resize(
    (input_w, input_h),
    Image.Resampling.BILINEAR
)

# ---------------------------------------
# 9. Visualize
# ---------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(16, 6))

axes[0].imshow(image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(image)

rectangle = plt.Rectangle(
    (x1, y1),
    x2 - x1,
    upper_y2 - y1,
    fill=False,
    edgecolor="lime",
    linewidth=3
)

axes[1].add_patch(rectangle)
axes[1].set_title("YOLO-Guided Upper Body")
axes[1].axis("off")

axes[2].imshow(processed_image)
axes[2].imshow(
    heatmap_resized,
    cmap="jet",
    alpha=0.5,
    vmin=0,
    vmax=255
)

axes[2].set_title("DINOv3 Upper-Body Similarity")
axes[2].axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()

print(f"Saved: {OUTPUT_PATH}")
print("Done!")
