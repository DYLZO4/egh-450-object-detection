import argparse
import os
import shutil

from ultralytics import YOLO

# COCO class indices for the classes we care about, mapped to our dataset's class indices
# COCO: 0 = person, 1 = backpack
COCO_TO_LOCAL = {0: 0, 1: 1}   # person -> 0, backpack -> 1
CLASS_NAMES = {0: 'person', 1: 'backpack'}  # local class id -> name, used for obj.names

MODEL_WEIGHTS = 'yolo11n.pt'    # stock COCO-pretrained checkpoint (auto-downloads if missing)
CONF_THRESHOLD = 0.4
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

parser = argparse.ArgumentParser(description='Auto-label images with a pretrained YOLO model, producing a CVAT-import-ready YOLO 1.1 folder.')
parser.add_argument('image_dir', help='Directory of unlabeled images to run detection on')
parser.add_argument('out_dir', help='Directory to build the CVAT import structure in (e.g. cvat_import)')
parser.add_argument('--weights', default=MODEL_WEIGHTS, help=f'Model weights to use (default: {MODEL_WEIGHTS})')
parser.add_argument('--conf', type=float, default=CONF_THRESHOLD, help=f'Confidence threshold (default: {CONF_THRESHOLD})')
args = parser.parse_args()

image_dir = args.image_dir
out_dir = args.out_dir
data_dir = os.path.join(out_dir, 'obj_train_data')

os.makedirs(data_dir, exist_ok=True)

model = YOLO(args.weights)

results = model.predict(source=image_dir, conf=args.conf, classes=list(COCO_TO_LOCAL.keys()), stream=True)

n_images = 0
n_boxes = 0
image_paths = []  # relative paths written into train.txt, in processed order

for r in results:
    n_images += 1
    img_filename = os.path.basename(r.path)
    img_name = os.path.splitext(img_filename)[0]
    label_path = os.path.join(data_dir, f'{img_name}.txt')

    # copy the source image alongside its label so the folder is self-contained
    shutil.copy2(r.path, os.path.join(data_dir, img_filename))

    with open(label_path, 'w') as f:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_TO_LOCAL:
                continue
            local_cls = COCO_TO_LOCAL[cls_id]
            x, y, w, h = box.xywhn[0].tolist()  # normalized YOLO format
            f.write(f'{local_cls} {x} {y} {w} {h}\n')
            n_boxes += 1

    image_paths.append(f'obj_train_data/{img_filename}')

# --- Build the rest of the CVAT YOLO 1.1 import structure ---

# obj.names: one class name per line, index = local class id
max_id = max(CLASS_NAMES.keys())
with open(os.path.join(out_dir, 'obj.names'), 'w') as f:
    for i in range(max_id + 1):
        f.write(f'{CLASS_NAMES.get(i, f"class_{i}")}\n')

# obj.data: points everything together
with open(os.path.join(out_dir, 'obj.data'), 'w') as f:
    f.write(f'classes = {max_id + 1}\n')
    f.write('train = train.txt\n')
    f.write('names = obj.names\n')
    f.write('backup = backup/\n')

# train.txt: list of image paths, matching what CVAT already has in the task
with open(os.path.join(out_dir, 'train.txt'), 'w') as f:
    for p in image_paths:
        f.write(f'{p}\n')

print(f'Done. Processed {n_images} images, wrote {n_boxes} boxes total.')
print(f'CVAT-import-ready folder built at "{out_dir}/":')
print(f'  {out_dir}/obj.names')
print(f'  {out_dir}/obj.data')
print(f'  {out_dir}/train.txt')
print(f'  {out_dir}/obj_train_data/  ({n_images} label files)')
print(f'\nZip it with:\n  cd {out_dir} && zip -r ../cvat_import.zip . && cd ..')