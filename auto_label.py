import argparse
import os

from ultralytics import YOLO

# COCO class indices for the classes we care about, mapped to our dataset's class indices
# COCO: 0 = person, 24 = backpack
COCO_TO_LOCAL = {0: 0, 24: 1}   # person -> 0, backpack -> 1
MODEL_WEIGHTS = 'yolo11n.pt'    # stock COCO-pretrained checkpoint (auto-downloads if missing)
CONF_THRESHOLD = 0.4

parser = argparse.ArgumentParser(description='Auto-label images with a pretrained YOLO model, producing YOLO-format label files.')
parser.add_argument('image_dir', help='Directory of unlabeled images to run detection on')
parser.add_argument('out_dir', help='Directory to save YOLO-format .txt label files to')
parser.add_argument('--weights', default=MODEL_WEIGHTS, help=f'Model weights to use (default: {MODEL_WEIGHTS})')
parser.add_argument('--conf', type=float, default=CONF_THRESHOLD, help=f'Confidence threshold (default: {CONF_THRESHOLD})')
args = parser.parse_args()

image_dir = args.image_dir
out_dir = args.out_dir

os.makedirs(out_dir, exist_ok=True)

model = YOLO(args.weights)

results = model.predict(source=image_dir, conf=args.conf, classes=list(COCO_TO_LOCAL.keys()), stream=True)

n_images = 0
n_boxes = 0

for r in results:
    n_images += 1
    img_name = os.path.splitext(os.path.basename(r.path))[0]
    label_path = os.path.join(out_dir, f'{img_name}.txt')

    with open(label_path, 'w') as f:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in COCO_TO_LOCAL:
                continue
            local_cls = COCO_TO_LOCAL[cls_id]
            x, y, w, h = box.xywhn[0].tolist()  # normalized YOLO format
            f.write(f'{local_cls} {x} {y} {w} {h}\n')
            n_boxes += 1

print(f'Done. Processed {n_images} images, wrote {n_boxes} boxes total, labels saved to "{out_dir}".')
