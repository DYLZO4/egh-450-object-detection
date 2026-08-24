import os
import random
import shutil

random.seed(42)

image_dir = 'cvat_export/obj_train_data/obj_train_data'  # adjust to wherever your export unzipped to

# Split ratios (must sum to 1.0)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1

images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
random.shuffle(images)

n = len(images)
train_end = int(n * TRAIN_RATIO)
val_end = train_end + int(n * VAL_RATIO)

train_imgs = images[:train_end]
val_imgs = images[train_end:val_end]
test_imgs = images[val_end:]

for split, img_list in [('train', train_imgs), ('val', val_imgs), ('test', test_imgs)]:
    os.makedirs(f'dataset/images/{split}', exist_ok=True)
    os.makedirs(f'dataset/labels/{split}', exist_ok=True)
    for img in img_list:
        base = os.path.splitext(img)[0]
        shutil.copy(os.path.join(image_dir, img), f'dataset/images/{split}/{img}')
        shutil.copy(os.path.join(image_dir, f'{base}.txt'), f'dataset/labels/{split}/{base}.txt')

print(f'Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}')