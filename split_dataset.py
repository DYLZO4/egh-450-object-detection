import os
import random
import shutil

random.seed(42)

image_dir = 'obj_train_data/obj_train_data'  # adjust to wherever your export unzipped to
images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
random.shuffle(images)

split_idx = int(len(images) * 0.8)
train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

for split, img_list in [('train', train_imgs), ('val', val_imgs)]:
    for img in img_list:
        base = os.path.splitext(img)[0]
        shutil.copy(os.path.join(image_dir, img), f'dataset/images/{split}/{img}')
        shutil.copy(os.path.join(image_dir, f'{base}.txt'), f'dataset/labels/{split}/{base}.txt')

print(f'Train: {len(train_imgs)}, Val: {len(val_imgs)}')
