import os
import shutil
import random

src_path = "/home/decamargo/Documents/uni/data/"

src_dirs = ['Yellow_tool', 'Multimeter', 'Tape', 'Ninja', 'Screwdriver', 'Hot_glue']
output_dir = 'output'

train_ratio = 0.7
test_ratio = 0.15
valid_ratio = 0.15

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create subdirectories for training, testing, and validation sets
sets = ['train', 'test', 'val']
for set_name in sets:
    os.makedirs(os.path.join(output_dir, set_name, 'images'))
    os.makedirs(os.path.join(output_dir, set_name, 'annotations'))

all_data = []

# Collect all image and corresponding annotation files from the source directories
for src_dir in src_dirs:
    src_dir = src_path + src_dir
    images = [f for f in os.listdir(src_dir) if f.endswith('.jpg') or f.endswith('.png')]

    for img in images:
        img_path = os.path.join(src_dir, img)
        img_name, img_ext = os.path.splitext(img)
        anno_name = img_name + '.txt'
        anno_path = os.path.join(src_dir, anno_name)

        if os.path.exists(anno_path):
            all_data.append((img_path, anno_path))

# Shuffle the data and split it into training, testing, and validation sets
random.shuffle(all_data)

train_size = int(len(all_data) * train_ratio)
test_size = int(len(all_data) * test_ratio)
valid_size = len(all_data) - train_size - test_size

train_data = all_data[:train_size]
test_data = all_data[train_size:train_size + test_size]
valid_data = all_data[train_size + test_size:]

# Copy the files to the respective output folders
for set_name, dataset in zip(sets, [train_data, test_data, valid_data]):
    for img, anno in dataset:
        img_filename = os.path.basename(img)
        anno_filename = os.path.basename(anno)

        shutil.copyfile(img, os.path.join(output_dir, set_name, 'images', img_filename))
        shutil.copyfile(anno, os.path.join(output_dir, set_name, 'annotations', anno_filename))

print("Data split and copied to the output directory.")
