import albumentations as A
import cv2
from pathlib import Path
import os

def read_yolo_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
    labels = [list(map(float, line.strip().split())) for line in lines]
    return labels

def write_yolo_labels(labels, label_path):
    with open(label_path, 'w') as f:
        for label in labels:
            line = ' '.join(map(str, label))
            f.write(line + '\n')

def get_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2),
        A.RGBShift(p=0.2),
        A.RandomCrop(height=400, width=400, p=0.5),
        # Add more augmentation techniques as needed
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))



images_path = '/home/decamargo/Documents/uni/yolov7/train/images'
labels_path = '/home/decamargo/Documents/uni/yolov7/train/labels'

# Get a list of all image files in the images folder
image_files = [f for f in os.listdir(images_path) if f.endswith('.jpg')]

# Loop through the image files
for image_file in image_files:
    # Construct the paths to the image and label files
    image_path = os.path.join(images_path, image_file)
    label_path = os.path.join(labels_path, image_file.replace('.jpg', '.txt'))

    # Read the image and labels
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    labels = read_yolo_labels(label_path)

    # Separate class labels and bounding box coordinates
    class_labels, bboxes = zip(*[(int(label[0]), label[1:]) for label in labels])

    # Apply augmentation
    augmentation = get_augmentation_pipeline()
    augmented = augmentation(image=image, bboxes=bboxes, class_labels=class_labels)

    # Extract the augmented image and labels
    augmented_image = augmented['image']
    augmented_bboxes = augmented['bboxes']
    augmented_class_labels = augmented['class_labels']

    # Convert the class labels back to float and concatenate with the augmented bounding boxes
    augmented_labels = [[cls, *bbox] for cls, bbox in zip(augmented_class_labels, augmented_bboxes)]

    # Construct the paths to save the augmented image and label files
    augmented_image_file = 'augmented_' + image_file
    augmented_label_file = 'augmented_' + image_file.replace('.jpg', '.txt')
    augmented_image_path = os.path.join(images_path, augmented_image_file)
    augmented_label_path = os.path.join(labels_path, augmented_label_file)

    # Save the augmented image
    augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(augmented_image_path, augmented_image)

    # Save the augmented labels
    write_yolo_labels(augmented_labels, augmented_label_path)


