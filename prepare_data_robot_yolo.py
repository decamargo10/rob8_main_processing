import cv2
import os
import glob

# Input and output directory
input_dir = '/home/decamargo/Documents/dataset/yellowtool'  # Input directory
output_dir = f'{input_dir}_converted'  # Output directory

# Create output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Get all image files in the input directory
image_files = glob.glob(os.path.join(input_dir, '*'))
image_files.sort()  # Sort the files

# Process every 10th image
for i in range(0, len(image_files), 10):
    # Read image in BGR format
    img = cv2.imread(image_files[i])

    # Check if the image is read properly
    if img is None:
        print(f'Unable to read image: {image_files[i]}')
        continue

    # Convert image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save image in the output directory
    output_file = os.path.join(output_dir, f'image_{i // 10:04}.jpg')
    cv2.imwrite(output_file, img_rgb)

print('Done processing images')
