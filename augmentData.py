import os
import cv2
import albumentations as A
import numpy as np
import shutil

# Define augmentations
augmentations = A.Compose([
    A.RandomCrop(width=640, height=480),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.2),
    A.GaussNoise(p=0.2),
    A.Blur(blur_limit=3, p=0.2),
    A.Resize(width=640, height=480),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def adjust_bbox_centered(bboxes, image_width, image_height, original_centers, threshold=0.3):
    adjusted_bboxes = []
    for bbox, original_center in zip(bboxes, original_centers):
        x_center, y_center, width, height = bbox
        
        # Convert normalized coordinates to pixel coordinates
        x1 = (x_center - width / 2) * image_width
        y1 = (y_center - height / 2) * image_height
        x2 = (x_center + width / 2) * image_width
        y2 = (y_center + height / 2) * image_height
        
        # Calculate new bbox center after augmentation
        new_x_center = (x1 + x2) / 2
        new_y_center = (y1 + y2) / 2
        
        # Calculate distance from the original center
        original_x_center, original_y_center = original_center
        distance = np.sqrt((new_x_center - original_x_center) ** 2 + (new_y_center - original_y_center) ** 2)
        
        # Keep bbox if within threshold distance, otherwise discard
        if distance <= threshold * max(image_width, image_height):
            # Normalize the adjusted bbox
            adjusted_width = (x2 - x1) / image_width
            adjusted_height = (y2 - y1) / image_height
            adjusted_x_center = (x1 + x2) / 2 / image_width
            adjusted_y_center = (y1 + y2) / 2 / image_height
            
            adjusted_bboxes.append([adjusted_x_center, adjusted_y_center, adjusted_width, adjusted_height])
    
    return adjusted_bboxes

def augment_image(image, bboxes, class_labels, max_retries=10):
    retries = 0
    while retries < max_retries:
        augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
        
        # Calculate image dimensions
        image_width, image_height = image.shape[1], image.shape[0]
        original_centers = [(bbox[0] * image_width, bbox[1] * image_height) for bbox in bboxes]
        
        # Adjust bounding boxes based on their distance from original centers
        adjusted_bboxes = adjust_bbox_centered(augmented['bboxes'], image_width, image_height, original_centers)
        
        if adjusted_bboxes:  # If there are bounding boxes after augmentation
            return augmented['image'], adjusted_bboxes, augmented['class_labels']
        
        retries += 1
    
    # Return the last augmentation attempt even if no bounding boxes are left
    return augmented['image'], adjusted_bboxes, augmented['class_labels']

def augment_dataset(image_folder, label_folder, output_image_folder, output_label_folder, num_augmentations=3):
    for image_file in os.listdir(image_folder):
        if image_file.endswith('.jpg'):
            image_path = os.path.join(image_folder, image_file)
            label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt'))

            if os.path.exists(label_path):
                image = cv2.imread(image_path)
                with open(label_path, 'r') as file:
                    lines = file.readlines()
                    bboxes = [list(map(float, line.strip().split()[1:])) for line in lines]
                    class_labels = [int(line.strip().split()[0]) for line in lines]

                for i in range(num_augmentations):
                    augmented_image, augmented_bboxes, augmented_class_labels = augment_image(image, bboxes, class_labels)

                    output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_file)[0]}_aug_{i}.jpg")
                    output_label_path = os.path.join(output_label_folder, f"{os.path.splitext(image_file)[0]}_aug_{i}.txt")

                    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

                    cv2.imwrite(output_image_path, augmented_image)

                    with open(output_label_path, 'w') as file:
                        for bbox, class_label in zip(augmented_bboxes, augmented_class_labels):
                            file.write(f"{class_label} " + ' '.join(map(str, bbox)) + '\n')

# Function to delete contents of a directory
def clear_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# Augment train
# Paths
TRAIN_IMAGE_FOLDER = '/home/student/Desktop/CVSA_HW1_data/labeled_image_data/images/train'
TRAIN_LABEL_FOLDER = '/home/student/Desktop/CVSA_HW1_data/labeled_image_data/labels/train'
OUTPUT_IMAGE_FOLDER = '/home/student/Desktop/CVSA_HW1_data/labeled_augmented_data/images/train'
OUTPUT_LABEL_FOLDER = '/home/student/Desktop/CVSA_HW1_data/labeled_augmented_data/labels/train'

os.makedirs(OUTPUT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_LABEL_FOLDER, exist_ok=True)

# Clear existing data in the relevant directories
clear_directory(OUTPUT_IMAGE_FOLDER)
clear_directory(OUTPUT_LABEL_FOLDER)

augment_dataset(TRAIN_IMAGE_FOLDER, TRAIN_LABEL_FOLDER, OUTPUT_IMAGE_FOLDER, OUTPUT_LABEL_FOLDER)


#copy validation
# Paths
VAL_IMAGE_FOLDER = '/home/student/Desktop/CVSA_HW1_data/labeled_image_data/images/val'
VAL_LABEL_FOLDER = '/home/student/Desktop/CVSA_HW1_data/labeled_image_data/labels/val'
OUTPUT_VAL_IMAGE_FOLDER = '/home/student/Desktop/CVSA_HW1_data/labeled_augmented_data/images/val'
OUTPUT_VAL_LABEL_FOLDER = '/home/student/Desktop/CVSA_HW1_data/labeled_augmented_data/labels/val'

os.makedirs(OUTPUT_VAL_IMAGE_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_VAL_LABEL_FOLDER, exist_ok=True)


# Copy the entire directory tree from source to destination
shutil.copytree(VAL_IMAGE_FOLDER, OUTPUT_VAL_IMAGE_FOLDER, dirs_exist_ok=True)
shutil.copytree(VAL_LABEL_FOLDER, OUTPUT_VAL_LABEL_FOLDER, dirs_exist_ok=True)