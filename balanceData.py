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

def augment_image(image, bboxes, class_labels):
    augmented = augmentations(image=image, bboxes=bboxes, class_labels=class_labels)
    
    # Calculate image dimensions
    image_width, image_height = image.shape[1], image.shape[0]
    original_centers = [(bbox[0] * image_width, bbox[1] * image_height) for bbox in bboxes]
    
    # Adjust bounding boxes based on their distance from original centers
    adjusted_bboxes = adjust_bbox_centered(augmented['bboxes'], image_width, image_height, original_centers)
    
    return augmented['image'], adjusted_bboxes, augmented['class_labels']

def augment_images_with_class(image_folder, label_folder, output_image_folder, output_label_folder, class_id, target_num_augmentations=30):
    # Extract images with specified class_id
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    augmentations_done = 0
    
    for image_file in image_files:
        if augmentations_done >= target_num_augmentations:
            break
        
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt'))

        if os.path.exists(label_path):
            image = cv2.imread(image_path)
            with open(label_path, 'r') as file:
                lines = file.readlines()
                bboxes = [list(map(float, line.strip().split()[1:])) for line in lines if int(line.strip().split()[0]) == class_id]
                class_labels = [int(line.strip().split()[0]) for line in lines if int(line.strip().split()[0]) == class_id]

            if bboxes:
                # Augment the image with specified class_id labels
                while augmentations_done < target_num_augmentations:
                    augmented_image, augmented_bboxes, augmented_class_labels = augment_image(image, bboxes, class_labels)

                    if not augmented_bboxes:
                        print(f"Augmentation failed for {image_file}. Retrying...")
                        continue  # Retry if augmentation does not yield valid bounding boxes

                    output_image_path = os.path.join(output_image_folder, f"{os.path.splitext(image_file)[0]}_aug_{augmentations_done}.jpg")
                    output_label_path = os.path.join(output_label_folder, f"{os.path.splitext(image_file)[0]}_aug_{augmentations_done}.txt")

                    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
                    os.makedirs(os.path.dirname(output_label_path), exist_ok=True)

                    cv2.imwrite(output_image_path, augmented_image)

                    with open(output_label_path, 'w') as file:
                        for bbox, class_label in zip(augmented_bboxes, augmented_class_labels):
                            file.write(f"{class_label} " + ' '.join(map(str, bbox)) + '\n')
                    
                    augmentations_done += 1
                    print(f"Created {augmentations_done} augmentations...")

                    if augmentations_done >= target_num_augmentations:
                        break

# Function to delete contents of a directory
def clear_directory(directory):
    if os.path.exists(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)

def copy_directory(src_dir, dest_dir):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)

# Paths
SOURCE_DIR = '/home/student/Desktop/CVSA_HW1_data/labeled_image_data'
DUPLICATE_DIR = '/home/student/Desktop/CVSA_HW1_data/balanced_labeled_image_data'

IMAGE_FOLDER = os.path.join(DUPLICATE_DIR, 'images/train')
LABEL_FOLDER = os.path.join(DUPLICATE_DIR, 'labels/train')
OUTPUT_IMAGE_FOLDER = IMAGE_FOLDER
OUTPUT_LABEL_FOLDER = LABEL_FOLDER

# Clear existing data and create output directories if they do not exist
clear_directory(DUPLICATE_DIR)

# Copy original data
copy_directory(SOURCE_DIR, DUPLICATE_DIR)

# Augment images with class 0 labels
augment_images_with_class(IMAGE_FOLDER, LABEL_FOLDER, OUTPUT_IMAGE_FOLDER, OUTPUT_LABEL_FOLDER, class_id=0, target_num_augmentations=10)
augment_images_with_class(IMAGE_FOLDER, LABEL_FOLDER, OUTPUT_IMAGE_FOLDER, OUTPUT_LABEL_FOLDER, class_id=1, target_num_augmentations=10)
