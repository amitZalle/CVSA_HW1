import cv2
import os
import shutil
from ultralytics import YOLO
import random
import torch

# Paths
source_folder = '/home/student/Desktop/CVSA_HW1_data/ood_video_data'
frames_folder = '/home/student/Desktop/CVSA_HW1_data/HCframes_OOD/frames'
labels_folder = '/home/student/Desktop/CVSA_HW1_data/HCframes_OOD/labels'
dataset_folder = '/home/student/Desktop/CVSA_HW1_data/HCframes_OOD/YOLOds'
train_images_folder = os.path.join(dataset_folder, 'images/train')
train_labels_folder = os.path.join(dataset_folder, 'labels/train')
val_images_folder = os.path.join(dataset_folder, 'images/val')
val_labels_folder = os.path.join(dataset_folder, 'labels/val')

# Function to delete contents of a directory
def clear_directory(directory):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

# Load the trained YOLOv8 model
model = YOLO('/home/student/Desktop/pseudo_trained_2_model.pt')  # path to the saved model

# Define source folder and output folders for high-confidence frames and labels
os.makedirs(frames_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)
os.makedirs(dataset_folder, exist_ok=True)

# Clear existing data in the relevant directories
clear_directory(frames_folder)
clear_directory(labels_folder)
clear_directory(dataset_folder)

def normalize_bbox(x1, y1, x2, y2, image_width, image_height):
    x_center = (x1 + x2) / 2 / image_width
    y_center = (y1 + y2) / 2 / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height
    return x_center, y_center, width, height

# Get a list of all video files in the source folder
video_files = [f for f in os.listdir(source_folder) if f.endswith(('.mp4', '.avi', '.mov'))]

# Process each video file
for video_file in video_files:
    input_video_path = os.path.join(source_folder, video_file)

    # Open the input video
    cap = cv2.VideoCapture(input_video_path)

    frame_count = 0

    # Process each frame of the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run the model on the current frame
        results = model(frame)

        # Dictionary to keep track of the highest confidence box for each class
        highest_conf_boxes = {}

        # Extract bounding boxes, labels, and confidences
        for result in results:
            boxes = result.boxes.xyxy  # Get bounding boxes in xyxy format
            confidences = result.boxes.conf  # Get confidence scores
            class_ids = result.boxes.cls  # Get class indices
            class_names = result.names  # Get class names


            for box, conf, class_id in zip(boxes, confidences, class_ids):
                x1, y1, x2, y2 = map(float, box[:4])  # Get the coordinates of the box
                width = x2 - x1
                height = y2 - y1
                #print(conf)

                if (torch.min(confidences) >= 0.90  and len(torch.unique(class_ids))>1) or (torch.min(confidences) >= 0.99  and len(torch.unique(class_ids))==1):
                    class_id = int(class_id)
                    if class_id not in highest_conf_boxes or max(width,height)<highest_conf_boxes[class_id]['size']:    #conf > highest_conf_boxes[class_id]['conf']:
                        highest_conf_boxes[class_id] = {'conf': conf, 'box': box, 'size': max(width,height)}

        if highest_conf_boxes:
            # Save the frame with high confidence detections
            frame_filename = f"frame_{frame_count}.jpg"
            frame_path = os.path.join(frames_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

            # Save the label file for this frame with normalized coordinates
            label_filename = f"frame_{frame_count}.txt"
            label_path = os.path.join(labels_folder, label_filename)
            with open(label_path, 'w') as label_file:
                for class_id, data in highest_conf_boxes.items():
                    box = data['box']
                    x1, y1, x2, y2 = map(float, box[:4])  # Get the coordinates of the box
                    image_height, image_width = frame.shape[:2]  # Get image dimensions

                    # Normalize the bounding box coordinates
                    x_center, y_center, width, height = normalize_bbox(x1, y1, x2, y2, image_width, image_height)
                    label = class_names[class_id]  # Get the class label

                    label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        frame_count += 1

    # Release resources for the current video
    cap.release()

print("Frames with high-confidence detections have been saved and normalized.")

# Define paths for dataset organization
os.makedirs(train_images_folder, exist_ok=True)
os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)

# Move high-confidence frames and labels to the dataset folders
for file_name in os.listdir(frames_folder):
    if file_name.endswith('.jpg'):
        # Move frame
        src_frame = os.path.join(frames_folder, file_name)
        dest_frame = os.path.join(train_images_folder, file_name)
        shutil.move(src_frame, dest_frame)
        
        # Move corresponding label
        label_file_name = file_name.replace('.jpg', '.txt')
        src_label = os.path.join(labels_folder, label_file_name)
        dest_label = os.path.join(train_labels_folder, label_file_name)
        
        if os.path.exists(src_label):
            shutil.move(src_label, dest_label)
        else:
            print(f"Label file {label_file_name} not found for {file_name}")

# List all image files in the training images folder
image_files = [f for f in os.listdir(train_images_folder) if f.endswith('.jpg')]

print(f"Dataset prepared successfully. with {len(image_files)} frames")




# Ensure validation directories exist
os.makedirs(val_images_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)


# Copy the entire directory tree from source to destination
shutil.copytree('/home/student/Desktop/CVSA_HW1_data/labeled_image_data/images/val', val_images_folder, dirs_exist_ok=True)
shutil.copytree('/home/student/Desktop/CVSA_HW1_data/labeled_image_data/labels/val', val_labels_folder, dirs_exist_ok=True)