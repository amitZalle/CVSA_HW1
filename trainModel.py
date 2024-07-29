import cv2
from ultralytics import YOLO

# Define the model and dataset paths
model_path = '/home/student/Desktop/start_model.pt'  # Path to the saved model
data_path = '/home/student/Desktop/labeled_images.yaml'  # Path to the dataset YAML file

# Initialize the YOLO model
model = YOLO('yolov8n.pt')  # Initialize with a pre-trained model if available

# Train the model
try:
    # Note: the `project` parameter is used to specify the directory for TensorBoard logs
    model.train(data=data_path, epochs=100, save=True, project='/home/student/Desktop/start_model_logs')
except Exception as e:
    print(f"An error occurred during training: {e}")
    exit(1)

# After training
print(model.ckpt)  # Check if the checkpoint is not None
if model.ckpt:
    try:
        model.save(model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"An error occurred while saving the model: {e}")
        exit(1)
else:
    print("Model checkpoint is None, unable to save the model.")
    exit(1)
