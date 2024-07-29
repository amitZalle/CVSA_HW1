from ultralytics import YOLO

# Paths to the data configuration files
augmented_data_yaml = '/home/student/Desktop/augmented_images.yaml'

# Initialize the start model
model = YOLO('/home/student/Desktop/start_model.pt')  # Use the appropriate YOLOv8 variant

model.train(data=augmented_data_yaml, epochs=100, save=True, project='/home/student/Desktop/augmented_model_logs')

# Save the final model
model.save('/home/student/Desktop/augmented_model.pt')
