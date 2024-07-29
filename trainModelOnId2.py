from ultralytics import YOLO

# Load the pretrained YOLO model
pretrained_model_path = '/home/student/Desktop/pseudo_trained_model.pt'  # Path to the pre-trained model
new_model_save_path = '/home/student/Desktop/pseudo_trained_2_model.pt'  # Path to save the new fine-tuned model

# Load the model
model = YOLO(pretrained_model_path)

# Train the model on the new high-confidence dataset
model.train(data='/home/student/Desktop/pseudo_id_2.yaml', epochs=100, save=True, project='/home/student/Desktop/pseudo2_trained_model_logs')

# Save the new model separately
model.save(new_model_save_path)

print(f"New fine-tuned model saved to {new_model_save_path}")
