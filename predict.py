import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

def get_color_and_name_for_label(label):
    """
    Define colors and names for each label.
    :param label: Integer label.
    :return: Tuple of (color, name).
    """
    colors = {
        0: (255, 0, 0),   # Blue - Empty
        1: (0, 255, 0),   # Green - Tweezers
        2: (0, 0, 255),   # Red - Needle driver
        3: (255, 255, 0), # Yellow
        4: (255, 0, 255), # Magenta
        5: (0, 255, 255)  # Cyan
    }
    names = {
        0: 'Empty',
        1: 'Tweezers',
        2: 'Needle driver',
        3: 'Class 3',
        4: 'Class 4',
        5: 'Class 5'
    }
    color = colors.get(label, (255, 255, 255))  # Default to white if label not in colors
    name = names.get(label, 'Unknown')
    return color, name

def process_image_with_predictions(image_path, model_path):
    """
    Process an image, perform object detection, and annotate the image with bounding boxes and class names.

    :param image_path: Path to the input image file.
    :param model_path: Path to the YOLO model weights file.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Load the image
    image = cv2.imread(image_path)
    image_height, image_width = image.shape[:2]

    # Make predictions
    frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb)

    # Extract predictions from the results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes in xyxy format
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)  # Class IDs
    confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores

    # Organize predictions by class
    predictions = []
    for box, class_id, confidence in zip(boxes, class_ids, confidences):
        predictions.append({'box': box, 'class': class_id, 'confidence': confidence})

    # Filter boxes: keep only the highest confidence box per class and at most two for class 0
    filtered_boxes = {}
    for pred in predictions:
        class_id = pred['class']
        if class_id not in filtered_boxes:
            filtered_boxes[class_id] = []
        filtered_boxes[class_id].append(pred)

    final_boxes = []
    for class_id, preds in filtered_boxes.items():
        if class_id == 0:
            # Sort by confidence and take up to 2 for class 0
            top_preds = sorted(preds, key=lambda x: x['confidence'], reverse=True)[:2]
        else:
            # Take the top 1 for other classes
            top_preds = [max(preds, key=lambda x: x['confidence'])]

        final_boxes.extend(top_preds)

    # Draw bounding boxes and class names on the image
    thickness = 10
    font_scale = 1.5
    font_color = (255, 255, 255)  # White text color
    font = cv2.FONT_HERSHEY_SIMPLEX
    background_color = (0, 0, 0)  # Black background for text

    for pred in final_boxes:
        box = pred['box']
        class_id = pred['class']
        confidence = pred['confidence']
        x1, y1, x2, y2 = map(int, box)
        color, name = get_color_and_name_for_label(class_id)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)  # Draw the box

        # Draw a black rectangle behind the text
        label = f"{name}: {confidence:.2f}"
        text_size, _ = cv2.getTextSize(label, font, font_scale, 1)
        text_w, text_h = text_size
        text_x = x1
        text_y = y1 - 10 if y1 - 10 > text_h else y1 + text_h + 10

        rect_x1 = text_x
        rect_y1 = text_y - text_h - 10
        rect_x2 = text_x + text_w
        rect_y2 = text_y + 10
        cv2.rectangle(image, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, thickness=cv2.FILLED)

        # Put the label text on top of the black rectangle
        cv2.putText(image, label, (text_x, text_y), font, font_scale, font_color, 2, cv2.LINE_AA)

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Save the plot as an image file
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.savefig('/home/student/Desktop/image_prediction.png')  # Specify the path where you want to save the plot
    plt.close()

# Example usage
image_path = '/home/student/Desktop/CVSA_HW1_data/HCframes_OOD/YOLOds/images/train/frame_468.jpg'
model_path = '/home/student/Desktop/pseudo_trained_ood_model.pt'
process_image_with_predictions(image_path, model_path)
