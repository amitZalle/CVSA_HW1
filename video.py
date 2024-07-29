import cv2
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

def process_video_with_predictions(video_path, model_path, output_path):
    """
    Process a video, perform object detection on each frame, and save the annotated video.

    :param video_path: Path to the input video file.
    :param model_path: Path to the YOLO model weights file.
    :param output_path: Path to the output video file.
    """
    # Load the YOLO model
    model = YOLO(model_path)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Process each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Make predictions
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

        # Draw bounding boxes and class names on the frame
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
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)  # Draw the box

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
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, thickness=cv2.FILLED)

            # Put the label text on top of the black rectangle
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, font_color, 2, cv2.LINE_AA)

        # Write the frame to the output video
        out.write(frame)

    # Release resources
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

# Example usage
video_path = '/home/student/Desktop/CVSA_HW1_data/ood_video_data/surg_1.mp4'
model_path = '/home/student/Desktop/pseudo_trained_ood_model.pt'
output_path = '/home/student/Desktop/video_prediction.avi'
process_video_with_predictions(video_path, model_path, output_path)
