# CVSA_HW1
HW1 under the course CV for surgical application.

This project create a model that predict boundrary boxes, classes(Empty, Tweezers, Needle-Driver) and confidence level on photos and images of surgical operations.
We have an initial 61 labeled images in YOLO format, two ID videos, and two OOD videos.

To run this you must have the data needed inside CVSA_HW1_data folder(inside it: labeled_image_data, id_video_data, ood_video_data).
You also must install ultralytics and albumentations libraries, and use the requierments.txt file given.

First connect to conda enviournment of YOLO in the terminal with:
source yolov8_env/bin/activate

To run the code itself you must fix the paths in all files that will fit. if you want to use it as it is, you must have the python files in '/home/student/Desktop' and the data in a folder as said above in it.

To run the training of the model, you must have 'yolov8n.pt' file in the same folder, and run the file 'runProgram.py'.

The final model weights are saved as "pseudo_trained_ood_model.pt" and the download link of the final model's weights is: https://drive.google.com/file/d/1wrFZihOdjx314JzT6Va0Pe7LxtDrLJ00/view?usp=sharing

To predict on an image, you must insert the image path inside the 'predict.py' file and run it. the prediction will appear in the main folder under 'image_prediction.png'.

To predict on a video, you must insert the video path inside the 'video.py' file and run it. the prediction will appear in the main folder under 'video_prediction.avi'.
