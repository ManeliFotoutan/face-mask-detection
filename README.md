"# face-mask-detection" 
# Face Mask Detection

This repository contains code for training a face mask detection model using TensorFlow and Keras. The project consists of two main parts:

1. **Face Mask Detection Model Training:**
   - The `train_mask_detector.py` script trains a Convolutional Neural Network (CNN) using the MobileNetV2 architecture for face mask detection.
   - It uses a dataset of images containing faces with and without masks.
   - The trained model is saved as `mask_detector.model` and can be used for real-time mask detection.

2. **Real-time Face Mask Detection:**
   - The `real_time_mask_detection.py` script utilizes the trained model to perform real-time face mask detection using a webcam or video stream.
   - The detected faces are classified as "Mask" or "No Mask," and an alarm sound is played if a person is detected without a mask.

## Prerequisites

- Python 3
- TensorFlow
- OpenCV
- NumPy
- imutils
- playsound
- pygame
- Matplotlib

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
Usage
Training:

Place your dataset in the specified directory (DIRECTORY) with subdirectories for each class (e.g., with_mask, without_mask).

Run the training script:

python train_mask_detector.py
Real-time Detection:

After training, run the real-time detection script:

python real_time_mask_detection.py
Press 'q' to quit the video stream.

Files
train_mask_detector.py: Script for training the face mask detection model.
real_time_mask_detection.py: Script for real-time face mask detection.
face_detector: Directory containing the face detection model files (deploy.prototxt and res10_300x300_ssd_iter_140000.caffemodel).
dataset: Directory to store the training dataset.
alarm.mp3: Sound file used for the alarm.
mask_detector.model: Trained face mask detection model (generated after training).
Acknowledgments
The face detection model is based on the work by davisking.
