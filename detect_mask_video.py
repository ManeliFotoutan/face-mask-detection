# Import necessary libraries and modules
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os
from time import sleep
from threading import Thread
from playsound import playsound
import pygame

# Initialize pygame
pygame.init()

# Flag to control the alarm state
alarm_active = False

# Load the alarm sound using pygame
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound("alarm.mp3")

# Function to stop the alarm
def stop_alarm():
    global alarm_active
    alarm_sound.stop()
    alarm_active = False

# Function to play the alarm sound in a separate thread
def play_alarm():
    global alarm_active
    alarm_sound.play()
    time.sleep(3)  # Adjust the sleep time as needed
    stop_alarm()  # Stop the alarm after the specified time

# Function to detect faces and predict whether they are wearing a mask
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
        (104.0, 177.0, 123.0))

    # Set the input blob for the face detection model
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    faces = []
    locs = []
    preds = []

    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            # Calculate the bounding box coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensure the bounding box coordinates are within the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Extract the face region, preprocess it, and add to the list
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # If faces are detected, perform mask prediction
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

# Paths to the face detection model and pre-trained weights
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load the trained mask detection model
maskNet = load_model("mask_detector.model")

# Start the video stream
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# Main loop to process video frames
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=900)

    # Detect faces and predict masks
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Loop over the detected faces and their predictions
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # Determine label and color based on mask prediction
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # Check if the label is "No Mask" and the alarm is not active
        if label == "No Mask" and not alarm_active:
            # Use threading to play the alarm sound in the background
            alarm_thread = Thread(target=play_alarm)
            alarm_thread.start()
            alarm_active = True
        elif label == "Mask" and alarm_active:
            # If the label is now "Mask", stop the alarm
            stop_alarm()

        # Display the label and bounding box on the frame
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Display the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Check if the 'q' key is pressed to exit the loop
    if key == ord("q"):
        # If the alarm is still active when quitting, stop it
        if alarm_active:
            stop_alarm()
        break

# Clean up and release resources
cv2.destroyAllWindows()
vs.stop()
