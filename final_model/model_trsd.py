import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from pathlib import Path

# Load the trained model
model = load_model("model_bbox_regression_and_classification.h5")

# Load the label binarizer
with open("lb.pickle", "rb") as f:
    lb = pickle.load(f)

# OpenCV setup for live video feed
cap = cv2.VideoCapture(0)  # Use 0 for default camera, change to a different number for other cameras

while True:
    # Read a frame from the video feed
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame")
        break

    # Preprocess the frame for the model
    frame_resized = cv2.resize(frame, (224, 224))
    frame_resized = img_to_array(frame_resized)
    frame_resized = frame_resized.reshape((1, frame_resized.shape[0], frame_resized.shape[1], frame_resized.shape[2]))
    frame_resized = frame_resized / 255.0

    # Perform inference
    predictions = model.predict(frame_resized)

    # Extract class label and bounding box from predictions
    class_index = np.argmax(predictions["class_label"])
    class_label = lb.classes_[class_index]
    bbox = predictions["bounding_box"][0]

    # Scale bounding box back to the original frame size
    h, w, _ = frame.shape
    xmin, ymin, xmax, ymax = int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h)

    # Draw bounding box and label on the frame
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    cv2.putText(frame, f"{class_label}", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Object Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
