import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
h5_path = 'F:\\use_vgg16\\Traffic_Signal_Vgg16_10epoch_96%.h5'
model = load_model(h5_path)  # Replace with the path to your saved model

labels_path = 'F:\\use_vgg16\\dataset\\labels.csv'
classes_from_csv = pd.read_csv(labels_path)

# Define the class labels and their definitions
class_labels = [i for i in range(0, 58)]  # Replace with your class labels
class_definitions = list(classes_from_csv['Name'])  # Replace with your class definitions

# Open a connection to the video source (webcam or video file)
cap = cv2.VideoCapture(0)  # Use 0 for the default camera. You can replace it with the video file path.

while True:
    # Read a frame from the video source
    ret, frame = cap.read()

    # Preprocess the frame for model prediction
    img = cv2.resize(frame, (224, 224))  # Resize to match the model's input shape
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Check if the predicted class index is within the range of class_labels
    if 0 <= predicted_class < len(class_labels):
        predicted_label = class_labels[predicted_class]
        predicted_definition = class_definitions[predicted_class]

        # Display the prediction on the frame
        text = f"{predicted_label}: {predicted_definition}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        print("Invalid predicted class index:", predicted_class)

    # Display the frame
    cv2.imshow('Traffic Sign Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
