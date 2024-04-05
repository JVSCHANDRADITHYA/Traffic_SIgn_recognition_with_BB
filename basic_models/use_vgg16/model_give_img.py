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

# Ask the user for the path to the test image
test_image_path = r'F:\use_vgg16\dataset\traffic_Data\TEST\000_1_0004_1_j.png'

# Read the test image
img = cv2.imread(test_image_path)

# Check if the image is loaded successfully
if img is None:
    print(f"Error: Unable to read image from {test_image_path}")
else:
    # Preprocess the image for model prediction
    img = cv2.resize(img, (224, 224))  # Resize to match the model's input shape
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Check if the predicted class index is within the range of class_labels
    if 0 <= predicted_class < len(class_labels):
        predicted_label = class_labels[predicted_class]
        predicted_definition = class_definitions[predicted_class]

        # Display the prediction
        print(f"Predicted class: {predicted_label}, Definition: {predicted_definition}")
    else:
        print("Invalid predicted class index:", predicted_class)
