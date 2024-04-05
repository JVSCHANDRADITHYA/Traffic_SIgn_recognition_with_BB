import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import gradio as gr

# Load the trained model
h5_path = r"F:\basic_models\road_sign_detect_inception_net\resources\traffic_signal_model_final.h5"
model = load_model(h5_path)  # Replace with the path to your saved model

labels_path = r"F:\basic_models\road_sign_detect_inception_net\dataset\labels.csv"
classes_from_csv = pd.read_csv(labels_path)

# Define the class labels and their definitions
class_labels = [i for i in range(0, 58)]  # Replace with your class labels
class_definitions = list(classes_from_csv['Name'])  # Replace with your class definitions

# Function to make predictions
def predict_traffic_sign(img):
    # Preprocess the image for model prediction
    img = cv2.resize(img, (75, 75))
    img = img / 255.0  # Normalize the pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Make predictions
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)

    # Check if the predicted class index is within the range of class_labels
    if 0 <= predicted_class < len(class_labels):
        predicted_label = class_labels[predicted_class]
        predicted_definition = class_definitions[predicted_class]

        return f"{predicted_label}: {predicted_definition}"
    else:
        return "Invalid predicted class index"

# Gradio Interface
iface = gr.Interface(
    fn=predict_traffic_sign,
    inputs=gr.Image(sources="webcam", streaming=True),
    outputs="text",
    live=True,
)

# Launch the Gradio interface
iface.launch()
