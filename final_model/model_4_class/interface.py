import gradio as gr
import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model and label binarizer
model = load_model(r"F:\TSR_Model\final_model\resources\model_bbox_regression_and_classification.h5")
lb = pickle.loads(open(r"F:\TSR_Model\final_model\resources\lb.pickle", "rb").read())

# Confidence threshold
confidence_threshold = 0

# Function to make predictions
def classify_image(frame):
    # Resize the frame to match the input size of the model
    resized_frame = cv2.resize(frame, (224, 224))
    image = img_to_array(resized_frame) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    (_, classes) = model.predict(image)
    class_idx = np.argmax(classes)
    confidence = classes[0][class_idx]

    # Display the class label if confidence is above the threshold
    if confidence > confidence_threshold:
        class_label = lb.classes_[class_idx]
        result = f"Detected: {class_label} (Confidence: {confidence:.2f})"
    else:
        result = "No object detected"

    return result

# Gradio interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(sources="webcam", streaming=True),
    outputs="text",
    live=True,
)

# Launch the Gradio interface
iface.launch()