import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model and label binarizer
model = load_model(r"F:\TSR_Model\final_model\resources\model_bbox_regression_and_classification.h5")
lb = pickle.loads(open(r"F:\TSR_Model\final_model\resources\lb.pickle", "rb").read())

# Confidence threshold
confidence_threshold = 0.6

# Open the webcam
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()

    # Resize the frame to match the input size of the model
    resized_frame = cv2.resize(frame, (224, 224))
    image = img_to_array(resized_frame) / 255.0
    image = np.expand_dims(image, axis=0)

    # Make predictions
    (boxes, classes) = model.predict(image)
    (startX, startY, endX, endY) = (boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3])

    # Convert class index to class label
    class_idx = np.argmax(classes)
    confidence = classes[0][class_idx]

    # Draw bounding box and class label on the frame if confidence is above the threshold
    if confidence > confidence_threshold:
        class_label = lb.classes_[class_idx]
        cv2.rectangle(frame, (int(startX * frame.shape[1]), int(startY * frame.shape[0])),
                      (int(endX * frame.shape[1]), int(endY * frame.shape[0])), (0, 255, 0), 2)
        y = int(startY * frame.shape[0]) - 10 if int(startY * frame.shape[0]) - 10 > 10 else int(
            startY * frame.shape[0]) + 10
        label_text = f"{class_label}: {confidence:.2f}"
        cv2.putText(frame, label_text, (int(startX * frame.shape[1]), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("Live Webcam Feed", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
