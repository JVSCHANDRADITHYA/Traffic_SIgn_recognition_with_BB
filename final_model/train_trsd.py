import sys
import os
import cv2
import datetime
import numpy as np
from pathlib import Path
import random
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import VGG16
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
import pickle

import time

import matplotlib.patches as patches

# Function to plot random images with bounding boxes
def plot_random_images_with_boxes(images, bboxes, labels, lb, num_images=10):
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))
    fig.subplots_adjust(hspace=0.5)
    axes = axes.flatten()

    for i in range(num_images):
        index = random.randint(0, len(images) - 1)
        image = images[index]
        bbox = bboxes[index]
        label = labels[index]

        # Plot image
        axes[i].imshow(image)
        axes[i].axis('off')

        # Convert one-hot encoded label back to string
        class_label = lb.classes_[np.argmax(label)]

        # Add bounding box
        rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1],
                                 linewidth=2, edgecolor='r', facecolor='none')
        axes[i].add_patch(rect)

        axes[i].set_title(f"{class_label}\nBox: {bbox}")

    plt.show()

# Load annotations from the text file
annotations_file = Path(r"F:\TSR_Model\final_model\TRSD_dataset\TSRD-Train Annotation\TsignRecgTrain4170Annotation.txt")

data = []
labels = []
bboxes = []
imagePaths = []

with open(annotations_file, "r") as f:
    lines = f.readlines()

for line in lines:
    line = line.strip().split(';')
    imname = line[0]
    xmin_1, ymin_1, xmax_1, ymax_1 = map(int, line[3:7])
    label = int(line[7])
    
    # Assuming image paths are in a separate file, load them
    impath = Path(r"F:\final_model\TRSD_dataset\tsrd-train_img") / imname  # Adjust the path as needed
    image = load_img(impath, target_size=(224, 224))
    image = img_to_array(image)
    
    target_size=(224, 224)
    scale_x = target_size[0] / image.shape[1]
    scale_y = target_size[1] / image.shape[0]
    
    xmin = xmin_1 * scale_x
    ymin = ymin_1 * scale_y
    xmax = xmax_1 * scale_x
    ymax = ymax_1 * scale_y

    data.append(image)
    labels.append(label)
    bboxes.append((xmin, ymin, xmax, ymax))
    imagePaths.append(impath)

data = np.array(data, dtype="float32") / 255.0

labels = np.array(labels)
bboxes = np.array(bboxes, dtype="float32")
imagePaths = np.array(imagePaths)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)

split = train_test_split(data,
                         labels,
                         bboxes,
                         imagePaths,
                         test_size=0.20,
                         random_state=12)

(trainImages, testImages) = split[:2]
(trainLabels, testLabels) = split[2:4]
(trainBBoxes, testBBoxes) = split[4:6]
(trainPaths,  testPaths)  = split[6:]

# Visualize random images with bounding boxes before training
plot_random_images_with_boxes(trainImages, trainBBoxes, trainLabels, lb)

vgg = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Freeze training any of the layers of VGGNet
vgg.trainable = False

# Preprocess input images using VGG preprocessing function
trainImages = tf.keras.applications.vgg16.preprocess_input(trainImages)
testImages = tf.keras.applications.vgg16.preprocess_input(testImages)

# Max-pooling is output of VGG, flattening it further
flatten = vgg.output
flatten = Flatten()(flatten)

bboxHead = Dense(128, activation="relu")(flatten)
bboxHead = Dense(64, activation="relu")(bboxHead)
bboxHead = Dense(32, activation="relu")(bboxHead)
bboxHead = Dense(4, activation="sigmoid", name="bounding_box")(bboxHead)

softmaxHead = Dense(512, activation="relu")(flatten)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(512, activation="relu")(softmaxHead)
softmaxHead = Dropout(0.5)(softmaxHead)
softmaxHead = Dense(len(lb.classes_), activation="softmax", name="class_label")(softmaxHead)

model = Model(inputs=vgg.input, outputs={"bounding_box": bboxHead, "class_label": softmaxHead})

INIT_LR = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 16
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7)
lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5)

losses = {
    "class_label": "categorical_crossentropy",
    "bounding_box": "huber_loss",  # Change to huber_loss for bounding box regression
}

lossWeights = {
    "class_label": 1.0,
    "bounding_box": 1.0
}

trainTargets = {
    "class_label": trainLabels,
    "bounding_box": trainBBoxes
}

testTargets = {
    "class_label": testLabels,
    "bounding_box": testBBoxes
}

opt = Adam(INIT_LR)

model.compile(loss=losses, optimizer=opt, metrics=["accuracy"], loss_weights=lossWeights)

print(model.summary())

H = model.fit(
    trainImages, trainTargets,
    validation_data=(testImages, testTargets),
    batch_size=BATCH_SIZE,
    epochs=NUM_EPOCHS,
    callbacks=[early_stopping, lr],
    verbose=1)

model.save("./model_bbox_regression_and_classification.h5")

# Save the label binarizer
with open("lb.pickle", "wb") as f:
    f.write(pickle.dumps(lb))

lossNames = ["loss",
             "class_label_loss",
             "bounding_box_loss"]
