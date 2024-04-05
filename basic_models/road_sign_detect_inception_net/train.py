import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential, Model

import os
for dirname, _, filenames in os.walk('F:\\road_sign_detect\\dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


labels_path = 'F:\\road_sign_detect\\dataset\\labels.csv'
classes = pd.read_csv(labels_path)
class_names = list(classes['Name'])
print(class_names)

data_dir = 'F:\\road_sign_detect\\dataset\\traffic_Data\\DATA'

# Load the class labels
class_labels = pd.read_csv(labels_path)
class_names = list(class_labels['Name'])
num_classes = len(class_names)

# Data Preprocessing and Augmentation
batch_size = 32
image_size = (75, 75)  # Minimum input size required by InceptionV3

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest',
    preprocessing_function=lambda x: tf.image.resize(x, image_size)
)

# Split the dataset into training and validation sets
train_datagen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

validation_datagen = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False,
    subset='validation' 
)


sample_images, sample_labels = next(train_datagen)

plt.figure(figsize=(12, 8))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(sample_images[i])
    plt.title(class_names[np.argmax(sample_labels[i])])
    plt.axis('off')

plt.show() # show the sample images


from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# Define input shape
input_shape = (75, 75, 3)  # Minimum input size required by InceptionV3

# Create an instance of InceptionV3 without the top (fully connected) layers
base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)

# Replace AveragePooling2D with GlobalAveragePooling2D
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Global Average Pooling instead of AveragePooling2D
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Combine the base model and the custom classification layers
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Define callbacks model checkpointing
model_checkpoint = ModelCheckpoint('traffic_signal_model.h5', save_best_only=True)


##### TRAIN THE MODEL

epochs = 100
history = model.fit(
    train_datagen,
    steps_per_epoch=len(train_datagen),
    validation_data=validation_datagen,
    validation_steps=len(validation_datagen),
    epochs=epochs,
    callbacks=[model_checkpoint],
    verbose=1
)

# Save the model
model.save('traffic_signal_model_final.h5')

