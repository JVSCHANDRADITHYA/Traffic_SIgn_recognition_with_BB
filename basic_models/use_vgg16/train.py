import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten


vgg16_obj = VGG16(include_top = False, input_shape = (224,224,3))
for layer in vgg16_obj.layers:             
    layer.trainable = False
vgg16_obj.summary()

f1=Flatten()(vgg16_obj.output)
final_layer=Dense(58,activation='Softmax')(f1)
final_layer

model=Model(inputs=vgg16_obj.input,outputs=final_layer)
model.summary()

data_gen = ImageDataGenerator(zoom_range=0.5, shear_range=0.8, horizontal_flip=True, rescale=1/255)

path='F:\\use_vgg16\\dataset\\traffic_Data\\DATA'
data_gen =data_gen.flow_from_directory(
    directory=path,
    target_size=(224,224),
    batch_size=3,
    class_mode="categorical",
    )

model.compile(loss='categorical_crossentropy', metrics=['accuracy']) 
model.fit_generator(data_gen, epochs=10)

model.save('Traffic_Signal_Vgg16_10epoch_96%.h5')

