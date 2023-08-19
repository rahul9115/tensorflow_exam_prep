import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,load_img
from tensorflow.keras.applications.inception_v3 import InceptionV3

local_zip='./horse-or-human.zip'
zip_ref=zipfile.ZipFile(local_zip,'r')
zip_ref.extractall('/training')

val_local_zip='./validation-horse-or-human.zip'
zip_ref=zipfile.ZipFile(val_local_zip,'r')
zip_ref.extractall('/validation')

train_dir='/training'
validation_dir='/validation'

train_horses_dir=os.path.join(train_dir,'horses')
train_humans_dir=os.path.join(train_dir,'humans')

validation_horses_dir=os.path.join(validation_dir,'horses')
validation_humans_dir=os.path.join(validation_dir,'humans')

train_datagen=ImageDataGenerator(1.0/255)
training_generator=train_datagen.flow_from_directory(directory=train_dir,
                                  class_mode="binary",
                                  batch_size=32,
                                  target_size=(150,150))

validation_datagen=ImageDataGenerator(1.0/255)
validation_generator=train_datagen.flow_from_directory(directory=validation_dir,
                                  class_mode="binary",
                                  batch_size=32,
                                  target_size=(150,150))

local_weights='inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model=InceptionV3(input_shape=(150,150,3),
                              include_top=False,
                              weights=None)

pre_trained_model.load_weights(local_weights)

for layer in pre_trained_model.layers:
    layer.trainable=False

last_desired_layer=pre_trained_model.get_layer("mixed7")
last_output=last_desired_layer.output

x=layers.Flatten()(last_output)
x=layers.Dense(1024,activation="relu")(x)
x=layers.Dense(1,activation="sigmoid")(x)

model=Model(pre_trained_model.input,x)
model.compile(optimizer=RMSprop(learning_rate=0.0001),loss="binary_crossentropy",metrics=["accuracy"])

history=model.fit(training_generator,
          validation_data=validation_generator,
          epochs=2,
          verbose=2)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()


horses=os.path.join(validation_dir,"horses")

for i in os.listdir(horses):
    path=os.path.join(horses,i)
    img=load_img(path,target_size=(150,150,3))
    x=img_to_array(img)
    x/=255
    x=np.expand_dims(x,axis=0)
    print("shape",x.shape)
    images=np.vstack([x])

    classes=model.predict(images,batch_size=32)
    if classes[0]>0.5:
        print("It is a human")
    else:
        print("It's a horse")







