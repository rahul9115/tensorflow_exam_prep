import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


(train_images,train_labels),(test_images,test_labels)=tf.keras.datasets.mnist.load_data()
class myCallbacks(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['accuracy']>=0.99:
            print("Reached 99% accuracy so cancelling training!")
            self.model.stop_training=True
train_images,test_images=train_images/255,test_images/255
callbacks=myCallbacks()
'''
This is where Keras flatten comes to save us. 
This function converts the multi-dimensional 
arrays into flattened one-dimensional arrays 
or single-dimensional arrays. It takes all 
the elements in the original tensor (multi-dimensional array) 
and puts them into a single-dimensional array.
'''
model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(256,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)


])

model.compile(optimizer=tf.optimizers.Adam(),loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=10,callbacks=[callbacks])
print(model.evaluate(test_images,test_labels))
classifications=model.predict(test_images)
# to get the position of highest probablity in 0 to 9 position
val=np.unravel_index(classifications[0].argmax(), classifications.shape)
print(val)
print(test_labels[0])
