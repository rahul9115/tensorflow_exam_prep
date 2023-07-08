import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch,logs={}):
      print(logs.get('accuracy'))
      if (logs.get('accuracy')>0.9):
          print("Reached 90% accuracy so cancelling training!")
          self.model.stop_training=True
print(tf.__version__)
fashion_mnist=tf.keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()\
#Normalize the images

train_images=train_images/255
test_images=test_images/255

callbacks=myCallback()

model=tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=100,callbacks=[callbacks])
predicted_labels=model.predict(test_images)
print(model.evaluate(test_images,test_labels))

'''
This is less a question about TensorFlow and more a general question about activation functions in neural network layers. 
Let’s break down what each function actually does.
1. ReLU stands for Rectified Linear Unit, and is represented by the function,
   ReLU(x)=max(0,x)
   Yes, as mentioned by others, ReLU is an excellent hidden layer activation choice because 
   it doesn’t suffer the weaknesses of other common hidden layer activations like sigmoid and tanh 
   such as the vanishing gradient problem. However, if you think of the function qualitatively, 
   ReLU is simply controlling input neuron values, only allowing “pertinent” information to be retained 
   and propagated forward to the next layers. “Pertinent” may not be the best word - 
   in reality it’s simply values >= 0. I choose that word because of the biological implications
   behind this function (which you can read about, R Hahnloser, R. Sarpeshkar, M A Mahowald, R. J. Douglas, H.S. Seung (2000). Digital selection and analogue amplification coexist in a cortex-inspired silicon circuit)
   .In a nutshell,ReLU is used for filtering which information is propagating forward through the network.
2. Softmax (AKA normalized exponential function) on the other hand is quite different - 
   given a set of N inputs, Softmax will output a probability distribution over each of the N inputs.
   This is why we typically used a softmax activation in the last layer of our classifier networks - 
   we want a range of probabilities which represents which “class” our input data belongs to. 
   So, the input data is propagated through the network, and finally passed through a softmax layer 
   which will give us that coveted probability distribution.Another key characteristic about this 
   output distribution is that all the values add up to 1.
'''


