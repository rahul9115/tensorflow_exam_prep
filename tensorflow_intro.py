import tensorflow
import numpy as np
model=tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(units=1,input_shape=[1]))
model.add(tensorflow.keras.layers.Dense(units=1,input_shape=[1]))
'''
optimizer for each epoch checks other values and reduces the training loss
by updating the weights and biases 
'''
model.compile(optimizer='sgd',loss='mean_squared_error')


x=np.array([-1.0,0.0,1.0,2.0,3.0])
y=np.array([-3.0,-1.0,1.0,3.0,5.0])
#Epochs are the training iterations
model.fit(x,y,epochs=20)
print(model.predict([10.0]))

'''
The tensorflow predicts a number closer to 19 but not exactly 19 since the data is linear
and there is no enough points to train on so there is a high-probability it will be 19
but the neural network is not positive
'''
