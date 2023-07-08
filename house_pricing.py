import tensorflow
import numpy as np

model=tensorflow.keras.models.Sequential()
model.add(tensorflow.keras.layers.Dense(units=1,input_shape=[1]))
model.add(tensorflow.keras.layers.Dense(units=1,input_shape=[1]))
model.compile(optimizer="sgd",loss="mean_squared_error")
x=np.arange(1,11,dtype='float')
y=np.arange(1.5,6.5,0.5,dtype='float')
model.fit(x,y,epochs=1000)
print(model.predict([11]))
#1.5+(n-1)*0.5





