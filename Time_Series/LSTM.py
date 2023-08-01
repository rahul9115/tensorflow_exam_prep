import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K
from keras.layers import Activation

def custom_activation(x, beta = 1):
        return (K.sigmoid(beta * x) * x)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})
def create_dataset(dataset,look_back):
    dataX,dataY= [],[]
    for i in range(len(dataset)-look_back-1):
        a=dataset[i:i+look_back]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return dataX,dataY
print(create_dataset([1,2,3,4,5],2))

def future_values(df,test,model,scaler,look_back):
    future=50
    next_day=[]

    for i in range(future):
        ws=test[look_back:]
        ws=np.reshape(ws,(1,1,look_back))
        pred=model.predict(ws).reshape(-1,-1)
        next_day.append(pred)
        val=list(test)
        val.append(np.array(pred))
        test=np.asanyarray(val,dtype=np.float)
    test=np.asanyarray([scaler.inverse_transform[i] for i in test],dtype=np.float)
    return test

def transferred_lstm(trained_model,scaler,new_data):
    dataset=scaler.transform(new_data)

    look_back=3
    train_size=int(len(dataset))*0.67)
    train,test=dataset[0:train_size],dataset[train_size:len(dataset)]
    trainX,trainY=create_dataset(train,look_back)
    testX,testY=create_dataset(test,look_back)
    model=tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.LSTM(64,input_shape=(1,look_back),return_sequences=True))
    model.add(tensorflow.keras.layers.ReLU())
    model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.LSTM(64,input_shape=(1,look_back),return_sequences=False))
    model.add(tensorflow.keras.layers.ReLU())
    model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.Dense(1))
    model.add(Activation(custom_activation,name = "Swish"))
    model.compile(loss=tensorflow.losses.MeanSquaredError(),optimizer=tensorflow.optimizers.Adam(learning_rate=,epsilon=))
    model.set_weights(weights=trained_model.get_weights())
    history=model.fit(trainX,trainY,epochs=40,shuffle=False,callbacks=[])


def lstm(dataset):
    look_back=3
    scaler=MinMaxScaler(feature_range=(0,1))
    dataset=scaler.fit_transform(dataset)
    train_size=int(len(dataset))*0.67)
    train,test=dataset[0:train_size],dataset[train_size:len(dataset)]
    trainX,trainY=create_dataset(train,look_back)
    testX,testY=create_dataset(test,look_back)
    model=tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.LSTM(64,input_shape=(1,look_back),return_sequences=True))
    model.add(tensorflow.keras.layers.ReLU())
    model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.LSTM(64,input_shape=(1,look_back),return_sequences=False))
    model.add(tensorflow.keras.layers.ReLU())
    model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.Densse(1))
    model.add(Activation(custom_activation,name = "Swish"))
    model.compile(loss=tensorflow.losses.MeanSquaredError(),optimizer=tensorflow.optimizers.Adam(learning_rate=,epsilon=))
    history=model.fit(trainX,trainY,epochs=40,shuffle=False,callbacks=[])



