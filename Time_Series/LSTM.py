import numpy as np
import sklearn
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import tensorflow
import pandas as pd
# from tensorflow.keras.utils.generic_utils import get_custom_objects
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Activation
import matplotlib.pyplot as plt
from datetime import timedelta
# def custom_activation(x, beta = 1):
#         return (K.sigmoid(beta * x) * x)


def create_dataset(dataset,look_back):
    dataX,dataY= [],[]
    print("create",dataset)
    for i in range(len(dataset)-look_back-1):
        a=dataset[i:i+look_back,0]
        dataX.append(a)
        dataY.append(dataset[i+look_back,0])
    print("val",np.array(dataX).shape)
    return np.array(dataX),np.array(dataY)
#print(create_dataset([1,2,3,4,5],2))

def future_values(df,model,scaler,look_back,column,test):
    future=50
    next_day=[]
    #df=df.iloc[::-1]
    for i in range(future):
        ws=test[-look_back:]
        ws=np.reshape(ws,(1,1,look_back))
        pred=model.predict(ws).reshape(-1,1)
        print(pred)
        next_day.append(pred)
        test=np.append(test,np.array(pred))


    test=np.asanyarray([scaler.inverse_transform([[i]]) for i in test])
    print(test)
    df_past=df[['timestamp',f'{column}']]
    df_past.rename(columns={f'{column}':f'{column}_ACTUAL'},inplace=True)
    df_past['timestamp']=pd.to_datetime(df_past['timestamp'])
    df_past[f'{column}_FORECAST']=np.nan
    df_past[f'{column}_FORECAST'].iloc[-1]=df_past[f'{column}_ACTUAL'].iloc[-1]

    df_future=pd.DataFrame(columns=['timestamp',f'{column}_ACTUAL',f'{column}_FORECAST'])
    df_future['timestamp']=df_past['timestamp']
    df_future[f'{column}_ACTUAL']=df_past[f'{column}_ACTUAL']
    l=[np.nan for i in range(len(df_past)-50)]
    forecast=np.append(np.asanyarray(l),test[-50:].flatten())
    # l=[np.nan for i in range(50)]
    # forecast=np.append(forecast,np.asanyarray(l,dtype=np.float))
    #df_future[f'{column}_FORECAST']=forecast
    df_past[f'{column}_FORECAST']=forecast
    print(df_past.columns)

    #df_past=df_past.sort_values(by="timestamp",ascending=False)
    df_past.to_csv("df_past.csv")
    plt.figure(figsize=(10,6))
    df_past=df_past.sort_values(by="timestamp",ascending=True)
    plt.plot(df_past['timestamp'],df_past[f'{column}_ACTUAL'],color="blue")
    plt.plot(df_past['timestamp'],df_past[f'{column}_FORECAST'],color="red")
    plt.legend(["Actual","Forecast"])
    plt.savefig("predictions.png")
    # plot=df_past[[f"{column}_ACTUAL",f"{column}_FORECAST"]].plot(title="predictions")
    # fig=plot.get_figure()
    # fig.savefig('predictions.png')
def future_forecasts(df,model,scaler,look_back,column,test):
    future=500
    next_day=[]

    for i in range(future):
        ws=test[-look_back:]
        ws=np.reshape(ws,(1,1,look_back))
        pred=model.predict(ws).reshape(-1,1)
        print(pred)
        next_day.append(pred)
        test=np.append(test,np.array(pred))


    test=np.asanyarray([scaler.inverse_transform([[i]]) for i in test])
    print(test)
    df_past=df[['timestamp',f'{column}']]
    df_past.rename(columns={f'{column}':f'{column}_ACTUAL'},inplace=True)
    df_past['timestamp']=pd.to_datetime(df_past['timestamp'])
    df_past[f'{column}_FORECAST']=np.nan

    df_past=df_past.sort_values(by="timestamp",ascending=True)
    l=[]
    for i in range(1,501):
        l.append(df_past['timestamp'].iloc[-1]+timedelta(days=i))


    df_future=pd.DataFrame(columns=['timestamp',f'{column}_ACTUAL',f'{column}_FORECAST'])
    df_future['timestamp']=l
    df_future[f'{column}_ACTUAL']=np.nan
    df_future[f'{column}_FORECAST']=test[-500:].flatten()
    df_future.to_csv("future.csv")
    #results=df_past.append(df_future).set_index('timestamp')

    #df_past=df_past.sort_values(by="timestamp",ascending=False)
    #df_past.to_csv("df_past.csv")
    plt.figure(figsize=(10,6))

    plt.plot(df_past['timestamp'],df_past[f'{column}_ACTUAL'],color="blue")
    plt.plot(df_future['timestamp'],df_future[f'{column}_FORECAST'],color="red")
    plt.legend(["Actual","Forecast"])
    plt.savefig("Forecasts.png")
    # plot=df_past[[f"{column}_ACTUAL",f"{column}_FORECAST"]].plot(title="predictions")
    # fig=plot.get_figure()
    # fig.savefig('predictions.png')
def transferred_lstm(trained_model,scaler,new_data):
    dataset=scaler.transform(new_data)

    look_back=3
    train_size=int(len(dataset))*0.67
    train,test=dataset[0:train_size],dataset[train_size:len(dataset)]
    trainX,trainY=create_dataset(train,look_back)
    testX,testY=create_dataset(test,look_back)
    model=tensorflow.keras.models.Sequential()
    model.add(tensorflow.keras.layers.LSTM(4,input_shape=(1,look_back)))
    model.add(tensorflow.keras.layers.ReLU())
    model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.LSTM(4,input_shape=(1,look_back)))
    model.add(tensorflow.keras.layers.ReLU())
    model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.Dense(1))
    get_custom_objects().update({'custom_activation': Activation(custom_activation)})
    model.add(Activation(custom_activation,name = "Swish"))
    model.compile(loss=tensorflow.losses.MeanSquaredError(),optimizer=tensorflow.optimizers.Adam())
    model.set_weights(weights=trained_model.get_weights())
    history=model.fit(trainX,trainY,epochs=40,shuffle=False,callbacks=[])


def lstm(df,dataset,column):
    look_back=100
    scaler=RobustScaler()
    dataset=scaler.fit_transform(dataset.values)
    print("dataset",dataset)

    train_size=int(len(dataset)*0.67)
    print("Train_size",train_size)
    print(dataset[0:train_size])
    train,test=dataset[0:train_size,:],dataset[train_size:len(dataset),:]
    trainX,trainY=create_dataset(train,look_back)
    testX,testY=create_dataset(test,look_back)
    trainX=np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
    testX=np.reshape(testX,(testX.shape[0],1,testX.shape[1]))

    print("this",trainX,trainY)
    model=tensorflow.keras.models.Sequential()
    # model.add(tensorflow.keras.layers.LSTM(16,input_shape=(1,look_back),activation="relu",return_sequences=True))
    # model.add(tensorflow.keras.layers.Dropout(0.2))
    # model.add(tensorflow.keras.layers.Bidirectional(tensorflow.keras.layers.LSTM(4,input_shape=(1,look_back),return_sequences=True)))
    # model.add(tensorflow.keras.layers.ReLU())
    # model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.LSTM(64,input_shape=(1,look_back),activation="relu",return_sequences=True))
    model.add(tensorflow.keras.layers.Dropout(0.2))
    # model.add(tensorflow.keras.layers.LSTM(64,input_shape=(1,look_back),activation="relu",return_sequences=False))
    # model.add(tensorflow.keras.layers.Dropout(0.2))
    model.add(tensorflow.keras.layers.Dense(1,activation=tensorflow.keras.activations.swish))

    model.compile(loss=tensorflow.losses.MeanSquaredError(),optimizer=tensorflow.optimizers.Adam(),metrics=[tensorflow.metrics.MeanSquaredError()])
    history=model.fit(trainX,trainY,epochs=10,batch_size=10,validation_data=(testX,testY))
    plt.figure(figsize=(10,6))
    x=np.arange(1,len(history.history['mean_squared_error'])+1,1)
    plt.plot(x,history.history['mean_squared_error'],color="blue")
    plt.plot(x,history.history['val_mean_squared_error'],color="red")
    plt.legend(["mean_squared_error","val_mean_squared_error"])
    plt.savefig("accuracy.png")
    print("TrainX",trainX)
    train_predict=model.predict(trainX)
    test_predict=model.predict(testX)
    print("train_predict",train_predict)
    train_predict=scaler.inverse_transform(train_predict[0])
    trainY=scaler.inverse_transform([trainY])
    test_predict=scaler.inverse_transform(test_predict[0])
    testY=scaler.inverse_transform([testY])
    trainPredictPlot=np.empty_like(dataset)
    trainPredictPlot[:,:]=np.nan
    trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
    testPredictPlot=np.empty_like(dataset)
    testPredictPlot[:,:]=np.nan
    testPredictPlot[len(train_predict)+(look_back*2)+1:len(dataset)-1,:]=test_predict
    plt.figure(figsize=(10,6))
    plt.plot(scaler.inverse_transform(dataset))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.savefig("train_test_predict.png")
    df_past=df[['timestamp',f'{column}']]
    results=future_values(df,model,scaler,look_back,f'{column}',test)
    future_forecasts(df,model,scaler,look_back,f'{column}',df[f'{column}'].values)

'''
https://discuss.tensorflow.org/t/keras-nn-shows-0-accuracy/11106
metrics=['accuracy'] is for a classification problem 
'''


if __name__=='__main__':
    df=pd.read_csv("HistoricalQuotes.csv")
    df=df.iloc[::-1]
    df.rename(columns={'Date':'timestamp'},inplace=True)
    print(df.columns)
    print(df[' Close/Last'].values[-1])

    lstm(df,df[[' Close/Last']],' Close/Last')
