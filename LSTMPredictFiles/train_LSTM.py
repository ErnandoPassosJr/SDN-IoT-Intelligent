# -*- coding: utf-8 -*-
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import datetime

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
from IPython.display import clear_output

from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score

def preprocessiing(path):
   
    prep_dataset1 = pd.read_csv(path, delimiter=",")
    df = prep_dataset1.iloc[:,1:4]
    #print(df)
    WINDOW = 35

    for i in np.arange(df.shape[0]):    
        init = i*WINDOW
        init2 = (i+1)*WINDOW
        if(init2<df.shape[0]):
            df.iloc[init:init+WINDOW,2] =  df.iloc[init2,2]
    df2 = normalizing(df,path)
    train_size = int(len(df2) * 0.95)
    return df2.iloc[0:train_size], df2.iloc[train_size:len(df2)]


def normalizing(dataset,path):
    df_norm = pd.read_csv(path, delimiter=",")
    df_norm = df_norm.iloc[:,1:4]
    scaler = StandardScaler().fit(df_norm)

    scaler = scaler.fit(df_norm[['delay']])

    dataset['delay'] = scaler.transform(dataset[['delay']])
    return dataset


def unormalizing(Y_test,y_pred,path):
    df_norm = pd.read_csv(path, delimiter=",")
    df_norm = df_norm.iloc[:,1:4]
    scaler = StandardScaler().fit(df_norm)
    scaler = scaler.fit(df_norm[['delay']])
    y_test_inv = scaler.inverse_transform(Y_test.reshape(1,-1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    return y_test_inv, y_pred_inv

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []    
    start = timer()
    for i in range(len(X) - time_steps):
        clear_output(wait=True)
        print('modeling to keras ',round((i/(len(X) - time_steps))*100,2), ('%'), end='')
        s = round(timer() - start)
        if(s>60):
            s /=60
            print(' ', s, ' seconds')
        v = X.iloc[i: (i+time_steps), 2:3].to_numpy()
        Xs.append(v)
        ys.append(y.iloc[i+time_steps])
    return np.array(Xs), np.array(ys)

def LSTMconf(X_train):
    print('Init config LSTM')
    model = keras.Sequential()
    model.add(
            keras.layers.LSTM(
                units=512,
                input_shape=(X_train.shape[1],X_train.shape[2]),
                 kernel_initializer="glorot_uniform",
                unit_forget_bias=True,
                recurrent_dropout=0.75,
            )
        )
        
    model.add(keras.layers.Dense(units=512, ))
    model.add(keras.layers.Dense(units=512, ))
    model.add(keras.layers.Dense(units=512, ))
    model.add(keras.layers.Dropout(rate=0.75))
    model.add(keras.layers.Dense(units=1))
    
    loss ="mse"
    optim = tf.keras.optimizers.Adam(
    learning_rate=0.0001)
    

    model.compile(loss=loss, optimizer=optim, 
             )
    
    
    return model



def checkFileExistance(filePath):
    return None
    try:
        with open(filePath, 'r') as f:
            return True
    except FileNotFoundError as e:
        return False
    except IOError as e:
        return False

def best_value(history):
	minin = history.history['loss'][0]
	pos = 0
	for i in range(len(history.history['loss'])):
	   
		for j in range(len(history.history['val_loss'])):
			lossDif = abs(history.history['loss'][i] - history.history['val_loss'][j])
			if(lossDif<minin):
				minin = lossDif
				pos = i
				
	print(minin, ' - ', pos)
    
def LSTMfit():
    
    start = timer()
    train, test = preprocessiing()
    
    X_train,Y_train = create_dataset(train, train.delay)
    model = LSTMconf(X_train)
    
    if(checkFileExistance('lstm.h5')):
        model = keras.models.load_model('lstm.h5')
    else:
        model = LSTMconf(X_train)
        
    print('Init Train')
    batch_size = round(X_train.shape[0]*0.08)
    history = model.fit(
        X_train, Y_train, 
        epochs=128, 
        batch_size= batch_size,
        validation_split=0.1,
        shuffle=False,
    )
    
    
    print('Saving Model')
    model.save('lstm.h5')
    #best_value(history)
    return history

def predict():
    train, test = preprocessiing()
    model = keras.models.load_model('lstm.h5')
    X_test,Y_test = create_dataset(test, test.delay)
    
    y_pred = model.predict(X_test)
    
    y_test_inv, y_pred_inv = unormalizing(Y_test, y_pred)
   
    size = np.min([y_pred_inv.shape[0],y_test_inv.shape[0] ])
    rmse =  mean_squared_error(y_test_inv.flatten()[0:size], y_pred_inv.flatten()[0:size])
    mae =  mean_absolute_error(y_test_inv.flatten()[0:size], y_pred_inv.flatten()[0:size])
    median_mae = median_absolute_error(y_test_inv.flatten()[0:size], y_pred_inv.flatten()[0:size])
    evs = explained_variance_score(y_test_inv.flatten()[0:size], y_pred_inv.flatten()[0:size])
    
    print()
    print("RMSE",rmse)
    print("MAE",mae)
    print("MEDIAN MAE",median_mae)
    print('Explained Variance Score: ',evs)
    return y_pred


