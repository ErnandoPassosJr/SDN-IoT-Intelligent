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
    WINDOW = 35

    for i in np.arange(df.shape[0]):    
        init = i*WINDOW
        init2 = (i+1)*WINDOW
        if(init2<df.shape[0]):
            df.iloc[init:init+WINDOW,2] =  df.iloc[init2,2]
    df2 = normalizing(df,path)
    #train_size = int(len(df2) * 0.95)
    #return df2.iloc[0:train_size], df2.iloc[train_size:len(df2)]
    return df2


def normalizing(dataset,path):
    df_norm = pd.read_csv(path, delimiter=",")
    df_norm = df_norm.iloc[:,1:4]
    scaler = StandardScaler().fit(df_norm)

    scaler = scaler.fit(df_norm[['delay']])

    dataset['delay'] = scaler.transform(dataset[['delay']])
    return dataset


def unormalizing(y_pred):
    df_norm = pd.read_csv(path, delimiter=",")
    df_norm = df_norm.iloc[:,1:4]
    scaler = StandardScaler().fit(df_norm)
    scaler = scaler.fit(df_norm[['delay']])
    #y_test_inv = scaler.inverse_transform(Y_test.reshape(1,-1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    
    return y_pred_inv

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
    


def predict(path):
    train, test = preprocessiing(path)
    model = keras.models.load_model('lstm.h5')
    X_test,Y_test = create_dataset(test, test.delay)
    
    y_pred = model.predict(X_test)
    
    y_test_inv, y_pred_inv = unormalizing(Y_test, y_pred,path)
   
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


def predict3(window):
	model = keras.models.load_model('models/lstmv3.h5')

	window=np.array(window)

	df = pd.DataFrame(window,columns=['delay'])

	
	df2=normalizing(df)
	
	nparray=np.array(df2)
	nparray=nparray.reshape(len(nparray),1,1)


	y_pred = model.predict(nparray)
	y_pred_inv=unormalizing(y_pred)
	print(np.mean(y_pred_inv))



