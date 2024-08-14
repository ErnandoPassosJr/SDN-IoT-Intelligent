import time
import threading
from threading import Thread
import sys
sys.path.insert(0, '/home/openflow/predict')
sys.path.insert(0, '/home/mininet/projeto_ml/FoT-Stream_Simulation/FoTStreamServer/tsDeep')
import paho.mqtt.client as mqtt
from reg import utils_hosts 
import timeit
import os
import random
import numpy as np
import json
import series
import pandas as pd

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
import datetime


from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
from IPython.display import clear_output

from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score


PortaBroker = 1883
KeepAliveBroker = 60
l_sensors=[]
break_all=False

class Preditive_obj(Thread):
	global publish_time, current_time, name_device, name_gateway, init_time, gateway_ip, sensor_ip, to_install
	def __init__ (self):
		self.publish_time=0
		self.predict_time=0.0
		self.current_time=0.0
		self.init_time=0.0
		self.name_device=''
		self.gateway_ip=''
		self.sensor_ip=''
		self.name_gateway=''
		self.to_install=True

def thread_flow(name,value):
	global l_sensors, break_all
	
	def is_flow(gateway_ip,sensor_ip,data):
		for i in range(0,len(data)):
			if(data[i].ip==sensor_ip and data[i].gateway==gateway_ip):
				return True
		return False
	
	def remove_flow(gateway_ip,sensor_ip):
		for i in range(0,len(l_sensors)):
			if(l_sensors[i].gateway_ip==gateway_ip and l_sensors[i].sensor_ip==sensor_ip):
				ob=l_sensors[i]
				l_sensors.remove(ob)
		
	
	
	while True:
		time.sleep(0.2)
		try:
			for i in range(0,len(l_sensors)):
				if(float(l_sensors[i].current_time)>=(float(l_sensors[i].predict_time)*0.85)):	
					print("**** Installing Flow Device:",l_sensors[i].name_device, "Predict time:", l_sensors[i].predict_time,"Current time:",l_sensors[i].current_time)
					#os.system('python install_flow.py -g '+l_sensors[i].gateway_ip+' -s '+l_sensors[i].sensor_ip+' &')
					remove_flow(l_sensors[i].gateway_ip,l_sensors[i].sensor_ip)
					break
				l_sensors[i].current_time=timeit.default_timer()-l_sensors[i].init_time+0.0057
		except IndexError:
			continue
		except ValueError:
			print ("ValueError Resolved")
		#	continue
		except ValueError:
			continue

topic = ""
group = ""
Sensorid = ""
DeviceId = ""
last_concept = time.time()
last_delay = 0
init = True	
#THREAD
def thread(name,gateway):
	
	##INIT THREAD FUNCTIONS
	def contains_name(name_device):
		global l_sensors
		for i in range(0,len(l_sensors)):
			if(l_sensors[i].name_device==name_device and l_sensors[i].name_gateway==name):
				return True
		return False
	
	def print_list(name):
		global l_sensors
		for i in range(0,len(l_sensors)):
			if(l_sensors[i].name_device==name):
				print ("Installing "+l_sensors[i].name_device+" Publish "+str(l_sensors[i].publish_time)+" Name gateway "+l_sensors[i].name_gateway+" Current Time "+str(l_sensors[i].current_time))
	
	def message_to_publish(msg):
		msg=msg.rsplit('\"time\": ')[1].rsplit('}')[0]+'}'
		obj=utils_hosts.to_object(msg)
		return obj.publish
	
	def modifi_time_publish(msg,name_device):
		global l_sensors
		for i in range(0,len(l_sensors)):
			if(l_sensors[i].name_device==name_device):
				msg=msg.rsplit('FLOW INFO temperatureSensor ')[1]
				msg=msg.replace("collect","\"collect\"")
				msg=msg.replace("publish","\"publish\"")
				obj=utils_hosts.to_object(msg)
				ob=l_sensors[i]
				l_sensors.remove(ob)
				#l_sensors[i].publish_time=obj.publish
				break
				
	
	def catch_message(topic,message):
		global l_sensors
		if(topic.find('dev/')!=-1 and message.find("\"METHOD\":\"FLOW\"")!=-1):
			name_device=topic.replace('dev/','')		
			if(contains_name(name_device)==False):
				#print(message)
				obj=utils_hosts.to_object(message)
				#Nome do sensor
				print("Nome: "+obj.HEADER['NAME'])
				
				#retorna a janela de dados em formato vetor e nao como string
				#print(predict([120.5,120.5,120.5,120.5,120.5,120.5,120.5,120.5,120.5,120.5,120.5,120.5,120.5,120.5]))
				ob=Preditive_obj()
				ob.name_device=name_device
				ob.sensor_ip=utils_hosts.return_host_per_name(name_device).ip
				ob.publish_time=0.0
				ob.name_gateway=name
				ob.gateway_ip=utils_hosts.return_host_per_name(name).ip
				ob.init_time=timeit.default_timer()
				#ob.predict_time=predict
				#l_sensors.append(ob)
		elif(topic.find('dev/')==0 and message.find("FLOW INFO temperatureSensor")==0):
			name_device=topic.replace('dev/','')
			if(contains_name(name_device)==True):
				modifi_time_publish(message,name_device)
				
				
	def on_connect(client, userdata, flags, rc):
		client.subscribe('#')
		
	def on_message(client, userdata, msg):
		MensagemRecebida = msg.payload.decode('utf-8')
		dataRaw = json.loads(msg.payload)
		value, change = parser_msg_value(dataRaw)
		
		ts=time.time()
		
		check_windows(value, change, ts)
		
		
		catch_message(msg.topic,MensagemRecebida)
	
	def parser_msg_value(data):
		global topic,group,Sensorid,DeviceId,last_concept,last_delay,init
		if(str(data).find('BODY') != -1):
			try:
					
				body = data['BODY']
				value = body['temperatureSensor']
				change = body['conceptDrift']
				
				print("change: ", change)
				#print("Teste topic",topic)
				return value, change
			except Exception as inst:
				print(inst)
	
	
	def check_windows(value, change, timestamp):
		global topic,group,Sensorid,DeviceId,last_concept,last_delay,init
		window_data = value.replace('[','').replace(']','').split(',')
		window_data_np=np.array(window_data)
		window_data_np=window_data_np.astype(float)	
		window_data = window_data_np.tolist()
		#print(window_data)
		df_data = pd.DataFrame(window_data)
		if(change == "True" or init == True):
			df_data['concept'] = 1
			last_delay = timestamp - last_concept 
			df_data['delay'] = last_delay
			last_concept = timestamp
			init = False
		else:
			df_data['concept'] = 0
			df_data['delay'] = last_delay
		#print(df_data)
	
	##Settings to paho mqtt
	print ("Init "+name)	
	client =mqtt.Client(client_id='', clean_session=True, userdata=None, protocol=mqtt.MQTTv31)
	client.on_connect = on_connect
	client.on_message = on_message
	client.connect(gateway, PortaBroker, KeepAliveBroker)
	client.loop_forever()


def preprocessiing():
   
    prep_dataset1 = pd.read_csv('/home/mininet/projeto_ml/FoT-Stream_Simulation/FoTStreamServer/kafkaMqtt/dataset_moteid-04-24_8.csv', delimiter=",")
    df = prep_dataset1.iloc[:,1:4]
    WINDOW = 35

    for i in np.arange(df.shape[0]):    
        init = i*WINDOW
        init2 = (i+1)*WINDOW
        if(init2<df.shape[0]):
            df.iloc[init:init+WINDOW,2] =  df.iloc[init2,2]
    df2 = normalizing(df)
    #train_size = int(len(df2) * 0.95)
    #return df2.iloc[0:train_size], df2.iloc[train_size:len(df2)]
    return df2


def normalizing(dataset):
    df_norm = pd.read_csv('/home/mininet/projeto_ml/FoT-Stream_Simulation/FoTStreamServer/kafkaMqtt/dataset_moteid-04-24_8.csv', delimiter=",")
    df_norm = df_norm.iloc[:,1:4]
    scaler = StandardScaler().fit(df_norm)

    scaler = scaler.fit(df_norm[['delay']])

    dataset['delay'] = scaler.transform(dataset[['delay']])
    return dataset


def unormalizing(y_pred):
    df_norm = pd.read_csv('/home/mininet/projeto_ml/FoT-Stream_Simulation/FoTStreamServer/kafkaMqtt/dataset_moteid-04-24_8.csv', delimiter=",")
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
        
def predict(window):
	model = keras.models.load_model('models/lstmv3.h5')

	window=np.array(window)

	#df = pd.DataFrame(window,columns=['temperature','concept','delay'])
	df = pd.DataFrame(window,columns=['delay'])

	
	df2=normalizing(df)
	
	nparray=np.array(df2)
	nparray=nparray.reshape(len(nparray),1,1)


	y_pred = model.predict(nparray)
	#print(y_pred)
	y_pred_inv=unormalizing(y_pred)
	return np.mean(y_pred_inv)
				
##start thread mqtt to each gateway
gateways=utils_hosts.return_hosts_per_type('gateway')
for i in range(0,len(gateways)):
	a = Thread(target=thread,args=(gateways[i].name_iot,gateways[i].ip))
	a.daemon=True
	a.start()
##End start thread to each gateway

for i in range(0,len(gateways)):
	a = Thread(target=thread_flow,args=(gateways[i].name_iot,gateways[i].ip))
	a.daemon=True
	a.start()
##End start thread to each gateway

#start thread mqtt_install_flow to each gateway


#LOOP To keep the prompt
while True:
	#try:
	time.sleep(4)
	#except KeyboardInterrupt:
	#	print "\nCtrl+C saindo..."
	#	sys.exit(0)
	#except Exception as e:
	#	print(e)
	#	sys.exit(0)
