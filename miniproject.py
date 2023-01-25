#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas_datareader.data as pdr


# In[2]:


key='0611223235d3e2874e5253a424db62d4a652f271'


# In[3]:


df=pdr.get_data_tiingo('AAPL',api_key=key)


# In[4]:


df.to_csv('AAPL.csv')


# In[5]:


import pandas as pd


# In[6]:


df=pd.read_csv('AApl.csv')


# In[7]:


df


# In[8]:


df1=df.reset_index()['close']


# In[9]:


df1.shape


# In[10]:


df1


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


plt.plot(df1)
plt.xlabel("Apple_StockPrice_day(02-10-2017 to 29-09-2022)")
plt.ylabel("Close_Price_USD")


# In[13]:


import numpy as np


# In[14]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[15]:


df1


# In[16]:


training_size=int(len(df1)*0.70)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


# In[17]:


import numpy
def create_dataset(dataset,time_step=1):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        a=dataset[i:(i+time_step),0]
        dataX.append(a)
        dataY.append(dataset[i+time_step,0])
    return numpy.array(dataX),numpy.array(dataY)


# In[18]:


time_step=100
X_train,y_train=create_dataset(train_data,time_step)
X_test,y_test=create_dataset(test_data,time_step)


# In[19]:


print(X_train.shape),print(y_train.shape)


# In[20]:


print(X_test.shape),print(y_test.shape)


# In[21]:


X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],1)


# In[22]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


# In[23]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')


# In[24]:


model.summary()


# In[25]:


model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100,batch_size=64,verbose=1)


# In[26]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[27]:


train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[27]:


import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[28]:


math.sqrt(mean_squared_error(y_test,test_predict))


# In[29]:


look_back=100
trainPredictPlot=numpy.empty_like(df1)
trainPredictPlot[:,:]=np.nan
trainPredictPlot[look_back:len(train_predict)+look_back,:]=train_predict
testPredictPlot=numpy.empty_like(df1)
testPredictPlot[:,:]=numpy.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1,:]=test_predict
plt.plot(scaler.inverse_transform(df1))
plt.xlabel("Apple_StockPrice_day(02-10-2017 to 29-09-2022)")
plt.ylabel("Close_Price_USD")
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[30]:


x_input=test_data[278:].reshape(1,-1)
x_input.shape


# In[31]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[32]:


from numpy import array
lst_output=[]
n_steps=100
i=0
while(i<30):
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{}day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input=x_input.reshape((1,n_steps,1))
        yhat=model.predict(x_input,verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1

print(lst_output)
        


# In[33]:


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


# In[34]:


import matplotlib.pyplot as plt 


# In[35]:


len(df1)


# In[36]:


df3=df1.tolist()
df3.extend(lst_output)


# In[37]:


print("Orange line is predicted price for next 30 days(29-09-2022 to 29-10-2022)")
plt.plot(day_new,scaler.inverse_transform(df1[1158:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
plt.xlabel("Apple_StockPrice_day(22-06-2022 to 29-10-2022)")
plt.ylabel("Close_Price_USD")


# In[ ]:


df3=scaler.inverse_transform(df3)
plt.plot(df3[1200:])
plt.xlabel("Apple_StockPrice_day(01-08-2022 to 29-10-2022)")
plt.ylabel("Close_Price_USD")


# In[ ]:


scaler.inverse_transform(lst_output)


# In[38]:


scaler.inverse_transform(lst_output)


# In[39]:


print("Price predicted of tommorow(30-09-2022)-")
scaler.inverse_transform(lst_output[0:1])


# In[ ]:




