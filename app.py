#!/usr/bin/env python
# coding: utf-8

# # Install Package

# In[136]:


#coding=utf-8
import pandas as pd
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector,Input
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read File

# In[137]:


train2018 = pd.read_csv('台灣電力公司_過去電力供需資訊2018.csv', engine='python',header = None)
train2018 = train2018.drop(train2018.index[0])
train2018.head(3)


# In[138]:


train2019 = pd.read_csv('台灣電力公司_過去電力供需資訊2019.csv', engine='python',header = None)
train2019 = train2019.drop(train2019.index[0])
train2019.columns = ["date","y"]
train2019.head(3)


# In[139]:


train2018.head()


# # Data Preprocessing

# In[140]:


#train2018.describe(include="all")


# In[141]:


#train2018[["日期"]] = train2018["日期"].astype("float")
#train2018.iloc[:,0] = pd.to_datetime(train2018.iloc[:,0], format='%Y%m%d')
#train2019["date"] = pd.to_datetime(train2019["date"], format='%Y%m%d')


# In[142]:


#train2018.dtypes


# In[143]:


#查看特定變數欄
train_y = train2018.iloc[:,1]
#train_y.head(3)
#train2018 = train2018.drop(2, axis = 1)
train_y = train_y.astype("float64")


# In[144]:


plt.plot(train_y)
plt.show()


# # Feature Selection

# In[145]:


train2018 = train2018.loc[:,[0,2]]
train2018.head()


# In[146]:


#train2018["year"] = train2018.loc[:,0].dt.year
#train2018["month"] = train2018.loc[:,0].dt.month
#train2018["date"] = train2018.loc[:,0].dt.day
#train2018["day"] = train2018.loc[:,0].dt.dayofweek


# In[147]:


#train2018 = train2018.drop(0, axis=1)
#train2018 = train2018.drop(1, axis=1)


# In[148]:


train2018.columns = ["date", "y"]


# In[149]:


#train2018.head()


# In[150]:


#train2019["year"] = train2019.loc[:,0].dt.year
#train2019["month"] = train2019.loc[:,0].dt.month
#train2019["date"] = train2019.loc[:,0].dt.day
#train2019["day"] = train2019.loc[:,0].dt.dayofweek


# In[151]:


#train2019 = train2019.drop(["date"], axis=1)


# In[152]:


#train2019.columns = ["y", "year", "month", "date", "day"]


# In[153]:


train2019.head()


# In[154]:


#train2019 = pd.concat([train2018, train2019], axis = 0)


# In[155]:


train2019.head()


# In[156]:


#def normalize(df):
#    norm = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
#    return norm


# In[157]:


#traindata.iloc[:,1] = 
#traindata.iloc[:,1:] = normalize(traindata.iloc[:,1:])


# In[158]:


#traindata.head()


# # LSTM

# ## Model

# In[159]:


def buildTrain(train, pastDay=20, futureDay=8):
    X_train, Y_train = [], []
    for i in range(train.shape[0]-futureDay-pastDay):
        X_train.append(np.array(train.iloc[i:i+pastDay]))
        Y_train.append(np.array(train.iloc[i+pastDay:i+pastDay+futureDay]["y"]))
    return np.array(X_train), np.array(Y_train)


# In[160]:


def shuffle(X,Y):
    np.random.seed(1234)
    randomList = np.arange(X.shape[0])
    np.random.shuffle(randomList)
    return X[randomList], Y[randomList]


# In[161]:


def splitData(X,Y,rate):
    X_train = X[int(X.shape[0]*rate):]
    Y_train = Y[int(Y.shape[0]*rate):]
    X_val = X[:int(X.shape[0]*rate)]
    Y_val = Y[:int(Y.shape[0]*rate)]
    return X_train, Y_train, X_val, Y_val


# In[162]:


scaler = MinMaxScaler(feature_range=(0, 1))
train_norm = scaler.fit_transform(train2019)
train_norm = pd.DataFrame(train_norm)
train_norm['y'] = train_norm[0]
train_norm = train_norm.drop(0,axis=1)
#train_norm.columns = [["y"]]


# In[163]:


# build Data, use last 20 days to predict next 8 days
X_train, Y_train = buildTrain(train_norm, 20, 8)
# shuffle the data, and random seed is 1234
X_train, Y_train = shuffle(X_train, Y_train)


# In[164]:


print(X_train.shape)
print(Y_train.shape)


# In[165]:


def buildManyToManyModel(shape):
    model = Sequential()
    model.add(LSTM(256, input_shape=(shape[1], shape[2]), return_sequences=True))
    model.add(LSTM(256, return_sequences=True))
    model.add(TimeDistributed(Dense(1)))
    model.add(Flatten())
    model.add(Dense(20,activation='linear'))
    model.add(Dense(8,activation='linear'))
    model.compile(loss="mean_absolute_error", optimizer="adam",metrics=['mean_absolute_error'])
    model.summary()
    return model


# In[166]:


model = buildManyToManyModel(X_train.shape)
callback = EarlyStopping(monitor="mean_absolute_error", patience=10, verbose=1, mode="auto")

history = model.fit(X_train, Y_train, epochs=1000, batch_size=5, validation_split=0.1, callbacks=[callback],shuffle=True)


# In[167]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()


# In[168]:


X_train.shape


# ## Prediction

# In[169]:


X_predict = np.array(train_norm[len(train_norm)-20:]).reshape((1,20,2))
predict = model.predict(X_predict)
print(predict.reshape)


# In[170]:


print(scaler.data_max_)
print(scaler.data_min_)


# In[171]:


predict = predict * (scaler.data_max_[1]-scaler.data_min_[1]) + scaler.data_min_[1]
# = scaler.inverse_transform(predict)


# In[172]:


predict = predict.reshape((8))
predict


# In[173]:


predict = predict[1:8]


# In[174]:


guessdate = np.array([20190402, 20190403, 20190404, 20190405, 20190406, 20190407, 20190408])


# # Output .csv file

# In[176]:


guess = {"date":guessdate, "peak_load(MW)":predict}
df_guess=pd.DataFrame(guess)
df_guess.to_csv("submission.csv", header=True, index=False)

