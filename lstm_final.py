#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os,time
import scipy.io
import scipy.stats
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
#import sklearn.external.joblib as extjoblib
import joblib
#from sklearn.externals import joblib
import seaborn as sns
sns.set(color_codes=True)


print(tf.__version__)

#%%
#TensorFlow 환경설정
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
from numpy.random import seed
import tensorflow as tf
#tf.random.set_seed(x)
#tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers


# set random seed
seed(10)
tf.random.set_seed(10) 
#set_random_seed(10)


# In[ ]:

#파일 불러오기 & RMS 데이터프레임 생성
PHM_path = 'PHM\\mat'
PHM_bearing_files = [os.path.join(PHM_path,file) for file in os.listdir(PHM_path)]

def get_FPT(h2):

    kurt_list = []
    rms_list = []
    for i,row in enumerate(h2):
        kurt = scipy.stats.kurtosis(row)
        kurt_list.append(kurt)
        rms = np.mean(row**2)**0.5
        rms_list.append(rms)
        weight = np.concatenate([np.linspace(5,   4.5, 100),
                                 np.linspace(4.5, 4,   500),
                                 np.linspace(4,   3,   2000),
                                 np.linspace(3,   3,   3000)])
        w = weight[i]
        kurt_c = kurt > np.mean(kurt_list)+w*np.std(kurt_list)
        rms_c  = rms  > np.mean(rms_list) +w*np.std(rms_list)
        if kurt_c and rms_c:
            break
    return i
    

def mat_to_arr(file):
    h = scipy.io.loadmat(file)['h'].reshape(-1)
    h2 = h.reshape(-1, int(len(h)/2560))
#    print(len(h)/2560)
    rms = np.array( [np.mean(i**2)**0.5 for i in h2] )
    rms = np.convolve(rms,[0.3,0.4,0.3],mode='same')
    return h,rms


# In[ ]:
df = pd.DataFrame()   

plt.style.use(['dark_background'])

for file in PHM_bearing_files[:17]:

    h,rms = mat_to_arr(file)
    df[file[-14:-4]]=rms

df = df[['Bearing1_1','Bearing1_3','Bearing1_4']]
df=df[:-1]


print(df)
    
    
#%%
# train, test set 분류
train = df[0:1500]
test = df[1501:]
print("Training dataset shape:", train.shape)
print("Test dataset shape:", test.shape)



#%%

#그래프
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)

cols = df.columns.values

ax.plot(train['Bearing1_1'], label='Bearing1_1', color='b', animated = True, linewidth=2)
ax.plot(train['Bearing1_3'], label='Bearing1_3', color='r', animated =True, linewidth=2)
ax.plot(train['Bearing1_4'], label='Bearing1_4', color='g', animated =True, linewidth=2)


plt.legend(loc='upper left')
ax.set_title('Bearing Sensor Training Data', fontsize=16)

plt.show()


#%%

fig_test, ax_test = plt.subplots(figsize=(14, 6), dpi=80)


ax_test.plot(test['Bearing1_1'], label='Bearing1_1', color='b', animated = True, linewidth=2)
ax_test.plot(test['Bearing1_3'], label='Bearing1_3', color='r', animated =True, linewidth=2)
ax_test.plot(test['Bearing1_4'], label='Bearing1_4', color='g', animated =True, linewidth=2)


plt.legend(loc='upper left')
ax.set_title('Bearing Sensor Test Data', fontsize=16)

plt.show()


#%%

fig_test, ax_test = plt.subplots(figsize=(14, 6), dpi=80)

cols = df.columns.values

ax_test.plot(df['Bearing1_1'], label='Bearing1_1', color='b', animated = True, linewidth=2)
ax_test.plot(df['Bearing1_3'], label='Bearing1_3', color='r', animated =True, linewidth=2)
ax_test.plot(df['Bearing1_4'], label='Bearing1_4', color='g', animated =True, linewidth=2)


plt.legend(loc='upper left')
ax.set_title('Bearing Sensor Training Data', fontsize=16)

plt.show()


#%%
#FFT변환

train_fft = np.fft.fft(train)
test_fft = np.fft.fft(test)


#%%
#정규화 및 전처리
# normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
X_test = scaler.transform(test)
scaler_filename = "scaler_data"
joblib.dump(scaler, scaler_filename)

# reshape inputs for LSTM [samples, timesteps, features]
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
print("Training data shape:", X_train.shape)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
print("Test data shape:", X_test.shape)



#%%
# 학습모델 생성
# define the autoencoder network model
def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model


#%%
# create the autoencoder model
model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')
model.summary()


#%%

#학습
# fit the model to the data
nb_epochs = 100
batch_size = 5
history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                    validation_split=0.05).history
#%%


# plot the training losses
fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
ax.plot(history['loss'], 'b', label='Train', linewidth=2)
ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
ax.set_title('Model loss', fontsize=16)
ax.set_ylabel('Loss (mae)')
ax.set_xlabel('Epoch')
ax.legend(loc='upper right')
plt.show()


#학습 결과물 검증
#%%
# plot the loss distribution of the training set
X_pred = model.predict(X_train)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=train.columns)
X_pred.index = train.index



#%%
scored = pd.DataFrame(index=train.index)
Xtrain = X_train.reshape(X_train.shape[0], X_train.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
plt.figure(figsize=(16,9), dpi=80)
plt.title('Loss Distribution', fontsize=16)
plt.plot([0.22,0.22],[0,15],'r-.',label='Threshold', linewidth=2)
sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue')
plt.xlim([0.0,.5])



#%%
# calculate the loss on the test set
X_pred = model.predict(X_test)
X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
X_pred = pd.DataFrame(X_pred, columns=test.columns)
X_pred.index = test.index

scored = pd.DataFrame(index=test.index)
Xtest = X_test.reshape(X_test.shape[0], X_test.shape[2])
scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtest), axis = 1)
scored['Threshold'] = 0.35
scored['Anomaly'] = scored['Loss_mae'] > scored['Threshold']
scored.head()

#%%
# calculate the same metrics for the training set 
# and merge all data in a single dataframe for plotting
X_pred_train = model.predict(X_train)
X_pred_train = X_pred_train.reshape(X_pred_train.shape[0], X_pred_train.shape[2])
X_pred_train = pd.DataFrame(X_pred_train, columns=train.columns)
X_pred_train.index = train.index

scored_train = pd.DataFrame(index=train.index)
scored_train['Loss_mae'] = np.mean(np.abs(X_pred_train-Xtrain), axis = 1)
scored_train['Threshold'] = 0.35
scored_train['Anomaly'] = scored_train['Loss_mae'] > scored_train['Threshold']


scored = pd.concat([scored_train, scored])


#%%

# plot bearing failure time plot
scored.plot(logy=True,  figsize=(16,9), ylim=[1e-2,1e2], color=['blue','red'])




