#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import scipy.io
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import pandas as pd
from IPython.display import clear_output


# In[2]:


#파일 경로 설정
d = pd.read_csv('PHM\\Learning_set\\Bearing1_1\\acc_00001.csv',header=None,sep=',')
d


# In[3]:


plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(d.iloc[:,-2])
plt.title('Horizontal_vibration_signals')
plt.subplot(122)
plt.plot(d.iloc[:,-1])
plt.title('Vertical_vibration_signals')
plt.show()


# In[4]:


def get_a_bearings_data(folder):
    #csv 파일에서 데이터를 가져와서 numpy 배열로 변환
    names = os.listdir(folder)
    is_acc = ['acc' in name for name in names] 
    names = names[:sum(is_acc)]
    files = [os.path.join(folder,f) for f in names]
    print(pd.read_csv(files[0],header=None).shape)
    sep = ';' if pd.read_csv(files[0],header=None).shape[-1]==1 else ','
    h = [pd.read_csv(f,header=None,sep=sep).iloc[:,-2] for f in files]
    v = [pd.read_csv(f,header=None,sep=sep).iloc[:,-1] for f in files]
    H = np.concatenate(h)
    V = np.concatenate(v)
    print(H.shape,V.shape)
    return np.stack([H,V],axis=-1)

data = get_a_bearings_data('PHM\\Learning_set\\Bearing1_1')
data = get_a_bearings_data('PHM\\Full_Test_Set\\Bearing1_4')
data.shape


# In[5]:


plt.figure(figsize=(20,5))
plt.subplot(121)
plt.plot(data[:,0])
plt.title('Horizontal_vibration_signals')
plt.subplot(122)
plt.plot(data[:,1])
plt.title('Vertical_vibration_signals')
plt.show()


# In[6]:


p = 'PHM'

for i in ['Learning_set','Full_Test_Set']:
    pp = os.path.join(p,i)
    for j in os.listdir(pp):
        ppp = os.path.join(pp,j)
        print(ppp)
        
        
        data = get_a_bearings_data(ppp)
        save_name = p + '\\mat\\' + j+'.mat'
        print(save_name)
        scipy.io.savemat(save_name,{'h':data[:,0], 'v':data[:,1]})    
    print('\n')




