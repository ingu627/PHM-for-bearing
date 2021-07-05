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

print(tf.__version__)


# In[ ]:
#파일 불러오기

PHM_path = 'PHM\\mat1'
PHM_bearing_files = [os.path.join(PHM_path,file) for file in os.listdir(PHM_path)]
    

def mat_to_arr(file):
    h = scipy.io.loadmat(file)['h'].reshape(-1)
    h2 = h.reshape(-1,2560)
    kurtosis = np.array( [scipy.stats.kurtosis(i) for i in h2] )
    rms = np.array( [np.mean(i**2)**0.5 for i in h2] )
    rms = np.convolve(rms,[0.3,0.4,0.3],mode='same')
    ma = np.array( [np.max(np.abs(i)) for i in h2] )
    FPT = int(len(h2))*1700/2560
    print(FPT)
    return h,FPT,kurtosis,rms,ma



# In[ ]:
    
#RMS 그래프 그리기

plt.style.use(['dark_background'])

for file in PHM_bearing_files[:3]:

    h,FPT,kurtosis,rms,ma = mat_to_arr(file)
    tlong = np.linspace(0,len(rms)*10,len(h))
    tshort = np.linspace(0,len(rms)*10,len(rms))
    FPT = FPT*10
    print(FPT)
            
    plot_title=file[-14:-4]
    
    plt.figure(figsize=(5,6))
    plt.suptitle(file[-14:-4])
    
    plt.subplot(311)
    plt.plot(tlong,h,'b',label='Signal')
    plt.plot([FPT,FPT],[min(h),max(h)],'r-.',label='FPT')

    plt.yticks(fontproperties = 'Times New Roman', size = 10)
    plt.xticks(fontproperties = 'Times New Roman', size = 10)
    plt.ylabel('Signal (g)', fontproperties = 'Times New Roman', fontsize=12)

    
    plt.subplot(312)
    plt.plot(tshort,rms,'b',label='RMS')
    plt.plot([FPT,FPT],[min(rms),max(rms)],'r-.',label='FPT')

    plt.yticks(fontproperties = 'Times New Roman', size = 10)
    plt.xticks(fontproperties = 'Times New Roman', size = 10)
    plt.xlabel('Time (s)', fontproperties = 'Times New Roman', fontsize=12)
    plt.ylabel('RMS (g)', fontproperties = 'Times New Roman', fontsize=12)

    plt.show()


# In[ ]:

#Test 베어링 설정 

test_bearing = ['1_1','1_3','1_4']

test_bearing_files = []
for f in PHM_bearing_files:
    if f[-7:-4] in test_bearing:
        test_bearing_files.append(f)
test_bearing_files


# In[ ]:


def get_fea_FPT(bearing_files):
    feature_list = []
    FPT_list = []
    for i,file in enumerate(bearing_files):
        h,FPT,kurtosis,rms,ma = mat_to_arr(file)
        h2 = h.reshape(-1,2560)        
        fea = np.concatenate([rms[:].reshape(-1,1),
                                       ma[:].reshape(-1,1),],  axis=1)
        print(file[-14:-4], h2.shape, fea.shape,FPT, sep='\t')
        feature_list.append(fea)
        FPT_list.append(FPT)

    return feature_list,FPT_list


feature_list,FPT_list = get_fea_FPT(test_bearing_files)
len(feature_list), feature_list[0].shape


# In[ ]:


#FPT_list = [1463,1644,1091]
FPT_list = [1861,1577,948]
FT_list = [2763,2287,1139]


# In[ ]:
#학습모델 구축

class EModel(tf.Module):
    def __init__(self,init=[1.0,0.005,0.0,0.0],**kwargs):
        super().__init__(**kwargs)

        inita = tf.constant(init[0],dtype=tf.float32)
        initb = tf.constant(init[1],dtype=tf.float32)
        print('initb',initb)
        initc = tf.constant(init[2],dtype=tf.float32)
        initd = tf.constant(init[3],dtype=tf.float32)

        
        self.a = tf.Variable(tf.math.log(inita), name='a',dtype=tf.float32)
        self.b = tf.Variable(tf.math.log(initb), name='b',dtype=tf.float32)
        self.c = tf.Variable(initc, name='c',dtype=tf.float32)
        self.d = tf.Variable(initd, name='d',dtype=tf.float32)
        print('Init:',self.__call__(100.0).numpy(),self.a.numpy(),tf.exp(self.a).numpy())
        
        
        
    def __call__(self, x):
        x = tf.cast(x,dtype=tf.float32)
        a,b,c,d = self.a, self.b, self.c, self.d
        a = tf.exp(a)
        b = tf.exp(b)+0.0005
        y = a*tf.exp(b*x+d) +c

        return y
init_model = EModel()
zz = np.linspace(-200,1000,1000)
plt.figure(figsize=(10,3))
plt.plot(zz,init_model(zz),'b')
plt.xlim(-200,1000)
plt.ylim(-1,10)
plt.grid()


# In[ ]:
# 그래프 함수

def plot_predict_curve(init_model,trained_model,history_time,history_data,all_time,all_data,ckpt,
                       thd,predict_FT=None,fn=None,xlim=None,):
    plt.figure(figsize=(10,6))
    plt.scatter(all_time*10, all_data, c="r", label='After checkpoint')
    plt.scatter(history_time*10, history_data, c="b", label='Before checkpoint')
#    plt.suptitle(test_bearing_files[n][-14:-4],fontproperties = 'Times New Roman', fontsize=25)  
    
    # FPT,FT,Failure threshold
    plt.plot([FPT*10,FPT*10],[0,10],'m--',linewidth=3, label='FPT')
    plt.plot([FT*10,FT*10],[0,10],'w--',linewidth=3, label='FT')
    if predict_FT is not None:
        plt.plot([thd]*5000*10,'r',linewidth=3, label='Failure threshold')
        plt.plot([predict_FT*10,predict_FT*10],[0,10],'y',label='Predicted FT')
    else:
        plt.plot([thd]*5000*10,'r',linewidth=3,label='Failure threshold')

    

    t = np.arange(-1000,2000) 
    t2 = (t + FPT)*10  
    plt.plot(t2, init_model(t),    "c--", linewidth=3, label='Prediction before training')
    if trained_model is not None:
        plt.plot(t2, trained_model(t), "g--", linewidth=3, label='Prediction after training')
        
    
    

    plt.xlabel('Time (s)',fontproperties = 'Times New Roman', fontsize=20)
    plt.ylabel('RMS (g)',fontproperties = 'Times New Roman', fontsize=20)
    plt.yticks(fontproperties = 'Times New Roman', size = 20)
    plt.xticks(fontproperties = 'Times New Roman', size = 20)
    

    if xlim is not None:
        plt.xlim(xlim)
    plt.ylim(0,5)
    
    
    
    
    plt.legend(loc='upper left')
    if fn is not None:
        plt.savefig(fname=fn, dpi=300,)

    plt.show()
    
    
    
    
def get_init_model(data):
    st = np.min(data)
    k = 100 if len(data)>100 else 5
    en = np.mean(data[-k:])
    a = 0.5
    c = st-a 
    b = np.log((en-c)/a)/len(data)
    print(1111111,b)
    d = 0.0
    init = [a,b,c,d]
    model = EModel(init) 
    return model

threshold = [1.6,1.5,3.0]


# In[ ]:
# 그래프 생성

for n in range(3):

    data  = feature_list[n][:,0]

    
    FPT = FPT_list[n]
    FT = FT_list[n]
    thd = threshold[n]
    if n==0: FPT=2117


    max_RUL = FT - FPT
    checkpoint = FPT + int(max_RUL*0.99)

    history_data = data[:checkpoint+1]
    history_time = np.arange(len(history_data))


    all_data = data
    all_time = np.arange(len(all_data))


    init_model = get_init_model(history_data[FPT:])


    plot_predict_curve(init_model,None,history_time,history_data,all_time,all_data,ckpt=checkpoint,
                       thd=thd,fn=None,xlim=[10*(FPT-60),10*(FT+50)])    
    plot_predict_curve(init_model,None,history_time,history_data,all_time,all_data,ckpt=checkpoint,
                       thd=thd,fn=None,xlim=None) 


# In[ ]:
# 학습 모델 구축

def loss(y_true, y_pred):
    w = np.linspace(0.5,2.0,len(y_true))

    weight = w * y_true

    se = tf.square(y_true - y_pred) 
    se = se * weight
    mse = tf.reduce_mean(se)
    return mse


def train_step(model, x_true,y_true):
    lr = 5e-5
    opt = tf.keras.optimizers.Adam(lr)
    with tf.GradientTape() as gt:
        y_pred  = model(x_true)
        current_train_step_loss = loss(y_true, y_pred)
    gradients = gt.gradient(current_train_step_loss, model.trainable_variables) 
    opt.apply_gradients(zip(gradients, model.trainable_variables)) 
    return current_train_step_loss

def train_model(history_data):

    y_true = history_data[FPT:]
    x_true = np.arange(len(y_true))
    print("train_model  history_data.shape=%d,  y_true.shape=%d,  x_true[0]=%d, x_true[-1]=%d" %
                  (len(history_data),len(y_true),x_true[0],x_true[-1]))

    model  = get_init_model(y_true)
    aa,bb,cc,dd,losses =[], [], [], [], []
    for step in range(10000):
        train_step(model, x_true,y_true )
        aa.append(model.a.numpy())
        bb.append(model.b.numpy())
        cc.append(model.c.numpy())
        dd.append(model.d.numpy())
        y_pred = model(x_true)
        losses.append(loss(y_true, y_pred).numpy())
        
        if step%100==0:
            print("step  %3d: a=%1.8f    b=%1.8f    c=%1.8f    d=%1.8f,   loss=%2.8f" %
                  (step, aa[-1], bb[-1], cc[-1], dd[-1], losses[-1]))
            print("step  %3d: a=%1.8f    b=%1.8f    c=%1.8f    d=%1.8f,   loss=%2.8f \n" %
                  (step, np.exp(aa[-1]), np.exp(bb[-1]), np.exp(cc[-1]), np.exp(dd[-1]), losses[-1]))
        min_step = 50 
        max_step = 200
        if step>max_step or step>=min_step and max(losses[-20:-1] )<=0.2:
            break
        log = np.stack([aa,bb,cc,dd,losses] )
    return get_init_model(y_true),model, log

init_model,trained_model,train_log = train_model(history_data)


# In[ ]:


plot_predict_curve(init_model,trained_model,history_time,history_data,all_time,all_data,ckpt=checkpoint,
                   thd=thd,fn='train_process',xlim=[(FPT-int((FT-FPT)*1.5))*10,(FT+int((FT-FPT)*0.5))*10])


#%%
#predict FT 계산
def get_predict_FT(history_data,trained_model,ckpt,threshold):

    t = ckpt-FPT
    predictd_rms = trained_model(t).numpy()
    while predictd_rms<threshold:
        t = t+1
        predictd_rms = trained_model(t).numpy()
    t = t
    FT = t + FPT
    if FT<1 : FT=1
    return FT

history_data_list= []
history_time_list = []
checkpoint_list = []
all_data_list =[]
all_time_list =[]

for n in range(3):

    data  = feature_list[n][:,0]

    
    FPT = FPT_list[n]
    FT = FT_list[n]
    thd = threshold[n]
    if n==0: FPT=2117


    max_RUL = FT - FPT
    checkpoint = FPT + int(max_RUL*0.99)
    checkpoint_list.append(checkpoint)


    history_data = data[:checkpoint+1]
    history_data_list.append(history_data)
    
    history_time = np.arange(len(history_data))
    history_time_list.append(history_time)

    all_data = data
    all_data_list.append(all_data)
    all_time = np.arange(len(all_data))
    all_time_list.append(all_time)


    init_model = get_init_model(history_data[FPT:])

    
    init_model,trained_model,train_log = train_model(history_data)
    plot_predict_curve(init_model,trained_model,history_time,history_data,all_time,all_data,ckpt=checkpoint,
                   thd=thd,fn='train_process',xlim=[(FPT-int((FT-FPT)*1.5))*10,(FT+int((FT-FPT)*0.5))*10])
    

    predict_FT = get_predict_FT(history_data, trained_model,checkpoint,threshold=thd)
    print('predict FT is')
    print(predict_FT)

#    predict_FT = get_predict_FT(history_data, trained_model,checkpoint,threshold=thd)
#    print(predict_FT)

    plot_predict_curve(init_model,trained_model,history_time,history_data,all_time,all_data,ckpt=checkpoint,thd=thd,
                       predict_FT=predict_FT,fn='bearing1_3_RUL',
                       xlim=[(FPT-int((FT-FPT)*0.2))*10,(max(FT,predict_FT)+int((FT-FPT)*0.2))*10])


#%%


# 지수 함수 모델 그래프
zz = np.linspace(-100,2000,100)

# a1 = np.exp(zz)
# a2 = -1*np.exp(zz)
# a3 = np.exp(-1*zz)
# a4 = -1*np.exp(-1*zz)


plt.figure(figsize=(12,6))
plt.plot(zz, np.exp(0.001*zz), 'r-.', label='np.exp(x)')
plt.plot(zz, 0.001*np.exp(0.01*zz), 'b-.', label='-1*np.exp(x)')

plt.plot(zz, np.exp(0.002*zz), 'c-.', label='np.exp(-1*x)')
plt.plot(zz, np.exp(0.004*zz), 'w-.', label='-1*np.exp(-1*x)')
plt.legend()
plt.ylim(-5,50)
plt.grid()
plt.xlabel('input', fontproperties = 'Times New Roman', size = 15)
plt.ylabel('output', fontproperties = 'Times New Roman', size = 15)

plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
# plt.savefig('ab.png',dpi=300)
plt.show()



#%%
# 결과 비교 그래프
w=0.3

bearings = ['Bearing 1_1', 'Bearing 1_3', 'Bearing 1_4']

PFT=[2756,2279,1261]

x = np.arange(3)
plt.xticks([i+0.15 for i in range(len(x))], bearings, fontsize=12)
#plt.xticks(x, bearings, fontsize=15)

years = ['2017', '2018', '2019']
values = [100, 400, 900]

plt.bar(range(len(PFT)), PFT, width=w, color='b', label='Predict_FT')

plt.bar([i+w for i in range(len(FT_list))], FT_list, width=w, color='r', label='FT')
plt.legend(loc='upper right')

print(FT_list)
print(PFT)




