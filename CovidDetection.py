#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shutil
import os
import pandas as pd
from keras.preprocessing import image
import cv2
import numpy as np
from keras.utils import np_utils


# In[2]:


# df=pd.read_csv('covid_dataset/metadata.csv')
# df.head(n=3)


# In[3]:


# source_path='covid_dataset/images/'
# dest_path='covid/'
# for i,row in df.iterrows():
#     if row['finding']=='Pneumonia/Viral/COVID-19' and row['view']=='PA':
#             filname=row['filename']
#             shutil.copy2(os.path.join(source_path,filname),os.path.join(dest_path,filname))


# In[4]:


disease_to_num={
    'covid':0,
    'normal':1
}
num_to_disease={
    0:'covid',
    1:'normal'
}
image_dataset=[]
labels=[]
for foldername in os.listdir('dataset'):
    for filename in os.listdir(os.path.join('dataset/',foldername)):
        img=image.load_img(os.path.join(os.path.join('dataset/',foldername),filename))
        img_array=image.img_to_array(img)
        img_array=cv2.resize(img_array,(100,100))
        image_dataset.append(img_array)
        labels.append(disease_to_num[foldername])
image_dataset=np.array(image_dataset)
labels=np.array(labels)


# In[5]:


# combine=zip(image_dataset,labels)
# li =np.array(list(combine))
# np.random.shuffle(li)
# X=np.array(li[:,0])
# Y=np.array(li[:,1])


# In[ ]:





# In[6]:


from keras.models import Sequential
from keras.layers import Convolution2D,MaxPool2D,Dropout,Dense,Flatten


# In[7]:


model=Sequential()
model.add(Convolution2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))
model.add(Convolution2D(64,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Convolution2D(32,(3,3),activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[8]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[9]:


model.fit(image_dataset,labels,validation_split=0.2,epochs=18,shuffle=True,batch_size=115)


# In[10]:


image_dataset.shape


# In[ ]:





# In[ ]:




