#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import VGG19, VGG16, ResNet50, resnet50
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


# In[13]:


base_model=ResNet50(weights='imagenet',include_top=False) 
base_model.save('ResNet50model.h5')



base_model=VGG16(weights='imagenet',include_top=False) 
base_model.save('VGG16model.h5')



from keras.models import load_model
model = load_model('vgg19model.h5')

x=model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(120,activation='softmax')(x) #final layer with softmax activation


# In[6]:


base_model.save('vgg19model.h5')


# In[ ]:


from keras.models import load_model

model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'
del model  # deletes the existing model

# returns a compiled model
# identical to the previous one
model = load_model('my_model.h5')


# In[9]:


model=Model(inputs=model.input,outputs=preds)


# In[ ]:


model.predict()


# In[10]:


import os
os.getcwd()

