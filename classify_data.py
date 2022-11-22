#!/usr/bin/env python
# coding: utf-8

# In[26]:


#basic libraries
import joblib
# import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

#for images
# from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from skimage.transform import resize

#Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

#take input in form of (path to image file, text saying what classifier should be used (i.e. bird_audio, bird_image, insect_audio, insect_image))
def classify(image_path, classifier):
    #load saved model
    model = joblib.load(classifier + "_model.pkl")
    
    #do weird stuff to be able to resize image, do PCA, and input into model
    img_path = image_path
    img = mpimg.imread(img_path)
    img = resize(img, (224,224))
    img_np = np.array([img])
    img_rs = img_np.reshape(1, 224 * 224 * 3)
    pca = joblib.load(classifier + "_pca.pkl")
    img_pca = pca.transform(img_rs)
    
    #return predicted class
    return model.predict(img_pca)   

