#!/usr/bin/env python
# coding: utf-8

# In[11]:


#basic libraries
import joblib
import numpy as np
import matplotlib.image as mpimg
import os

#for images
# from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from skimage.transform import resize

#Model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


#take input in form of (path to data file, text saying what classifier should be used (i.e. bird_audio, bird_image, insect_audio, insect_image))
def classify(data_path, classifier):
    if not classifier in ["bird_audio", "bird_image", "insect_audio", "insect_image"]:
        raise ValueError("Given classifier must be one of: bird_audio, bird_image, insect_audio, insect_image")
    classifier = str(classifier)
    #load saved model
    model = joblib.load(classifier + "_model.pkl")
    img_path = data_path
    img = mpimg.imread(img_path)
    
    if classifier in ["bird_audio", "insect_audio"]:
        try:
            img = resize(img, (217, 334))
            img_np = np.array([img])
            img_rs = img_np.reshape(1, 217 * 334 * 4)
        except:
            raise Exception("Something went wrong, most likely the input spectrogram was of incorrect shape/dimensions.")
        
    elif classifier in ["bird_image","insect_image"]:
        try:
            img = resize(img, (224,224))
            img_np = np.array([img])
            img_rs = img_np.reshape(1, 224 * 224 * 3)
        except:
            raise Exception("Something went wrong, most likely the input image was of incorrect shape/dimensions.")
        
    pca = joblib.load(classifier + "_pca.pkl")
    img_pca = pca.transform(img_rs)
    
    #return predicted class
    return model.predict(img_pca)   

