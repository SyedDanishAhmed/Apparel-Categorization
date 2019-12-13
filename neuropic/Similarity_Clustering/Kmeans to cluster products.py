#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np

import imageio
import shutil
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# import skfuzzy


# In[2]:


############################### Function to get the categories (jackets etc..) ###########################
class Categories:
    def __init__(self, path):
        self.path = path
    def get_categories(self):
        categories=[]
        for i in range(len(self.path)):
            categories.append(self.path[i][findnth(self.path[i],"\\",1)+1:findnth(self.path[i],"\\",2)])

        categories=set(categories)
        return categories


# In[3]:


############################### Function to get the sub categories (file names) ###########################
class Sub_categories(Categories):
    def __init__(self, path):
        super().__init__(path)
    def get_sub_categories(path):
        sub_categories=[]
        for i in range(len(self.path)):
            sub_categories.append(self.path[i][findnth(self.path[i],"\\",2)+1:findnth(self.path[i],"\\",3)])

        sub_categories=set(sub_categories)
        return sub_categories


# In[4]:


########################### Function to decrease the resolution of the image ################### 
def resize_image(image_array):
    scale_percent = 60 # percent of original size
    width = int(image_array.shape[1] * scale_percent / 100)
    height = int(image_array.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(image_array, dim, interpolation = cv2.INTER_AREA) 
    return resized


# In[5]:


################################### function to convert an image into an np array ##########################
def create_image_np_array(paths):
    images=[]
    for path in paths:
        original_image=imageio.imread(path)
        resized_image=resize_image(original_image)
        images.append(resized_image.flatten().tolist())
    return np.array(images)


# In[6]:


############################### Model class that trains on images ################################
class Model:
    def __init__(self, image_array,cluster_size):
        self.image_array=image_array
        self.cluster_size=cluster_size
    def model_fitting(self):
        kmeans = KMeans(n_clusters=self.cluster_size, n_init=1000, random_state=0)
        return kmeans.fit(self.image_array)


# In[7]:


################################# function to assign clusters to the images #########################
def model_predict(kmeans,image_array):
    predicted_array=kmeans.predict(image_array)
    return predicted_array


# In[8]:


##################### function to find the nth letter ##########################
def findnth(haystack, needle, n):
    parts= haystack.split(needle, n)
    if len(parts)<=n:
        return -1
    return len(haystack)-len(parts[-1])-len(needle)


# In[9]:


######################## Class with the purpose of transfering the files to appropriate clusters ####################

class Copy:
    def __init__(self, cluster_size,predicted_array,source,category):
        self.cluster_size=cluster_size
        self.predicted_array=predicted_array
        self.source=source
        self.category=category
    def transfer(self):
        path_new="./neuropic/Clustering results/"+self.category+"/"
        if os.path.exists(path_new):
            shutil.rmtree(path_new)
        os.mkdir(path_new)
        for i in range(self.cluster_size):  
            os.mkdir(path_new+str(i))
        for i,j in enumerate(self.predicted_array):
            src_file=self.source[i]
            dest=path_new+str(j)
            shutil.copy(src_file, dest)


# In[10]:


###################### transfer classified images into appropriate folder for display in the User interface #################
def transfer_red_jackets():
        path_new="./neuropic/UI/img/product_details/jacket-vests-red/similar-products"
        src_path=glob.glob("D:/Jackets_Vests/1/*")
        for path in glob.glob(path_new+"/*"):
            os.remove(path)
        for src in src_path:
            dest=path_new
            shutil.copy(src, dest)
def transfer_yellow_jackets():
    path_new="./neuropic/UI/img/product_details/jacket-vests-yellow/similar-products"
    for path in glob.glob(path_new+"/*"):
        os.remove(path)
    src_path=glob.glob("D:/Jackets_Vests/2/*")
    for src in src_path:
        dest=path_new
        shutil.copy(src, dest)           


# In[11]:


all_path=glob.glob("./neuropic/Categorization Neural Net/Data/Raw Data/Train/*/*Train*") ### all files having train in them will be taken
CLUSTER_SIZE=3
paths=all_path
categories_object=Categories(all_path)
categories=categories_object.get_categories()  ##### finding out the categories
categories = list(categories)
for category in categories:   ### running clustring for each category to find the similar products , category wise
    paths=glob.glob("./neuropic/Categorization Neural Net/Data/Raw Data/Train\\"+category+"\\*Train*")
    image_array=create_image_np_array(paths)   ### converting images into an numpy array
    model=Model(image_array,CLUSTER_SIZE)      #### training the images using kmeans clustering
    kmeans=model.model_fitting()
    predicted_array=model_predict(kmeans,image_array)  ### classifying the images into clusters
    copy=Copy(CLUSTER_SIZE,predicted_array,paths,category)  ### transfering files into appropriate clusters for ease of viewing
    copy.transfer()
transfer_red_jackets()
transfer_yellow_jackets()
    


# In[ ]:




