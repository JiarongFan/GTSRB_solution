#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import cvxopt
import pandas as pd
import matplotlib.pyplot as plt
import os
from svmutil import *
from svm import *
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure

# read label and image location
train_data = pd.read_csv('archive/Train.csv', usecols=[7], skiprows=[0], header=None)
train_label = pd.read_csv('archive/Train.csv', usecols=[6], skiprows=[0], header=None)
test_data = pd.read_csv('archive/Test.csv', usecols=[7], skiprows=[0], header=None)
test_label = pd.read_csv('archive/Test.csv', usecols=[6], skiprows=[0], header=None)
train_data=train_data.values
train_label=train_label.values
test_data=test_data.values
test_label=test_label.values


# In[4]:


def HOG(ImageFile):
    # read image
    img = imread(ImageFile)
    #change the size of image
    resized_img = resize(img, (64,64))
    #generate hog feature
    fd, hog_image = hog(resized_img, orientations=9, pixels_per_cell=(8, 8),
    cells_per_block=(2, 2), visualize=True, multichannel=True)
    return fd


# In[5]:


train_mat=[]
# create training feature matrix
for x in train_data:
    train_mat.append(HOG("archive/"+x[0]))
train_mat=np.array(train_mat)

test_mat=[]
for x in test_data:
    test_mat.append(HOG("archive/"+x[0]))
test_mat=np.array(test_mat)


# In[7]:


from sklearn.svm import SVC
from thundersvm import *
classes = range(0,43)
train_label=train_label.ravel()
clf = SVC(decision_function_shape='ovo')
clf.fit(train_mat, train_label)


# In[19]:


import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
np.set_printoptions(threshold=np.inf)
y_test_pred = clf.predict(test_mat)
ovo_acc = metrics.accuracy_score(y_test_pred,test_label, classes)
cm = confusion_matrix(test_label, y_test_pred)
print("overall accuracy: %f"%(ovo_acc))
print("confusion matrix")
print(cm)
print("===========================================")
acc_for_each_class = metrics.precision_score(test_label,y_test_pred,average=None)
counter=0
print("acc:",acc_for_each_class)
plt.figure(figsize=(15,4))
plt.bar(range(len(classes)), acc_for_each_class,color='r',tick_label=classes)
plt.tight_layout()
plt.show()
print("acc_for_each_class:\n",acc_for_each_class)
print("===========================================")
avg_acc = np.mean(acc_for_each_class)
print("average accuracy:%f"%(avg_acc))


y_train_pred = clf.predict(train_mat)
ovo_acc = metrics.accuracy_score(y_train_pred,train_label, classes)
cm = confusion_matrix(train_label, y_train_pred)
print("overall accuracy(train): %f"%(ovo_acc))
print("confusion matrix")
print(cm)
print("===========================================")
acc_for_each_class = metrics.precision_score(train_label,y_train_pred,average=None)
counter=0
print("acc:",acc_for_each_class)
plt.figure(figsize=(15,4))
plt.bar(range(len(classes)), acc_for_each_class,color='r',tick_label=classes)
plt.tight_layout()
plt.show()
print("acc_for_each_class:\n",acc_for_each_class)
print("===========================================")
avg_acc = np.mean(acc_for_each_class)
print("average accuracy:%f"%(avg_acc))


# In[ ]:




