import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score



"""Variables Initialization """

classes=[] # first index of each class on data test

k_init=np.zeros((10,64)) 	# clusters

"""Load data"""

data_dir = "~/Downloads/ROB301"


l = [i for i in range(65)] #Setting labels for the parametres 

"""Load data train"""
fn = 'optdigits.tra'
dataTrain = pd.read_csv( os.path.join(data_dir, fn), names=l,  index_col=-1, usecols=[i for i in range(65)])
labelTrain = pd.read_csv( os.path.join(data_dir, fn) , names=["label"],  usecols=[64])

"""Load data test"""
ft = 'optdigits.tes'
dataTest = pd.read_csv( os.path.join(data_dir, ft) ,  names=l, index_col=-1,  usecols=[i for i in range(65)])
labelTest = pd.read_csv( os.path.join(data_dir, ft) , names=["label"],  usecols=[64])




"""Cluster Initialization:  choosing of a sample of each class to be the cluster center"""

for i in range (0,10):
	x=(np.where(labelTrain==i)[0][0])
	classes.append(x)
	k_init[i]=dataTrain.iloc[x].values

"""KMeans implementation to train data"""

kmeans = KMeans(n_clusters=10, init=k_init)
kmeans.fit(dataTrain.values)

"""KMeans prediction to train data"""

predicted=kmeans.predict(dataTest.values)

"""KMeans metrics calculation"""

"""Confusion matrix"""

print("confusion matrix for trainning data", confusion_matrix(labelTrain, kmeans.labels_))
print("confusion matrix for testing data", confusion_matrix(labelTest, predicted))

"""Score"""

print("Score for trainning data",f1_score(labelTrain, kmeans.labels_, average=None))
print("Score for testing data",f1_score(labelTest, predicted, average=None))

"""Plot of final clusters"""

for i in range (0,10):
	plt.imshow(np.reshape(kmeans.cluster_centers_[i], (8,8)))
	plt.show()

