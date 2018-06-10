import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
from sklearn.cluster import KMeans 
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import (precision_score, recall_score,f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer

traindata1 = pd.read_csv('sam1.csv')
traindata2 = pd.read_csv('sam2.csv')
traindata3 = pd.read_csv('sam3.csv')
traindata4 = pd.read_csv('sam4.csv')

for i in [0,2,4,5]:
    le = preprocessing.LabelEncoder()
    le.fit(traindata1.iloc[:,i])
    #print(le.classes_)
    traindata1.iloc[:,i]=le.transform(traindata1.iloc[:,i])
#print(traindata1.head())
for i in [0,2,4,5]:
    le = preprocessing.LabelEncoder()
    le.fit(traindata2.iloc[:,i])
    #print(le.classes_)
    traindata2.iloc[:,i]=le.transform(traindata2.iloc[:,i])
#print(traindata2.head())
for i in [0,2,4,5]:
    le = preprocessing.LabelEncoder()
    le.fit(traindata3.iloc[:,i])
    #print(le.classes_)
    traindata3.iloc[:,i]=le.transform(traindata3.iloc[:,i])
#print(traindata3.head())
for i in [0,2,4,5]:
    le = preprocessing.LabelEncoder()
    le.fit(traindata4.iloc[:,i])
    #print(le.classes_)
    traindata4.iloc[:,i]=le.transform(traindata4.iloc[:,i])
#print(traindata4.head())


data=pd.concat([traindata1,traindata2,traindata3,traindata4])

model=AgglomerativeClustering(n_clusters=2, linkage='ward')
model.fit(data)
labels=model.labels_

data=data.assign(label=labels)

X=data.iloc[:,0:6]
Y=data.iloc[:,6]
#print(X.head())
#print(Y.head())
scaler = Normalizer().fit(X)
trainX = scaler.transform(X)
traindata = np.array(X)
trainlabel = np.array(Y)
traindata, testdata, trainlabel, testlabel = model_selection.train_test_split(traindata,trainlabel , test_size=0.3)
#print(testdata.shape)
#print(traindata.shape)


model = KNeighborsClassifier()
model.fit(traindata, trainlabel)
print(model)
# make predictions
expected = testlabel
predicted = model.predict(testdata)
#np.savetxt('res/predictedKNN.txt', predicted, fmt='%01d')
# summarize the fit of the model
accuracy = accuracy_score(expected, predicted)
recall = recall_score(expected, predicted, average="binary")
precision = precision_score(expected, predicted , average="binary")
f1 = f1_score(expected, predicted , average="binary")

cm = metrics.confusion_matrix(expected, predicted)
print(cm)
tpr = float(cm[0][0])/np.sum(cm[0])
fpr = float(cm[1][1])/np.sum(cm[1])
print("%.3f" %tpr)
print("%.3f" %fpr)
print("Accuracy")
print("%.3f" %accuracy)
print("precision")
print("%.3f" %precision)
print("recall")
print("%.3f" %recall)
print("f-score")
print("%.3f" %f1)
print("fpr")
print("%.3f" %fpr)
print("tpr")
print("%.3f" %tpr)
print("***************************************************************")

