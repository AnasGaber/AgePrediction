#Anas Gaber Youssef
#2066017
#Ageprediction-project2


#importing
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
#Reading DataSet
def readDataSet(path):
    print("Reading DataSet From----->"+path)
    file = open(path,"r")
    d=file.read()
    s=d.split('@DATA')
    rdata=s[1]
    lines = []
    for line in enumerate(rdata.split('\n')):
        lines.append(line[1].split(','))
    lines.pop()
    lines.remove(lines[0])
    data=np.array(lines)
    fd=np.asarray(data, dtype=np.float64, order='C')
    random.shuffle(fd)
    file.close()
    return fd

def readDataSetNoSuffle(path):
    print("Reading DataSet From----->"+path)
    file = open(path,"r")
    d=file.read()
    s=d.split('@DATA')
    rdata=s[1]
    lines = []
    for line in enumerate(rdata.split('\n')):
        lines.append(line[1].split(','))
    lines.pop()
    lines.remove(lines[0])
    data=np.array(lines)
    fd=np.asarray(data, dtype=np.float64, order='C')
    file.close()
    return fd


#Reading IrisGeometicFeatures_TrainingSet
path="IrisGeometicFeatures_TrainingSet.txt"
dataset=readDataSet(path)
Gtrfeaturesset=[]
Gtrclassset=[]
for sample in dataset:
    Gtrfeaturesset.append(sample[:5])
    Gtrclassset.append(sample[-1:])
Gtrfeaturesset=np.asarray(Gtrfeaturesset)
Gtrclassset=np.asarray(Gtrclassset)

#Saving DataSet so we have the same suffled dataset for every layer
#IGFTRP=open("Gftrp","wb")
#pickle.dump(Gtrfeaturesset,IGFTRP)
#IGFTRP.close()
#IGCTRP=open("Gctrp","wb")
#pickle.dump(Gtrclassset,IGCTRP)
#IGCTRP.close()

#Loading the saved DataSet
Gtrfeaturesset=pickle.load(open("Gftrp","rb"))
Gtrclassset=pickle.load(open("Gctrp","rb"))
#print(Gtrfeaturesset)
#print(Gtrclassset.flatten()-1)

#Reading IrisTextureFeatures_TrainingSet
path="IrisTextureFeatures_TrainingSet.txt"
dataset=readDataSet(path)
Ttrfeaturesset=[]
Ttrclassset=[]
for sample in dataset:
    Ttrfeaturesset.append(sample[:9600])
    Ttrclassset.append(sample[-1:])
Ttrfeaturesset=np.asarray(Ttrfeaturesset)
Ttrclassset=np.asarray(Ttrclassset)

#Saving DataSet so we have the same suffled dataset for every layer
#ITFTRP=open("Tftrp","wb")
#pickle.dump(Ttrfeaturesset,ITFTRP)
#ITFTRP.close()
#ITCTRP=open("Tctrp","wb")
#pickle.dump(Ttrclassset,ITCTRP)
#ITCTRP.close()

#Loading the saved DataSet
Ttrfeaturesset=pickle.load(open("Tftrp","rb"))
Ttrclassset=pickle.load(open("Tctrp","rb"))
#print(Ttrfeaturesset)
#print(Ttrclassset.flatten()-1)

#Reading IrisTextureFeatures_TestingSet
path="IrisGeometicFeatures_TestingSet.txt"
dataset=readDataSet(path)
Gtsfeaturesset=[]
Gtsclassset=[]
for sample in dataset:
    Gtsfeaturesset.append(sample[:5])
    Gtsclassset.append(sample[-1:])
Gtsfeaturesset=np.asarray(Gtsfeaturesset)
Gtsclassset=np.asarray(Gtsclassset)
#print(Gtsclassset.flatten()-1)


#Saving DataSet so we have the same suffled dataset for every layer
#IGFTSP=open("Gftsp","wb")
#pickle.dump(Gtsfeaturesset,IGFTSP)
#IGFTSP.close()
#IGCTSP=open("Gctsp","wb")
#pickle.dump(Gtsclassset,IGCTSP)
#IGCTSP.close()

#Loading the saved DataSet
Gtsfeaturesset=pickle.load(open("Gftsp","rb"))
Gtsclassset=pickle.load(open("Gctsp","rb"))
#print(Gtrfeaturesset)
#print(Gtrclassset.flatten()-1)

#Reading IrisTextureFeatures_TestingSet
path="IrisTextureFeatures_TestingSet.txt"
dataset=readDataSet(path)
Ttsfeaturesset=[]
Ttsclassset=[]
for sample in dataset:
    Ttsfeaturesset.append(sample[:9600])
    Ttsclassset.append(sample[-1:])
Ttsfeaturesset=np.asarray(Ttsfeaturesset)
Ttsclassset=np.asarray(Ttsclassset)



#Saving DataSet so we have the same suffled dataset for every layer
#ITFTSP=open("Tftsp","wb")
#pickle.dump(Ttsfeaturesset,ITFTSP)
#ITFTSP.close()
#ITCTSP=open("Tctsp","wb")
#pickle.dump(Ttsclassset,ITCTSP)
#ITCTSP.close()

#Loading the saved DataSet
Ttsfeaturesset=pickle.load(open("Tftsp","rb"))
Ttsclassset=pickle.load(open("Tctsp","rb"))
#print(Gtrfeaturesset)
#print(Gtrclassset.flatten()-1)


#Concatenating Geometic With Texture FEATURES-TRANING
#Reading IrisGeometicFeatures_TrainingSet
path1="IrisGeometicFeatures_TrainingSet.txt"
dataset=readDataSetNoSuffle(path1)
Gtrfeaturesset=[]
Gtrclassset=[]
for sample in dataset:
    Gtrfeaturesset.append(sample[:5])
    Gtrclassset.append(sample[-1:])
Gtrfeaturesset=np.asarray(Gtrfeaturesset)
Gtrclassset=np.asarray(Gtrclassset)
#Reading IrisTextureFeatures_TrainingSet
path2="IrisTextureFeatures_TrainingSet.txt"
dataset=readDataSetNoSuffle(path2)
Ttrfeaturesset=[]
Ttrclassset=[]
for sample in dataset:
    Ttrfeaturesset.append(sample[:9600])
    Ttrclassset.append(sample[-1:])
Ttrfeaturesset=np.asarray(Ttrfeaturesset)
Ttrclassset=np.asarray(Ttrclassset)
CCTr=[]
CFTr=[]
CFTrrd=np.concatenate((Ttrfeaturesset,Gtrfeaturesset), axis=1)
CFTrd=np.concatenate((CFTrrd,Ttrclassset), axis=1)
random.shuffle(CFTrd)
for sample in CFTrd:
    CFTr.append(sample[:9605])
    CCTr.append(sample[-1:])

#Saving DataSet so we have the same suffled dataset for every layer
#ICFTRP=open("Cftrp","wb")
#pickle.dump(CFTr,ICFTRP)
#ICFTRP.close()
#ICCTRP=open("Cctrp","wb")
#pickle.dump(CCTr,ICCTRP)
#ICCTRP.close()

#Loading the saved DataSet
CFTr=pickle.load(open("Cftrp","rb"))
CCTr=pickle.load(open("Cctrp","rb"))
#print(np.array(CFTr))
#print(np.array(CCTr).flatten()-1)

#Concatenating Geometic With Texture FEATURES-TESTING
#Reading IrisGeometicFeatures_TrainingSet
path1="IrisGeometicFeatures_TestingSet.txt"
dataset=readDataSetNoSuffle(path1)
Gtsfeaturesset=[]
Gtsclassset=[]
for sample in dataset:
    Gtsfeaturesset.append(sample[:5])
    Gtsclassset.append(sample[-1:])
Gtsfeaturesset=np.asarray(Gtsfeaturesset)
Gtsclassset=np.asarray(Gtsclassset)
#Reading IrisTextureFeatures_TrainingSet
path2="IrisTextureFeatures_TestingSet.txt"
dataset=readDataSetNoSuffle(path2)
Ttsfeaturesset=[]
Ttsclassset=[]
for sample in dataset:
    Ttsfeaturesset.append(sample[:9600])
    Ttsclassset.append(sample[-1:])
Ttsfeaturesset=np.asarray(Ttsfeaturesset)
Ttsclassset=np.asarray(Ttsclassset)
CCTs=[]
CFTs=[]
CFTsrd=np.concatenate((Ttsfeaturesset,Gtsfeaturesset), axis=1)
CFTsd=np.concatenate((CFTsrd,Ttsclassset), axis=1)
random.shuffle(CFTsd)
for sample in CFTsd:
    CFTs.append(sample[:9605])
    CCTs.append(sample[-1:])
#print(np.array(CFTs))
#print(np.array(CCTs).flatten()-1)

#Saving DataSet so we have the same suffled dataset for every layer
#ICFTSP=open("Cftsp","wb")
#pickle.dump(CFTs,ICFTSP)
#ICFTSP.close()
#ICCTSP=open("Cctsp","wb")
#pickle.dump(CCTs,ICCTSP)
#ICCTSP.close()

#Loading the saved DataSet
CFTs=pickle.load(open("Cftsp","rb"))
CCTs=pickle.load(open("Cctsp","rb"))
#print(np.array(CFTs))
#print(np.array(CCTs).flatten()-1)

#Nirmalizing the features sets before fitting them into the model
Gtrfeaturesset=tf.keras.utils.normalize(Gtrfeaturesset, axis = 1)
Ttrfeaturesset=tf.keras.utils.normalize(Ttrfeaturesset, axis = 1)
Gtsfeaturesset=tf.keras.utils.normalize(Gtsfeaturesset, axis = 1)
Ttsfeaturesset=tf.keras.utils.normalize(Ttsfeaturesset, axis = 1)
CFTr=tf.keras.utils.normalize(CFTr, axis = 1)
CFTs=tf.keras.utils.normalize(CFTs, axis = 1)

#Creating the NN Model
#For changing the number of hidden layers pls comment the layer as you wish
model=Sequential()
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dense(3, activation=tf.nn.softmax))

model.compile(loss="sparse_categorical_crossentropy",
optimizer="RMSprop",
metrics=['accuracy'])


#Output
#Please one Opration at a time if you want 1 just uncommend 1 same for the rest

#print("TRAINING WITH GEOMETRIC FEATURES")
#model.fit(Gtrfeaturesset, Gtrclassset.flatten()-1, batch_size=36, epochs=1)#"[1]TRAINING WITH GEOMETRIC FEATURES"
#print("TRAINING WITH TEXTURE FEATURES")
#model.fit(Ttrfeaturesset, Ttrclassset.flatten()-1, batch_size=36, epochs=1)#"[2]TRANING WITH TEXTURE FEATURES"
#print("TRAINING WITH GEOMETRIC AND TEXTURE FEATURES")
#model.fit(np.array(CFTr), np.array(CCTr).flatten()-1, batch_size=36, epochs=1)#"[3]TRAINING WITH GEOMETRIC AND TEXTURE FEATURES"
print("PREDICTION FROM IRIS BIOMETRIC DATA")
#model.fit(Gtrfeaturesset, Gtrclassset.flatten()-1, batch_size=36, epochs=1,validation_data=(Gtsfeaturesset, Gtsclassset.flatten()-1))#"TESTING WITH GEOMETRIC FEATURES"
model.fit(Ttrfeaturesset, Ttrclassset.flatten()-1, batch_size=36, epochs=1,validation_data=(Ttsfeaturesset, Ttsclassset.flatten()-1))#"TESTING WITH TEXTURE FEATURES"
#model.fit(np.array(CFTr), np.array(CCTr).flatten()-1, batch_size=36, epochs=1,validation_data=(np.array(CFTs), np.array(CCTs).flatten()-1))#"TESTING WITH GEOMETRIC AND TEXTURE FEATURES"
tf.keras.backend.clear_session()
