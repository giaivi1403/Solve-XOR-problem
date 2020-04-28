import os
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import Activation
from keras.optimizers import SGD
import numpy as np

trainX = np.array([[0,0],[1,1],[0,1],[1,0]])
trainY = np.array([[0],[0],[1],[1]])

model = Sequential()	
model.add(Dense(32, input_dim = 2))
model.add(Activation("relu"))
model.add(Dense(32))
model.add(Activation("relu"))
model.add(Dense(1))
model.add(Activation("sigmoid"))
opt = SGD(lr = 0.001)

model.compile(loss = "mean_squared_error",optimizer = 'adam' , metrics = ["binary_accuracy"])
model.fit(trainX,trainY,batch_size = 32 , epochs = 500,verbose = 2)

'''weight = np.array(model.get_weights())
print(weight)'''
preds = model.predict(trainX)
print(preds)