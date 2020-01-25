# -*- coding: utf-8 -*-
# @Time    : 2020/1/25
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : neural_net.py
# @Software: PyCharm
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

def NN(input_dim=100, output_dim=2):
    '''Neural network with 3 hidden layers'''
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='relu', kernel_initializer='normal'))
    model.add(Dense(output_dim, activation="softmax", kernel_initializer='normal'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model