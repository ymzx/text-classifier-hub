# -*- coding: utf-8 -*-
# @Time    : 2020/1/14
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : split_dataset.py
# @Software: PyCharm
from sklearn import model_selection, preprocessing

def split_dataset(feature, label):
    train_x, test_x, train_y, test_y = model_selection.train_test_split(feature, label, test_size=0.30, random_state=10)
    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    test_y = encoder.fit_transform(test_y)
    return train_x, test_x, train_y, test_y

if __name__ == '__main__':
    trainDF = []
    train_x, test_x, train_y, test_y = split_dataset(trainDF)
    print(train_x, train_y)