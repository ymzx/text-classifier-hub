# -*- coding: utf-8 -*-
# @Time    : 2020/1/23
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : data_normalization.py
# @Software: PyCharm
from sklearn.preprocessing import MinMaxScaler, Normalizer, StandardScaler
from sklearn.externals import joblib
from config.config import normalization_param_path
import os

def data_normalization(X, normalization=None):
    if not os.path.exists(normalization_param_path):
        os.makedirs(normalization_param_path)
    scaler_filename = os.path.join(normalization_param_path, "scaler.save")
    # 默认'MinMaxScaler'
    if normalization is None: normalization = 'MinMaxScaler'
    # 标准化输入数据
    if normalization == 'MinMaxScaler':
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        joblib.dump(scaler, scaler_filename)  # save
    elif normalization == 'Normalizer':
        scaler = Normalizer()
        scaler.fit(X)
        X = scaler.transform(X)
        joblib.dump(scaler, scaler_filename)
    elif normalization == 'StandardScaler':
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        joblib.dump(scaler, scaler_filename)
    return X

if __name__ == '__main__':
    import numpy as np
    X = np.array([[0.1,0.8],[0.4, 0.2], [-0.1, 0.5]])
    print('原始数据', X)
    X_norm = data_normalization(X, normalization=None)
    print('归一化数据', X_norm)
