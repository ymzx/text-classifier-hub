# -*- coding: utf-8 -*-
# @Time    : 2020/1/15
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : main.py
# @Software: PyCharm
from util.load_dataset_and_create_dataframe import load_dataset, create_dataframe
from util.split_dataset import split_dataset
from util.feature_engineering import Vectors
from configs.config import *
from util.data_cleaning import data_clean
from util.data_normalization import data_normalization
from model.classifier_algorithm import classifier_hub
import matplotlib.pyplot as plt
from util.word_embedding_corpus import word_embedding_corpus
from util.text2txt import save2txt
from util.train_words_vectors import run
import numpy as np, time


print('----------------加载原始数据------------------------')
t1 = time.time()
ids, titles, authors, texts, labels = load_dataset(data_path)

print('----------------语料库生成以及词向量训练------------------------')
if train_word_embedding_flag:
    print('生成语料库')
    src_data = [titles, texts]
    text_corpus = word_embedding_corpus(src_data)
    save2txt(text_corpus, text_corpus_path)
    print('词向量训练')
    run(vectors_model) # 词向量训练

print('----------------数据预处理和生成dataFrame数据------------------------')
# 数据预处理
ids, titles, authors, texts, labels = data_clean(ids, titles, authors, texts, labels)
# 生成dataFrame数据
trainDF = create_dataframe(titles, authors, texts, labels)

print('----------------特征构造------------------------')
# 获取词向量特征矩阵，通过词向量求doc向量
vectors = Vectors(vectors_model)
text_vector = vectors.get_docs_vectors_matrix(trainDF['text'])
title_vector = vectors.get_docs_vectors_matrix(trainDF['title'])
# 特征维度拼接, add dimension
docs_vector = np.hstack((text_vector,title_vector))

print('----------------数据归一化------------------------')
normalization = ['MinMaxScaler', 'Normalizer', 'StandardScaler']
docs_vector = data_normalization(docs_vector, normalization[0])

print('----------------训练集和测试集划分------------------------')
X, y = docs_vector, trainDF['label']
x_train, x_test, y_train, y_test = split_dataset(X, y)


print('----------------分类模型调用------------------------')
data = [x_train, x_test, y_train, y_test]
print("分类器", '准确率', '召回率', '正确率', 'F1', 'AUC')
for mode in classifier_list:
    precision, recall, accuracy, f1_score, auc = classifier_hub(data, mode)
    print(mode, precision, recall, accuracy, f1_score, auc)
plt.show()
t2 = time.time()
print('耗时：',round(t2-t1, 3), '秒')






