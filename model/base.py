# -*- coding: utf-8 -*-
# @Time    : 2020/1/15
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : base.py
# @Software: PyCharm
from utils.get_sentences import GetSentences
from config.config import word2vec_model_path, fasttext_model_path

class Base(object):
    def __init__(self):
        sObj = GetSentences()
        self.data = sObj.get_data()
        self.word2vec_model = word2vec_model_path
        self.fasttext_model = fasttext_model_path
        self.data_len()


    def data_len(self):
        lens = [len(words) for words in self.data]
        print('词总数:{}'.format(sum(lens)))