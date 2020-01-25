# -*- coding: utf-8 -*-
# @Time    : 2020/1/15
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : fasttext.py
# @Software: PyCharm
from model.base import Base
from gensim.models import FastText
from configs.config import embedding_length

class FastTextTrain(Base):
    def main(self):
        print('使用fasttext 方式进行训练, start train...')
        model = FastText(sentences=self.data, sg= 1, size= embedding_length, window =2, min_count=1)
        model.save(self.fasttext_model)
        print('模型训练完成...')