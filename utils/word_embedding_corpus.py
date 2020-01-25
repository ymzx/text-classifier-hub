# -*- coding: utf-8 -*-
# @Time    : 2020/1/23
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : word_embedding_corpus.py
# @Software: PyCharm
from config.config import sentence_sign
import re

def get_sentence(data):
    sentences = re.split(sentence_sign, data)
    return sentences

def word_embedding_corpus(data):
    '''
    obtain all of corpus
    :param data: list
    :return:
    '''
    data_len = len(data[0])
    doc_sentences = []
    for i in range(data_len):
        sentences = []
        for j, ele in enumerate(data):
            sentences = sentences + get_sentence(ele[i]) # sentences is type of list
        doc_sentences.append(sentences)
    return doc_sentences


