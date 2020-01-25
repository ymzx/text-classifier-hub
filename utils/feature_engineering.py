# -*- coding: utf-8 -*-
# @Time    : 2020/1/14
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : feature_engineering.py
# @Software: PyCharm
'''
Word Embeddings:  Glove, FastText, and Word2Vec
1.Loading the pretrained word embeddings
2.Creating a tokenizer object
3.Transforming text documents to sequence of tokens and pad them
4.Create a mapping of token and their respective embeddings

we use fastText here, link is https://fasttext.cc/docs/en/english-vectors.html
download wiki-news-300d-1M.vec.zip
'''
from keras.preprocessing import text, sequence
import numpy
from gensim.models import word2vec
from gensim.models import FastText
from model.base import  Base
from config.config import embedding_length, vectors_model, filters, zero_embedding_vector

class Vectors(Base):
    def __init__(self, vectors_model):
        super(Vectors, self).__init__()
        self.vectors_model = vectors_model
        if self.vectors_model == 'word2vec':
           self.model = word2vec.Word2Vec.load(self.word2vec_model)
        else:
           self.model = FastText.load(self.fasttext_model)
        self.word_index = None

    def get_words_vectors(self, word):
        try:
            word_vec = self.model[word]
        except KeyError:
            word_vec = zero_embedding_vector
            # print('词 %s 无对应的词向量, 使用0向量代替'%word)
        return word_vec

    def get_words_embedding_matrix(self, data):
        # create a tokenizer
        token = text.Tokenizer(filters = filters, lower=True)
        token.fit_on_texts(data)
        self.word_index = token.word_index # 保存所有word对应的编号id，从1开始
        '''
        word_counts:字典，将单词（字符串）映射为它们在训练期间出现的次数。仅在调用fit_on_texts之后设置。
        print( tokenizer.word_counts) #[('some', 2), ('thing', 2), ('to', 2), ('eat', 1), ('drink', 1)] 
        word_index: 字典，将单词（字符串）映射为它们的排名或者索引。仅在调用fit_on_texts之后设置
        print( tokenizer.word_index) #{'some': 1, 'thing': 2,'to': 3 ','eat': 4, drink': 5}
        word_docs: 字典，将单词（字符串）映射为它们在训练期间所出现的文档或文本的数量。仅在调用fit_on_texts之后设置。
        print( tokenizer.word_docs) #{'some': 2, 'thing': 2, 'to': 2, 'drink': 1,  'eat': 1}
        '''

        # create token-embedding mapping
        embedding_matrix = numpy.zeros((len(self.word_index) + 1, embedding_length))
        for word, i in self.word_index.items():
            embedding_vector = self.get_words_vectors(word)
            embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def get_docs_vectors_matrix(self, data):
        self.embedding_matrix = self.get_words_embedding_matrix(data)
        docs_vector_matrix = numpy.zeros((len(data), embedding_length))
        for i, doc in enumerate(data):
            doc = doc.split()
            word_vector_matrix = numpy.zeros((len(doc), embedding_length))
            for j, word in enumerate(doc):
                try:
                    word_vector_matrix[j] = self.embedding_matrix[self.word_index[word]]
                except:
                    print('新词发现：', word)
                    word_vector_matrix[j] = zero_embedding_vector
            # 获取doc vector, 对doc中各word求和
            docs_vector_matrix[i] = word_vector_matrix.sum(axis=0)
        return docs_vector_matrix


