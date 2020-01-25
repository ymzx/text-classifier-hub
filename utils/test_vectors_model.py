# -*- coding: utf-8 -*-
# @Time    : 2020/1/15
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : test_vectors_model.py
# @Software: PyCharm
import os
from gensim.models import word2vec
from gensim.models import FastText
from model.base import  Base

class Test(Base):
    def __init__(self, model_path):
        super(Test, self).__init__()
        self.model_path = model_path
        if self.model_path == 'word2vec':
           self.model = word2vec.Word2Vec.load(self.word2vec_model)
        else:
           self.model = FastText.load(self.fasttext_model)

    def get_vectors(self, word):
        sentence_vec = self.model[word]
        return sentence_vec

    def words_similarity(self,word1, word2, mode):
        # cosine distance
        if mode == 'cosine':
            similarity = self.model.wv.similarity(word1, word2)
        else:
            # euclidean distance
            similarity = self.model.wv.distance(word1, word2)
        return similarity

    def words_similar_by_words(self, words, topn):
        similar_words = self.model.wv.similar_by_word(words, topn=topn)
        return similar_words

    def words_doesnt_match(self, words):
        doesnt_match_word = self.model.wv.doesnt_match(words)
        return doesnt_match_word


if __name__ == '__main__':
    obj = Test('word2vec')
    vectors = obj.get_vectors('House')
    print(len(vectors), vectors)