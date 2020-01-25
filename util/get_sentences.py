# -*- coding: utf-8 -*-
# @Time    : 2020/1/15
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : get.py
# @Software: PyCharm
from configs.config import text_corpus_path
from nltk.corpus import stopwords

class GetSentences(object):

    def __init__(self):
        self.words_file = text_corpus_path

    def get_data(self):
        with open(self.words_file, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
            sentences = []
            for line in lines:
                if not line:continue
                line = self.delete_stop_words(line)
                sentences.append([s for s in line.split() if s != '\n'])
            return sentences

    def delete_stop_words(self, text):
        text = text.lower().split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)
        return text
