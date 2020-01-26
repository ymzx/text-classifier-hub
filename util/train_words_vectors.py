# -*- coding: utf-8 -*-
# @Time    : 2020/1/15
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : train_vectors.py
# @Software: PyCharm
from model.word2vec import Word2VecTrain
from model.fasttext import FastTextTrain
from configs.config import vectors_model


def run(model):
    if model == 'word2vec':
        main = Word2VecTrain()
    else:
        main = FastTextTrain()
    main.main()



if __name__ == '__main__':
    # 可传入参数是 fasttext 或者是 word2vec  分别使用不同的词向量类型进行训练
    run(vectors_model)