# -*- coding: utf-8 -*-
# @Time    : 2020/1/14
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : config.py
# @Software: PyCharm
import numpy, os, sys, warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # 只输出TF中error
warnings.filterwarnings("ignore") # 忽略python中warnings

base_dir = os.path.dirname(os.path.abspath(sys.argv[0])) # main函数运行父路径

data_path = os.path.join(base_dir, r'data\data.csv')
text_corpus_path = os.path.join(base_dir, r'data\data.txt')

# 词向量的长度
embedding_length = 100

# 0向量
zero_embedding_vector = numpy.zeros((embedding_length))

# 词向量模型['word2vec', 'fasttext']
vectors_model = 'word2vec'

# 是否重新训练词向量
train_word_embedding_flag = False

# 调用词向量路径
word2vec_model_path = os.path.join(base_dir, r'ckpt\word2vec\word2vec.model')
fasttext_model_path = os.path.join(base_dir, r'ckpt\fasttext\fasttext.model')
# 分类器模型参数存放路径
classifier_model_param_path = os.path.join(base_dir, r'ckpt\classifier_model_param')
# 归一化参数文件存放路径
normalization_param_path = os.path.join(base_dir, r'ckpt\normalization_param')

# 需过滤的符号
filters='0123456789“”！《》—’”!"#$ %&()*+,./: ;<=>?@[\]^_`{|}~\t\n'

# 句子间隔符,中文以句号"。"w作为句子结束,英文一般以"."作为句子结束,
# 但是英文特殊场合也会用到".",例如"Mr.",用'|'隔开, '\'为转义字符
sentence_sign = "\.|。"

# 分类算法输入和输出数据维度
input_dim = 200
output_dim = 2

# 指定分类器
# ['GBDT', 'LR', 'AdaBoost', 'LGBM', 'RF', 'XGBoost', 'SVM', 'NB', 'NN'] 可选算法
classifier_list = ['NN']


