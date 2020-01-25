# -*- coding: utf-8 -*-
# @Time    : 2020/1/22
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : data_cleaning.py
# @Software: PyCharm
from config. config import filters

def lower(data):
    for i, ele in enumerate(data):
        # ele 为string
        data[i] = ele.lower()
    return data

def delete_pun(data):
    for pun in filters:
        for i, ele in enumerate(data):
            data[i] = ele.replace(pun,'')
    return data

def data_clean(ids, titles, authors, texts, labels):

    # 大写转小写
    titles, authors, texts = lower(titles), lower(authors), lower(texts)

    # 去除标点符号
    titles, authors, texts = delete_pun(titles), delete_pun(authors), delete_pun(texts)

    return ids, titles, authors, texts, labels