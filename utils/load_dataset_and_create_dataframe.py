# -*- coding: utf-8 -*-
# @Time    : 2020/1/14
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : load_dataset_and_create_dataframe.py
# @Software: PyCharm
import pandas, csv
csv.field_size_limit(500 * 1024 * 1024)
def load_dataset(data_path):
    '''
    :param data_path: csv文件路径
    :return: list
    '''
    ids, titles, authors, texts, labels = [], [], [], [], []
    with open(data_path, newline='', encoding='UTF-8') as csvfile:
        rows = csv.reader(csvfile)
        for i, row in enumerate(rows):
            if i==0: continue
            if row[3]=='': continue
            ids.append(row[0])
            titles.append(row[1])
            authors.append(row[2])
            texts.append(row[3])
            labels.append(row[4])

    return ids, titles, authors, texts, labels

def create_dataframe(titles, authors, texts, labels):
    trainDF = pandas.DataFrame()
    trainDF['author'] = authors
    trainDF['title'] = titles
    trainDF['text'] = texts
    trainDF['label'] = labels
    return trainDF

if __name__ == '__main__':
    data_path = 'data/data.csv'
    '''
    UnicodeDecodeError, you need debug this error！
    '''
    ids, titles, authors, texts, labels = load_dataset(data_path)
    trainDF = create_dataframe(titles, authors, texts, labels)
    print(trainDF)