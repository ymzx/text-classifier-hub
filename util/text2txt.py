# -*- coding: utf-8 -*-
# @Time    : 2020/1/23
# @Author  : JWDUAN
# @Email   : 494056012@qq.com
# @File    : text2txt.py
# @Software: PyCharm

def save2txt(corpus, txt_path):
    with open(txt_path, 'w', encoding='utf-8') as fw:
        for sentences in corpus:
            for sentence in sentences:
                # 去除' ' 和 '\n'以及'\t\n'字符,并将孤立字符作为奇异点剔除
                sentence = sentence.replace('\t', '').replace('\n', '')
                if len(sentence.replace(' ', ''))<=2: continue
                fw.write(sentence.lower()+'\n')

