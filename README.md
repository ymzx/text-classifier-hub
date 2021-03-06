# text-classifier-hub
*text or data classifier hub* 一站式文本和数据分类解决方案
---------
### Background 
以一个英文假新闻预测为例，提供一站式文本和数据分类解决方案。
- data
    - 路径：data/data.csv
    - 字段说明
        - id：新闻文章的唯一ID
        - title：新闻文章的标题
        - author：新闻文章的作者
        - text：文章的文本；可能不完整
        - label：将文章标记为可能不可靠的标签，1为假新闻, 0为真新闻
-----
### 流程

   ![流程图](https://github.com/ymzx/text-classifier-hub/blob/master/pics/flow/flow.png?raw=true)

---------
### Results
- ROC曲线以及AUC值（以case3为例）

    ![ROC曲线](https://github.com/ymzx/text-classifier-hub/blob/master/pics/sentence_with_delete_stopwords/graph.png?raw=true)
- 较优结果分类器为GBDT，词向量长度为100，去除停用词，详情如下

    | 分类器 | 准确率 | 召回率 | 正确率 | F1 | AUC |
    |:--------:|:--------:|:-------:|:--------:|:----:|:-----:|
    |GBDT |0.95| 0.959| 0.954 |0.955| 0.991|

---- 
### 词向量生成模型 to-do list
- [x] word2vec
- [x] fasttext
- [ ] glove
- [ ] bert

### 分类模型算法 to-do list
- [x] 随机森林 (RF)
- [x] 逻辑回归 (LR)
- [x] 支持向量机 (SVM)
- [x] 梯度提升决策树 (GBDT)
- [x] 极端梯度提升 (XGBoost)
- [x] LightGBM (LGBM) 
- [x] 自适应提升 (AdaBoost)
- [x] 朴素贝叶斯 (NB)
- [x] 神经网络 (NN)
- [ ] 长短期记忆网络 (LSTM)

### 计算性能优化 to-do list
- [ ] 去除停用词计算性能(耗时12分钟，占整体耗时55%)
------
### config.py
- classifier_list：指定分类器
- input_dim 和 output_dim：指定NN分类器的输入和输出维度
- filters：文本中需要过滤的符号
- delete_stop_words_flag：是否去除停用词语
- vectors_model：选择词向量模型
- train_word_embedding_flag：是否重新训练词向量
- zero_embedding_vector：零向量
- base_dir：main.py函数运行路径
- text_corpus_path：词向量训练所需的语料库的路径
- data_path：原始数据路径
- embedding_length：指定词向量长度
- word2vec_model_path 和 fasttext_model_path：生成词向量模型参数路径
- normalization_param_path：归一化参数文件路径
- sentence_sign：句子标识符
- warnings.filterwarnings("ignore")：忽略警告

### 注意
- 改动词向量长度时，需要同时改变NN分类器的输入维度
- 只要词向量语料库改变时，需要重新生成词向量，train_word_embedding_flag = True
- 程序运行，pycharm(2018)环境,run main.py 
- 环境要求，python 3.x, keras, nltk, numpy, sklearn, gensim, pandas, 版本要求较松，建议最新版
--------
### 词向量语料库构建中，以段落为样本，还是以句子为样本？(以word2vec为例)
- 以段落为样本
    > Of course, we now know that this was not the case . Comey was actually saying that it was reviewing the emails in light of “an unrelated case”–which we now know to be Anthony Weiner’s sexting with a teenager. But apparently such little things as facts didn’t matter to Chaffetz. The Utah Republican had already vowed to initiate a raft of investigations if Hillary wins–at least two years’ worth, and possibly an entire term’s worth of them. Apparently Chaffetz thought the FBI was already doing his work for him–resulting in a tweet that briefly roiled the nation before cooler heads realized it was a dud. 
- 以句子为样本
    > - Of course, we now know that this was not the case .
    > - Comey was actually saying that it was reviewing the emails in light of “an unrelated case”–which we now know to be Anthony Weiner’s sexting with a teenager.
    > - But apparently such little things as facts didn’t matter to Chaffetz.
    > - The Utah Republican had already vowed to initiate a raft of investigations if Hillary wins–at least two years’ worth, and possibly an entire term’s worth of them.
    > - Apparently Chaffetz thought the FBI was already doing his work for him–resulting in a tweet that briefly roiled the nation before cooler heads realized it was a dud. 

- case1. 以段落为样本，得到不同分类器性能 (语料库词量:16035536)

    | 分类器 | 准确率 | 召回率 | 正确率 | F1 | AUC |
    |:--------:|:--------:|:-------:|:--------:|:----:|:-----:|
    |GBDT| 0.936| 0.952| 0.944| 0.944| 0.986|
    |LR| 0.888| 0.955| 0.918| 0.921| 0.974|
    |AdaBoost| 0.902| 0.921 |0.91 |0.911 |0.972|
    |LGBM| 0.927| 0.963| 0.944| 0.945| 0.989|
    |RF| 0.898| 0.973| 0.931| 0.934| 0.984|
    |XGBoost| 0.912| 0.957| 0.932| 0.934| 0.984|
    |SVM| 0.85| 0.979| 0.903| 0.91| 0.949|
    |NB| 0.778| 0.818| 0.792| 0.797| 0.861|
    |NN| 0.501| 1.0| 0.503| 0.668| 0.966|

- case2. 以句子为样本，得到不同分类器性能 (语料库词量:16015409)

    | 分类器 | 准确率 | 召回率 | 正确率 | F1 | AUC |
    |:--------:|:--------:|:-------:|:--------:|:----:|:-----:|
    |GBDT| 0.944 |0.951| 0.947 |0.947| 0.987|
    |LR |0.888| 0.95 |0.916 |0.918 |0.974|
    |AdaBoost| 0.906| 0.903| 0.904| 0.904| 0.97|
    |LGBM |0.935| 0.959 |0.946| 0.947 |0.989|
    |RF| 0.89| 0.972| 0.926| 0.929 |0.982|
    |XGBoost |0.91 |0.955 |0.93 |0.932 |0.982|
    |SVM |0.85| 0.981| 0.904| 0.911 |0.951|
    |NB |0.776 |0.835| 0.797 |0.804| 0.87|
    |NN| 0.844| 0.984 |0.902 |0.909 |0.969|

- 结论

    除了"NN"分类器（具有一定的随机性，需要精心调参优化），采用句子和段落分别构建词向量训练语料库，对分类器性能影响很小，
    但普遍地，以句子为样本略优于段落。为什么段落样本整体差于句子样本？如下，
    > ...was not the case. Comey was actually...
    
    当以段落为样本时，"case"词向量受到
    "Comey"语义关联，实际上，该情境下，"case"和"Comey"语义无关，而以句子为样本时候
    不会出现以上语义干扰现象。为什么两种语料构造方式得到的结果差异很小？一种可能的解释是，利用word2vec进行词向量时，采用的窗口window=2，
    属于短程记忆，导致以段落为样本时候，也仅有句子和句子间极少部分词语的语义受到干扰，从而对分类器产生小的影响，**综合以上，这里
    采用以句子为样本构造训练词向量的语料库**

### 停用词 (以word2vec为例，采用句子样本)
- case3. 去除文本中的停用词

    | 分类器 | 准确率 | 召回率 | 正确率 | F1 | AUC |
    |:--------:|:--------:|:-------:|:--------:|:----:|:-----:|
    |GBDT |0.95| 0.959| 0.954 |0.955| 0.991|
    |LR |0.894 |0.959 |0.923 |0.925 |0.974|
    |AdaBoost |0.904| 0.921| 0.912 |0.912 |0.976|
    |LGBM| 0.939| 0.964| 0.95 |0.951 |0.991|
    |RF| 0.898 |0.978| 0.934| 0.937 |0.984|
    |XGBoost |0.916 |0.954| 0.933 |0.935 |0.986|
    |SVM |0.85| 0.98 |0.904 |0.911 |0.954|
    |NB |0.827| 0.868| 0.843 |0.847 |0.908|
    |NN| 0.738| 0.994| 0.821| 0.847| 0.952|
***去除停用词后，分类器性能整体有整体提升，最高正确率为0.954***

### 词向量长度
- case4. case3词向量长度为100，在case3基础上，仅改变词向量长度为50

    | 分类器 | 准确率 | 召回率 | 正确率 | F1 | AUC |
    |:--------:|:--------:|:-------:|:--------:|:----:|:-----:|
    |GBDT| 0.936| 0.95| 0.943| 0.943| 0.984|
    |LR |0.882 |0.941 |0.908 |0.91| 0.967|
    |AdaBoost| 0.913| 0.917| 0.915| 0.915| 0.974|
    |LGBM |0.934 |0.955 |0.944 |0.944 |0.988|
    |RF| 0.905| 0.961| 0.93| 0.932| 0.983|
    |XGBoost |0.918 |0.948 |0.932 |0.933 |0.983|
    |SVM| 0.852| 0.971| 0.901| 0.907| 0.95|
    |NB |0.805 |0.824 |0.813 |0.815 |0.882|
    |NN |0.871 |0.88| 0.875| 0.876| 0.948|
***正确率最大为0.944，对应分类器为LGBM***

### 补充
- NN网络可以精调达到稳定
- 可以优化原始数据处理方式，精简数据预处理流程
- 以上结果只能反映各分类器大致性能，不代表各分类器最优状况，如需更优结果需进一步调整各分类器参数










    





