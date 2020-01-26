# text-classifier-hub
*text classifier hub* 一站式文本分类解决方案

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

### 词向量语料库构建中，以段落为样本，还是以句子为样本？(以word2vec为例)
- 以段落为样本
    > Of course, we now know that this was not the case . Comey was actually saying that it was reviewing the emails in light of “an unrelated case”–which we now know to be Anthony Weiner’s sexting with a teenager. But apparently such little things as facts didn’t matter to Chaffetz. The Utah Republican had already vowed to initiate a raft of investigations if Hillary wins–at least two years’ worth, and possibly an entire term’s worth of them. Apparently Chaffetz thought the FBI was already doing his work for him–resulting in a tweet that briefly roiled the nation before cooler heads realized it was a dud. 
- 以句子为样本
    > - Of course, we now know that this was not the case .
    > - Comey was actually saying that it was reviewing the emails in light of “an unrelated case”–which we now know to be Anthony Weiner’s sexting with a teenager.
    > - But apparently such little things as facts didn’t matter to Chaffetz.
    > - The Utah Republican had already vowed to initiate a raft of investigations if Hillary wins–at least two years’ worth, and possibly an entire term’s worth of them.
    > - Apparently Chaffetz thought the FBI was already doing his work for him–resulting in a tweet that briefly roiled the nation before cooler heads realized it was a dud. 

- 以段落为样本，得到不同分类器性能 (语料库词量:16035536)

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

- 以句子为样本，得到不同分类器性能 (语料库词量:16015409)

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

    除了"NN"分类器，采用句子和段落分别构建词向量训练语料库，对分类器性能影响很小，
    但普遍地，以句子为样本略优于段落。为什么段落样本整体差于句子样本？如下，
    > ...was not the case. Comey was actually...
    
    当以段落为样本时，"case"词向量受到
    "Comey"语义关联，实际上，该情境下，"case"和"Comey"语义无关，而以句子为样本时候
    不会出现以上语义干扰现象。为什么两种语料构造方式得到的结果差异很小？那是因为利用word2vec进行词向量时，采用的窗口window=2，
    属于短程记忆，导致以段落为样本时候，也仅有句子和句子间极少部分词语的语义受到干扰，从而对分类器产生小的影响，**综合以上，这里
    采用以句子为样本构造训练词向量的语料库**
    


    





