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

### 词向量语料库构建中，以段落为样本，还是以句子为样本？
- 以段落为样本
    > Of course, we now know that this was not the case . Comey was actually saying that it was reviewing the emails in light of “an unrelated case”–which we now know to be Anthony Weiner’s sexting with a teenager. But apparently such little things as facts didn’t matter to Chaffetz. The Utah Republican had already vowed to initiate a raft of investigations if Hillary wins–at least two years’ worth, and possibly an entire term’s worth of them. Apparently Chaffetz thought the FBI was already doing his work for him–resulting in a tweet that briefly roiled the nation before cooler heads realized it was a dud. 
- 以句子为样本
    > - Of course, we now know that this was not the case .
    > - Comey was actually saying that it was reviewing the emails in light of “an unrelated case”–which we now know to be Anthony Weiner’s sexting with a teenager.
    > - But apparently such little things as facts didn’t matter to Chaffetz.
    > - The Utah Republican had already vowed to initiate a raft of investigations if Hillary wins–at least two years’ worth, and possibly an entire term’s worth of them.
    > - Apparently Chaffetz thought the FBI was already doing his work for him–resulting in a tweet that briefly roiled the nation before cooler heads realized it was a dud. 

- 以段落为样本，得到不同分类器性能

    | 分类器 | 准确率 | 召回率 | 正确率 | F1 | AUC |
    |:--------:|:--------:|:-------:|:--------:|:----:|:-----:|
    GBDT| 0.936| 0.952| 0.944| 0.944| 0.986|
    LR| 0.888| 0.955| 0.918| 0.921| 0.974|
    AdaBoost| 0.902| 0.921 |0.91 |0.911 |0.972|
    LGBM| 0.927| 0.963| 0.944| 0.945| 0.989|
    RF| 0.898| 0.973| 0.931| 0.934| 0.984|
    XGBoost| 0.912| 0.957| 0.932| 0.934| 0.984|
    SVM| 0.85| 0.979| 0.903| 0.91| 0.949|
    NB| 0.778| 0.818| 0.792| 0.797| 0.861|
    NN| 0.501| 1.0| 0.503| 0.668| 0.966|





