# Multi-label-classification
环境     python3 bert4keras 0.7.5  

预训练模型     本项目中使用的模型是small electra，对性能要求较低，可以使用其他模型替代  可以在https://github.com/bojone/bert4keras 找到更多信息  

原理      用bert族模型获得句子向量，进行分类微调,损失函数使用sigmoid代替softmax  sigmoid 测量离散分类任务中的概率误差，其中每个类是独立的而不是互斥的。这适用于多标签分类问题。 而softmax类之间是互斥的  预测一般的多元分类是通过argmax实现，返回的是最大的那个数值所在的label_id，因为logits对应每一个label_id都有一个概率。但是，在多标签分类中，我们需要得到的是每一个标签是否可以作为输出标签，所    以每一个标签可以作为输出标签的概率都会量化为一个0到1之间的值。所以当某一个标签对应输出概率小于0.5时，我们认为它不能作为当前句子的输出标签；反之，如果大于等于0.5，那么它代表了当前句子的输出标  签之一
 
 
推荐 https://github.com/hellonlp/classifier_multi_label     
  
