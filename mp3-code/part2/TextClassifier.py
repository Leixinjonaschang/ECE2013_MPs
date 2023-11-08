# TextClassifier.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Dhruv Agarwal (dhruva2@illinois.edu) on 02/21/2019

"""
You should only modify code within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""


class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # 初始化数据结构来存储概率
        self.class_word_counts = {}  # key是类别，value是该类别的单词数
        self.class_counts = {}  # key是类别，value是该类别的文档数
        self.class_priors = {}  # key是类别，value是该类别的先验概率
        self.vocab = set()  # 词汇表
        self.word_probabilities = {}  # key是类别，value是该类别下每个单词的概率

        # 计算每个类别的文档数目
        for label in train_label:  # 遍历每个类别
            if label not in self.class_counts:  # 如果类别不在class_counts中
                self.class_counts[label] = 0  # 初始化为0
            self.class_counts[label] += 1

        # 计算每个类别的先验概率
        total_docs = len(train_set)  # 训练集中文档总数
        for label in self.class_counts:  # 遍历每个类别
            # 该类别先验概率 = 改类别文档书 / 训练集中文档总数
            self.class_priors[label] = self.class_counts[label] / total_docs

        # 计算每个类别中的单词数目
        for text, label in zip(train_set, train_label):
            if label not in self.class_word_counts:  # 如果类别不在class_word_counts中
                self.class_word_counts[label] = {}  # 初始化为{}

            for word in text:  # 遍历每个单词
                self.vocab.add(word)  # 将单词加入词汇表

                if word not in self.class_word_counts[label]:
                    self.class_word_counts[label][word] = 0  # 初始化这个类别下这个单词的数目

                self.class_word_counts[label][word] += 1  # 该类别下该单词数目加1

        # 对每个类别下的单词概率进行拉普拉斯平滑处理
        vocab_size = len(self.vocab)  # 词汇表大小
        for label, word_counts in self.class_word_counts.items():  # label是类别，word_counts是该类别下的单词数目
            self.word_probabilities[label] = {}  # 初始化该类别下的单词概率
            total_words = sum(word_counts.values())  # 该类别下的单词总数

            # 计算该类别下每个单词的概率（条件概率 p(word|label)）
            for word in self.vocab:  # 遍历词汇表中的每个单词
                count = word_counts.get(word, 0)  # 该类别下该单词的数目
                self.word_probabilities[label][word] = (count + 1) / (total_words + vocab_size)

    def predict(self, x_set, dev_label, lambda_mix=0.0):
        """
        :param dev_set: List of list of words corresponding with each text in dev set that we are testing on
              It follows the same format as train_set
        :param dev_label : List of class labels corresponding to each text
        :param lambda_mix : Will be supplied the value you hard code for self.lambda_mixture if you attempt extra credit

        :return:
                accuracy(float): average accuracy value for dev dataset
                result (list) : predicted class for each text
        """

        accuracy = 0.0
        result = []

        # TODO: Write your code here
        pass

        return accuracy, result
