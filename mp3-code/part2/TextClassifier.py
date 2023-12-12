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
import math
import matplotlib.pyplot as plt


class TextClassifier(object):
    def __init__(self):
        """Implementation of Naive Bayes for multiclass classification

        :param lambda_mixture - (Extra Credit) This param controls the proportion of contribution of Bigram
        and Unigram model in the mixture model. Hard Code the value you find to be most suitable for your model
        """
        self.lambda_mixture = 0.0

        # 训练部分 初始化数据结构来存储概率
        self.class_word_counts = {}  # key是类别，value是该类别的单词数
        self.class_counts = {}  # key是类别，value是该类别的文档数
        self.class_priors = {}  # key是类别，value是该类别的先验概率
        self.vocab = set()  # 词汇表
        self.word_probabilities = {}  # key是类别，value是该类别下每个单词的概率

        # 预测部分 初始化数据结构来存储预测结果
        self.pred_actual = []  # 元素为 tuple (predicted_label, actual_label)
        self.test_class_counts = {}  # key是类别，value是该类别的文档数

    def fit(self, train_set, train_label):
        """
        :param train_set - List of list of words corresponding with each text
            example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
            Then train_set := [['i','like','pie'], ['i','like','cake']]

        :param train_labels - List of labels corresponding with train_set
            example: Suppose I had two texts, first one was class 0 and second one was class 1.
            Then train_labels := [0,1]
        """

        # 计算每个类别的文档数目
        for label in train_label:
            if label not in self.class_counts:
                self.class_counts[label] = 0
            self.class_counts[label] += 1

        # 计算每个类别的先验概率
        total_docs = len(train_set)
        for label in self.class_counts:
            # 该类别先验概率 = 改类别文档书 / 训练集中文档总数
            self.class_priors[label] = self.class_counts[label] / total_docs

        # 计算每个类别中的单词数目
        for text, label in zip(train_set, train_label):
            if label not in self.class_word_counts:
                self.class_word_counts[label] = {}

            for word in text:
                self.vocab.add(word)

                if word not in self.class_word_counts[label]:
                    self.class_word_counts[label][word] = 0

                self.class_word_counts[label][word] += 1

        # 对每个类别下的单词概率进行拉普拉斯平滑处理
        vocab_size = len(self.vocab)
        for label, word_counts in self.class_word_counts.items():
            self.word_probabilities[label] = {}
            total_words = sum(word_counts.values())

            # 计算该类别下每个单词的概率（条件概率 p(word|label)）
            for word in self.vocab:
                count = word_counts.get(word, 0)
                self.word_probabilities[label][word] = (count + 1) / (total_words + vocab_size)

        # 保存每个类别概率最大的前20个features
        num_top_features = 20
        top_feature_file_name = 'utils/top_20_features/top_features.txt'
        for label in range(1, 15):
            with open(top_feature_file_name, 'a') as top_feature_file:
                top_feature_file.write('Class ' + str(label) + ':\n')
                top_features = self.get_top_features(label, num_top_features)
                for feature in top_features:
                    top_feature_file.write(feature[0] + ' ' + str(feature[1]) + '\n')
                top_feature_file.write('\n')

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

        correct_predictions = 0
        for doc in x_set:
            # 计算每个类别的概率
            class_probabilities = self.calc_doc_probabilities(doc)

            # 计算预测概率最高的类别
            predicted_class = max(class_probabilities, key=class_probabilities.get)
            result.append(predicted_class)

            self.pred_actual.append((predicted_class, dev_label[x_set.index(doc)]))  # for confusion matrix making

            # 统计预测正确的次数
            correct_predictions += 1 if predicted_class == dev_label[x_set.index(doc)] else 0

        accuracy += correct_predictions / len(x_set)
        self.plot_confusion_matrix(dev_label)

        return accuracy, result

    # def calc_doc_probabilities(self, doc):  # with the class priors included in the calculation
    #     """
    #     :param text: list of words in a text
    #     :return: dictionary of probabilities for each class
    #     """
    #     class_probabilities = {}
    #
    #     for class_label in self.word_probabilities:
    #         class_probabilities[class_label] = self.class_priors[class_label]
    #
    #         for word in doc:
    #             if word in self.vocab:
    #                 class_probabilities[class_label] *= self.word_probabilities[class_label].get(word, 1.0 / (
    #                         sum(self.class_word_counts[class_label].values()) + len(self.vocab)))
    #
    #         class_probabilities[class_label] = math.log(class_probabilities[class_label])
    #
    #     return class_probabilities

    def calc_doc_probabilities(self, doc):  # without class priors included in the calculation
        class_probabilities = {}

        for class_label in self.word_probabilities:
            class_probabilities[class_label] = 1

            for word in doc:
                if word in self.vocab:
                    class_probabilities[class_label] *= self.word_probabilities[class_label].get(word, 1.0 / (
                            sum(self.class_word_counts[class_label].values()) + len(self.vocab)))

            class_probabilities[class_label] = math.log(class_probabilities[class_label])

        return class_probabilities

    def plot_confusion_matrix(self, dev_label):
        """
        :return: None, but saves a confusion matrix plot
        """
        confusion_mat = [[0 for _ in range(14)] for _ in range(14)]
        for pred_label, actual_label in self.pred_actual:
            confusion_mat[pred_label - 1][actual_label - 1] += 1

        # 统计测试集中各类别的数量
        for label in dev_label:  # 遍历每个类别
            if label not in self.test_class_counts:  # 如果类别不在class_counts中
                self.test_class_counts[label] = 0  # 初始化为0
            self.test_class_counts[label] += 1

        # 把预测成功次数转为百分比
        for i in range(14):
            for j in range(14):
                confusion_mat[i][j] = confusion_mat[i][j] / self.test_class_counts[i + 1]

        plt.imshow(confusion_mat, cmap='plasma', interpolation='nearest')
        plt.xlabel('Predicted Class')
        plt.ylabel('Actual Class')
        plt.title('Confusion Matrix')
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                   ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"])
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                   ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"])
        plt.colorbar()
        plt.gca().invert_yaxis()  # invert y axis
        plt.savefig('utils/confusion_matrix.png')
        plt.show()

    def get_top_features(self, label, num_top_features):
        # 找出每个类别下概率最大的前num_top_features个单词
        top_features = []
        for word in self.word_probabilities[label]:
            top_features.append((word, self.word_probabilities[label][word]))
        top_features.sort(key=lambda x: x[1], reverse=True)

        return top_features[:num_top_features]
