# -*- coding: utf-8 -*-

import jieba
import math

STOPWORDS = '../data/stop_words.txt'


class EuclidSimilarity():
    """
    step1: 将文本转换成向量，向量中元素为词频
    step2: 计算两个向量之间的欧氏距离
    """
    def load_stopwords(self, stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def cut_words(self, text, stopwords):
        return [word for word in jieba.lcut(text) if word not in stopwords]

    def euclid_dist(self, vector1, vector2):
        return math.sqrt(sum([(a-b)**2 for a, b in zip(vector1, vector2)]))

    def similarity(self, text1, text2):
        stopwords = self.load_stopwords(STOPWORDS)
        text1_words = set(self.cut_words(text1, stopwords))
        text2_words = set(self.cut_words(text2, stopwords))

        all_words = list(text1_words | text2_words)

        text1_vector = [text1.count(word) for word in all_words]
        text2_vector = [text2.count(word) for word in all_words]

        return 1.0 / (1 + self.euclid_dist(text1_vector, text2_vector))


if __name__ == '__main__':
    text1 = "小明，你妈妈喊你回家吃饭啦"
    text2 = "回家吃饭啦，小明"
    similarity = EuclidSimilarity()
    print(similarity.similarity(text1, text2))

