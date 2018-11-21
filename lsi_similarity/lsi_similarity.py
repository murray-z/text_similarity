# -*- coding: utf-8 -*-

import jieba
from gensim import corpora
from gensim.models import LsiModel
from sklearn.metrics.pairwise import cosine_similarity


class LsiSimilarity():
    def __init__(self, model='./model/lsi/lsi.model', dictionary='./model/lsi/dictionary.dic'):
        self.lsi = LsiModel.load(model)
        self.dic = corpora.Dictionary.load(dictionary)

    def sentence_to_vector(self, text):
        """
        将文本转换成向量
        :param text:
        :return:
        """
        bow = self.dic.doc2bow(jieba.lcut(text))
        return [v for k, v in self.lsi[bow]]

    def similarity(self, text1, text2):
        vector1 = self.sentence_to_vector(text1)
        vector2 = self.sentence_to_vector(text2)
        return cosine_similarity([vector1], [vector2])[0][0]


if __name__ == '__main__':
    text1 = "小明，你妈妈喊你回家吃饭啦"
    text2 = "小明，回家吃饭啦"
    lsi = LsiSimilarity()
    ret = lsi.similarity(text1, text2)
    print(ret)
