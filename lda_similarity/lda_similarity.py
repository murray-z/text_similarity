# -*- coding: utf-8 -*-

import jieba
from gensim import corpora
from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity


class LdaSimilarity():
    def __init__(self, model='./model/lda/lda.model', dictionary='./model/lda/dictionary.dic'):
        self.lda = LdaModel.load(model)
        self.dic = corpora.Dictionary.load(dictionary)

    def sentence_to_vector(self, text):
        """
        将文本转换成向量
        :param text:
        :return:
        """
        bow = self.dic.doc2bow(jieba.lcut(text))
        return [v for k, v in self.lda[bow]]

    def similarity(self, text1, text2):
        vector1 = self.sentence_to_vector(text1)
        vector2 = self.sentence_to_vector(text2)
        return cosine_similarity([vector1], [vector2])[0][0]


if __name__ == '__main__':
    text1 = "小明，你妈妈喊你回家吃饭啦"
    text2 = "回家吃饭啦，小明"
    lda = LdaSimilarity()
    ret = lda.similarity(text1, text2)
    print(ret)
