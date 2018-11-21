# -*- coding: utf-8 -*-

import jieba
from gensim import corpora
from gensim.models import LdaModel
from sklearn.metrics.pairwise import cosine_similarity


class LdaSimilarity():
    def __init__(self, model='./model/lda/lda.model', dictionary='./model/lda/dictionary.dic'):
        self.lda = LdaModel.load(model)
        self.dic = corpora.Dictionary.load(dictionary)

    def sentence_to_bow(self, text):
        """
        将文本转换成向量
        :param text:
        :return:[(id, val), (id, val)]
        """
        bow = self.dic.doc2bow(jieba.lcut(text))
        return self.lda[bow]

    def bow_to_vector(self, bow1, bow2):
        """
        转换数据格式
        :param bow1:
        :param bow2:
        :return: [val_id1, val_id2]
        """
        dic_1 = {}
        dic_2 = {}
        for id, tfidf in bow1:
            dic_1[id] = tfidf

        for id, tfidf in bow2:
            dic_2[id] = tfidf

        all_ids = set(dic_1.keys()) | set(dic_2.keys())

        vector1 = [dic_1[id] if id in dic_1 else 0 for id in all_ids]
        vector2 = [dic_2[id] if id in dic_2 else 0 for id in all_ids]
        return vector1, vector2

    def similarity(self, text1, text2):
        bow_1 = self.sentence_to_bow(text1)
        bow_2 = self.sentence_to_bow(text2)

        vector1, vector2 = self.bow_to_vector(bow_1, bow_2)

        return cosine_similarity([vector1], [vector2])[0][0]


if __name__ == '__main__':
    text1 = "小明，你妈妈喊你回家吃饭啦"
    text2 = "回家吃饭啦，小明"
    lda = LdaSimilarity()
    ret = lda.similarity(text1, text2)
    print(ret)
