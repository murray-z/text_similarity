# -*- coding: utf-8 -*-

import jieba
import json
from sklearn.feature_extraction.text import TfidfVectorizer


def train_idf(idf_dic='./idf.dic'):
    '''
    训练并保存语料中词语idf信息
    :param idf_dic:
    :return:
    '''
    corpus = ["This is very strange",
              "This is very nice"]
    vectorizer = TfidfVectorizer(
                            use_idf=True, # utiliza o idf como peso, fazendo tf*idf
                            norm=None, # normaliza os vetores
                            smooth_idf=False, #soma 1 ao N e ao ni => idf = ln(N+1 / ni+1)
                            sublinear_tf=False, #tf = 1+ln(tf)
                            binary=False,
                            min_df=1, max_df=1.0, max_features=None,
                            strip_accents='unicode', # retira os acentos
                            ngram_range=(1,1), preprocessor=None,
                            stop_words=None,
                            tokenizer=None,
                            vocabulary=None
                 )
    X = vectorizer.fit_transform(corpus)
    idf = vectorizer.idf_
    idf_dict = dict(zip(vectorizer.get_feature_names(), idf))
    with open(idf_dic, 'w', encoding='utf-8') as f:
        json.dump(idf_dict, f, ensure_ascii=False, indent=4)


class Md25Similarity:
    '''
    md25：求查询语句和文本之间的相似性
    '''
    def __init__(self, idf_path):
        self.idf = self.load_idf(idf_path)

    def load_idf(self, idf_path):
        with open(idf_path, 'r', encoding='utf-8') as f:
            return json.loads(f.read())

    def similarity(self, s1, s2, s_avg=10, k1=2.0, b=0.75):
        bm25 = 0
        s1_list = jieba.lcut(s1)
        for w in s1_list:
            idf_s = self.idf.get(w, 1)
            bm25_ra = s2.count(w) * (k1 + 1)
            bm25_rb = s2.count(w) + k1 * (1 - b + b * len(s2) / s_avg)
            bm25 += idf_s * (bm25_ra / bm25_rb)
        return bm25


if __name__ == '__main__':
    train_idf()
