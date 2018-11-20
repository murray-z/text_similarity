# -*- coding: utf-8 -*-

import jieba

STOPWORDS = '../data/stop_words.txt'


class CountNumSimilarity():
    """
    根据文本相同词汇数目，计算相似性

    similarity = 相同词汇数目/总的词汇数目
    """
    def load_stopwords(self, stopwords_path):
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f]

    def cut_words(self, text, stopwords):
        return [word for word in jieba.lcut(text) if word not in stopwords]

    def similarity(self, text1, text2):
        stopwords = self.load_stopwords(STOPWORDS)
        text1_words = set(self.cut_words(text1, stopwords))
        text2_words = set(self.cut_words(text2, stopwords))

        all_words = list(text1_words | text2_words)

        same_words = list(text1_words & text2_words)

        return len(same_words)*1.0/len(all_words)


if __name__ == '__main__':
    text1 = "小明，你妈妈喊你回家吃饭啦"
    text2 = "回家吃饭啦，小明"
    similarity = CountNumSimilarity()
    print(similarity.similarity(text1, text2))