# text_similarity

> 文本相似性分析

# 目录
- data
    - 存放数据
- cosion_similarity
    - 将文本转换成向量(采用one-hot)，根据向量余弦求文本相似性，余弦值越大，相似性越高。
    
- jaccard_similarity
    - 直接计算两个文本中相同词汇数目与总词汇数目的比值，获得文本相似性。
    
- simhash_similarity
    - 根据simhash算法，求得两文本的海明距离作为其文本相似性，海明距离越大，相似性越低。
    
- edit_distance_similarity
    - 根据编辑距离算法，求得两文本编辑作为其相似性，编辑距离越大，相似性越低。
    
- euclid_similarity
    - 根据欧氏距离计算文本相似性。
    
- manhattan_similarity
    - 根据曼哈顿距离计算文本相似性。
    
- lda_similarity
    - 基于lda对文本进行向量转换，采用cosion进行相似度计算。
    
- lsi_similarity
    - 基于lsi对文本进行向量转换，采用cosion进行相似度计算。
    
- tfidf_similarity
    - 基于tfidf对文本进行向量转换，采用cosion进行相似度计算
        

## gensim模型训练     
[lda, lsi, tfidf模型训练](https://github.com/zhangfazhan/gensim_train_model)