# text_similarity

> 文本相似性分析

- 目录
    - data
        - 存放数据
    - cosion_similarity
        - 将文本转换成向量，根据余弦求文本相似性。
        
    - count_num_similarity
        - 直接计算两个文本中相同词汇数目与总词汇数目的比值，获得文本相似性。
        
    - simhash_similarity
        - 根据simhash算法，求得两文本的海明距离作为其文本相似性。