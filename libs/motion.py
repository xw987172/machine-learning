# coding = utf8
import pyprind
import pandas as pd
import numpy as np
import os
'''
# 可视化进程条
pbar = pyprind.ProgBar(50000)
labels = {'pos': 1, 'neg': 0}
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = "../aclImdb/%s/%s" %(s, l)
        for file in os.listdir(path):
            with open(os.path.join(path, file), 'r',encoding='utf8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()

df.columns = ['review', 'sentiment']

print(df.head(3))
#转存成csv
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))  # 行标全都变成了0
df.to_csv('./movie_comment.csv', index=False)
# 读一下数据看看
# df = pd.read_csv('./movie_comment.csv')
# df.head(3)

'''
#  ## 接下来将单词转换为特征向量；例如根据单词在每个评论中出现的次数
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer()
docs = np.array([
    'the sun is shining',
    'the weather is sweat',
    'the sun is shining and the weather is sweet'
])
bag = count.fit_transform(docs)

from sklearn.feature_extraction.text import TfidfTransformer
tfidf = TfidfTransformer()
np.set_printoptions(precision=2)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
