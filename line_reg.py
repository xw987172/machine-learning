#/usr/local/bin/python3
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
# 加载boston 房屋数据
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
                 header=None,
                 sep='\s+')

df.columns = [
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
]

print(df.head(10))

# EDA 探索模型
import matplotlib.pyplot as plt
import seaborn as sns  # 绘制散点图
sns.set(style='whitegrid', context='notebook')
cols = ['INDUS', 'NOX', 'RM', 'PTRATIO', 'LSTAT', 'MEDV']
sns.pairplot(df[cols], size=2.5)  # 绘图
plt.show()
# 总结：初步观测出MEDV 和 RM之间存在线性关系

# 计算相关性矩阵 np.corrcoef
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt='.2f',
    annot_kws={'size': 15},
    yticklabels=cols,
    xticklabels=cols
)
plt.show()
# 总结：从相关性表格看出MEDV 和 RM有0.7 的相关性， 和LSTAT有-0.74 的相关性

# 接下来运用最小二乘法估计回归曲线的参数， OLS代价函数定为残差平方和Σ(y-y')²
# 用梯度下降，随机梯度下降( GD ) 使得代价最小


class LinearRegressionGD():
    def __init__(self,eta=0.001, niter=20):
        '''

        :param eta: 学习速率
        :param niter: 迭代次数
        '''
        self.eta = eta
        self.n_iter = niter

    def fit(self, X, y):
        self.w_ = np.zeros(1+X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y.T-output)
            data = self.eta*X.T.dot(errors.T)
            self.w_[1:] += data[0][0]
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


X = df[['RM']].values
y = df[['MEDV']].values
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y)
lr = LinearRegressionGD()
lr.fit(X_std, y_std)

plt.plot(range(1, lr.n_iter+1), lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

# 绘制实际值的散点图和拟合曲线


def line_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color="red")
    return None


line_regplot(X_std, y_std, lr)
plt.xlabel('Average number of rooms [RM](standardized)')
plt.ylabel("price in $1000\'s [MEDV](standardized)")
plt.show()

# 预测
num_rooms_std = sc_x.transform([[5.0]])
price_std = lr.predict(num_rooms_std)
print("Price in $1000's: %.3f" % sc_y.inverse_transform(price_std))
