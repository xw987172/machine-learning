#coding:utf8
import numpy as np
from numpy.core.numeric import (
    ones, zeros, arange, concatenate, array, asarray, asanyarray, empty,
    empty_like, ndarray, around, floor, ceil, take, dot, where, isscalar, absolute, AxisError
    )
import pandas as pd

# 求相关性--协方差
a = np.corrcoef([1,2,3,4,5],[1,3,6,10,15])
print(a)
# list转array
alist = [1,23,4,5]
aarray = np.asarray(alist)
# 查看矩阵维度
a.ndim

# array 一维转二维
dtype = np.result_type(alist,alist,np.float64) # np.result_type 获取numpy的类型
X= array(aarray, ndmin =2 , dtype = dtype)

# npc.shape 返回行列
ncol,nrow = X.shape

# 获取转置矩阵 npc.T
X = X.T

# 二维矩阵，追加行np.concatenate((X,[2,3213,2312,321]), axis = 0)
X =  np.concatenate((X,[5,6,7,8,9]),axis= 0)