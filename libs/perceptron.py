# coding: utf8
import os
import sys
reload(sys)
sys.setdefaultencoding("utf8")
from pickle import *
import numpy as np


class perceptron(object):
    def __init__(self,eta=0.01,n_iter =10):
        '''
        
        :param eta:float 学习速率 
        :param n_iter: 迭代次数
        '''
        self.eta = eta
        self.n_iter =n_iter

    def fit(self, X,y):
        '''
        
        :param X:[array_like] 
        :param y: [true result]
        :return: 
        '''
        self.w_ = np.zeros(1+X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors =0
            for xi,target in zip(X,y):
                update = self.eta *(target-self.predict(xi))
                self.w_[1:] = update*xi
                self.w_[0] += update
                errors += int(update!=0.0)
            self.errors_.append(errors)
        return self

    def net_input(self,X):
        '''
        计算净输入的值
        :param X: 
        :return: 
        '''
        return np.dot(X,self.w_[1:])+self.w_[0]

    def predict(self,X):
        '''
        获取预测值
        :param X: 
        :return: 1或者-1
        '''
        return np.where(self.net_input(X)>=0.0,1,-1)

if __name__=="__main__":
    import pandas as pd
    df =pd.read_csv('https://archive.ics.uci.edu/ml/'
                    'machine-learning-databases/iris/iris.data',header=None)
    df.tail()