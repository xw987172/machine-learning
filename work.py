#coding:utf8
import sys
import os
reload(sys)
sys.setdefaultencoding("utf8")
if __name__=="__main__":
    f_path = "libs"
    if os.path.exists(f_path):
        sys.path.append(f_path)
    '''引入perceptron模块 相当于from perceptron'''
    m = __import__("text")
    '''引入perceptron类  相当于from perceptron import perceptron'''
    c = getattr(m,"A")
    '''实例化类，并准备好执行某个函数'''
    f = getattr(c("xw"),"run")
    f()