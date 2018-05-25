#machine learning
##2018-01-23学习机器学习第二章节**感知器算法**，一种分类算法，对应文件为perceptron.py
反射调用法

~~~
    m = __import__("perceptron")
    '''引入perceptron类  相当于from perceptron import perceptron'''
    c = getattr(m,"perceptron")
    '''实例化类，并准备好执行某个函数'''
    f = getattr(c(),"fit")#可在c()中加入参数
    f()
~~~

#### timer 时序模型