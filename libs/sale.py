# coding = utf8
'''
店铺销量预测
ARIMA时序模型
'''
import pandas as pd

# 参数初始化
discfile = "511.xls"
forecastnum = 5

# 读取数据，指定日期列为指标， Pandas自动识别日期为DateTime格式
data = pd.read_excel(discfile, index_col="日期")

# 绘图
import matplotlib.pyplot as plt

# 用来正常显示中文标签
# plt.rcParams["font.sans-serif"] = ["SimHei"]

# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False
data.plot()
plt.show()

# 自相关图
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(data).show()

# 平稳性检测
from statsmodels.tsa.stattools import adfuller as ADF
print('原始序列的ADF检验结果为：\n',ADF(data["销量"]))
# 返回值依次为adf 、pvalue、usedlag、nobs、critical values、icbest、 regresults、resstore

# 差分后的结果
D_data = data.diff().dropna()
D_data.columns = ["销量差分"]

# 时序图
D_data.plot()
plt.show()