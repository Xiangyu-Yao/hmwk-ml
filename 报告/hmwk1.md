# 模式识别与机器学习报告
| 姓名   | 学号       | 班级     |
| ------ | ---------- | -------- |
| 姚翔宇 | 2021302474 | 10042101 |

</div>  

- [模式识别与机器学习报告](#模式识别与机器学习报告)
  - [任务1：多项式回归](#任务1多项式回归)
    - [1.任务描述](#1任务描述)
    - [2.数据集描述](#2数据集描述)
    - [3.任务分析](#3任务分析)
      - [普通最小二乘法](#普通最小二乘法)
    - [4.代码实现与分析](#4代码实现与分析)
      - [4.1 数据集读取与数据预处理](#41-数据集读取与数据预处理)
      - [4.2 模型训练](#42-模型训练)
      - [4.3 模型评价与预测](#43-模型评价与预测)
      - [4.4 结果可视化](#44-结果可视化)
    - [5.测试结果展示](#5测试结果展示)
      - [5.1 不同阶数下的MSE](#51-不同阶数下的mse)
      - [5.2 不同阶数下的图像曲线以及表达式](#52-不同阶数下的图像曲线以及表达式)
  - [任务2：概率分类法](#任务2概率分类法)
    - [1.任务描述](#1任务描述-1)
    - [2.数据集描述](#2数据集描述-1)
    - [3.任务分析](#3任务分析-1)
- [pass](#pass)
      - [数学基础](#数学基础)
        - [贝叶斯估计](#贝叶斯估计)
        - [最大似然估计](#最大似然估计)
    - [4.代码实现与分析](#4代码实现与分析-1)
    - [5.运行结果展示](#5运行结果展示)



---

## 任务1：多项式回归

### 1.任务描述
多项式回归是一种回归分析形式，在这种形式中，自变量 ( x ) 和因变量 ( y ) 之间的关系被建模为 ( n ) 阶多项式。使用机器学习的方法来创建一个多项式回归模型，该模型可以根据给定的数据集预测结果。数据集由自变量 ( x ) 和因变量 ( y ) 组成，你的任务是找到一个多项式，能最好地描述 ( x ) 和 ( y ) 之间的关系。

### 2.数据集描述
本实验选取数据集包含125个样本点，每个样本具有一个自变量( x )和一个因变量( y )。数据集根据4:1的比例划分为训练集和测试集。

### 3.任务分析
多项式回归是一种回归分析方式，通过多项式函数来拟合数据集，这种方法适用于数据和一个或多个非线性关系的情况。多项式回归模型的形式如下：
<div align=center>

$$
y = \beta_0+\beta_1x+\beta_2x^2+\beta_3x^3+...+\beta_nx^n+\epsilon 
$$

</div>

在多项式回归模型中，自变量的不同次方$x^2$,$x^3$,...$x^n$被作为新特征加入到模型中。通过训练数据，使用普通最小二乘法等算法来估计回归系数$\{ \beta_0,\beta_1,\beta_2...\beta_n\}$,从而捕捉数据中的非线性关系。

#### 普通最小二乘法
普通最小二乘法是估计多项式回归模型系数的常用方法，其目标是最小化实际值与预测值之间的误差平方和：
<div align=center>

$RSS = \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$

m为样本数，$y_i$ 为实际值, $\hat{y}_i$为预测值。通过最小化RSS，就可以找到最佳的回归系数$\beta_0,\beta_1,\beta_2,...,\beta_n$。

</div>

### 4.代码实现与分析
#### 4.1 数据集读取与数据预处理

数据集存储在`homwework.csv`文件中，使用`numpy`库的`loadtxt`方法来读取数据。之后利用`reshape`增加一个维度以方便模型训练。

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

train_data = np.loadtxt(
    open("作业一/poly_reg/train_dataset.csv", "rb"), delimiter=",", skiprows=1
)
test_data = np.loadtxt(
    open("作业一/poly_reg/test_dataset.csv", "rb"), delimiter=",", skiprows=1
)
new_test_data = pd.read_csv('作业一/new_test.csv',header=0)


x_train = train_data[:, 0].reshape(-1, 1)
y_train = train_data[:, 1].reshape(-1, 1)
x_test = test_data[:, 0].reshape(-1, 1)
y_test = test_data[:, 1].reshape(-1, 1)
new_x_test=np.array(new_test_data['x'].values).reshape(-1,1)
```
接着设置多项式阶数，并采用`sklearn.preprocessing`中的`PolynomialFeatures`方法对自变量进行特征拓展。
```python
degree = 4

# 多项式特征
poly_features = PolynomialFeatures(degree=degree)
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.transform(x_test)
new_x_test_poly=poly_features.transform(new_x_test)
```

#### 4.2 模型训练
采用`sklearn.linear_model`中的`LinearRegression`作为模型，通过模型的`fit()`方法进行训练

```python
# 模型训练
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train_poly, y_train)

```
#### 4.3 模型评价与预测
使用测试集对模型进行评价，并对作业数据集进行预测，通过`sklearn.metrics`中的`mean_squared_error`计算测试集的MSE，将作业数据集的预测结果直接存储回原文件
```python
from sklearn.metrics import mean_squared_error

#测试
train_predictions = model.predict(x_train_poly)
test_predictions = model.predict(x_test_poly)

#方差
train_mse= mean_squared_error(y_train,train_predictions)
test_mse = mean_squared_error(y_test,test_predictions)

print("训练集方差:",train_mse)
print("测试集方差",test_mse)

#作业提交部分
new_test_prediction=model.predict(new_x_test_poly)
#print(new_test_prediction)
new_test_data['y']=new_test_prediction
new_test_data.to_csv("作业一/homework.csv",index=False)
```

#### 4.4 结果可视化
为了能够更加直观的观察模型训练效果，采用`matplotlib.pyplot`进行图像绘制，并且通过格式化输出来输出最终的回归曲线表达式。
```python
import matplotlib.pyplot as plt

# 绘图
plt.scatter(x_train, y_train, color='blue', label='Train data')
plt.scatter(x_test, y_test, color='red', label='Test data')

# 为了绘制平滑曲线，生成许多点来预测
x_range = np.linspace(x_train.min(), x_train.max(), 500).reshape(-1, 1)
x_range_poly = poly_features.transform(x_range)
y_range_pred = model.predict(x_range_poly)

# 输出多项式表达式（仅包含非零系数的项）
coefficients = model.coef_[0]
intercept = model.intercept_[0]

# 构建多项式表达式字符串
poly_expression = f"f(x) = {intercept:.2f}"
for i in range(1, degree + 1):
    if coefficients[i] != 0:
        poly_expression += f" + {coefficients[i]:.2f}x^{i}"

print("多项式回归模型表达式：", poly_expression)

#绘图
plt.plot(x_range, y_range_pred, color='green', label=f'Polynomial Degree {degree}')
plt.title(poly_expression)
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.savefig("作业一/png_homework.png")
plt.show()
```


### 5.测试结果展示
#### 5.1 不同阶数下的MSE
degree=2
![alt text](image.png)
degree=4
![alt text](image-1.png)
degree=6
![alt text](image-2.png)
degree=8
![alt text](image-3.png)

#### 5.2 不同阶数下的图像曲线以及表达式
degree=2
![alt text](img_1.png)
degree=4
![alt text](img_2.png)
degree=6
![alt text](img_3.png)
degree=8
![alt text](img_4.png)

---

## 任务2：概率分类法
### 1.任务描述
使用贝叶斯估计或MLE（最大似然估计），来预测鸢尾花数据集中花的种类。
### 2.数据集描述
鸢尾花数据集是统计学和机器学习中用于分类的经典数据集。该数据集包含了三种不同的鸢尾花：Setosa、Versicolor和Virginica，每种各50个样本。每个样本有四个属性：萼片长度、萼片宽度、花瓣长度和花瓣宽度，所有的测量单位都是厘米。数据集根据4:1的比例划分为训练集和测试集。概率分类法是一种基于概率理论的方法，适合处理此类分类问题。
### 3.任务分析
贝叶斯估计和MLE是统计学中常用的两种方法，用于参数估计与分类任务。其中，贝叶斯估计利用先验知识和数据来进行参数估计，MLE则是通过最大化似然参数来估计参数。这两种方法都适用于分类问题，如本次的鸢尾花数据集分类。
# pass
#### 数学基础
##### 贝叶斯估计
贝叶斯估计通过结合先验分布和似然函数来估计参数。贝叶斯公式如下：
<div align = center>

$P(\theta|X)=\frac{P(X|\theta)P(\theta)}{P(X)}$

</div>  

其中，$P(\theta|X)$ 是给定数据 $X$ 后参数 $\theta$ 的后验概率；$P(X|\theta)$ 是给定参数 $\theta$ 后数据 $X$ 的似然函数；$P(\theta)$ 是参数 $\theta$ 的概率；$P(X)$ 是数据X的概率
##### 最大似然估计
最大似然估计通过最大化似然函数来估计参数。似然函数表示给定参数 $\theta$ 时数据出现的概率。最大似然估计就是要找到使得似然函数最大的参数 $\theta$. 似然函数如下
<div align=center>

$ 𝐿(𝜃|𝑋)=𝑃(𝑋|𝜃)$

</div>

### 4.代码实现与分析
### 5.运行结果展示