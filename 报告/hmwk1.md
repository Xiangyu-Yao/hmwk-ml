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
      - [数学基础](#数学基础)
        - [贝叶斯估计](#贝叶斯估计)
        - [最大似然估计](#最大似然估计)
    - [4.代码实现与分析](#4代码实现与分析-1)
      - [4.1 数据集读取与划分](#41-数据集读取与划分)
      - [4.2 定义高斯朴素贝叶斯分类器类](#42-定义高斯朴素贝叶斯分类器类)
      - [4.3 训练模型并评估](#43-训练模型并评估)
      - [4.4 测试模型并保存结果](#44-测试模型并保存结果)
    - [5.运行结果展示](#5运行结果展示)
  - [任务3：支持向量机](#任务3支持向量机)
    - [1.任务描述](#1任务描述-2)
    - [2.数据集描述](#2数据集描述-2)
    - [3.任务分析](#3任务分析-2)
      - [SVM](#svm)
    - [4.代码实现与分析](#4代码实现与分析-2)
      - [4.1 数据读取，预处理与数据集划分](#41-数据读取预处理与数据集划分)
      - [4.2 SVM实现](#42-svm实现)
      - [4.3 训练和评估模型](#43-训练和评估模型)
      - [4.4 作业结果录入](#44-作业结果录入)
    - [5.运行结果展示](#5运行结果展示-1)
  - [任务4：决策树](#任务4决策树)
    - [1.任务描述](#1任务描述-3)
    - [2.数据集描述](#2数据集描述-3)
    - [3.任务分析](#3任务分析-3)
      - [决策树](#决策树)
      - [代码思路分析](#代码思路分析)
    - [4.代码实现和分析](#4代码实现和分析)
      - [4.1 数据加载和预处理](#41-数据加载和预处理)
      - [4.2 决策树实现](#42-决策树实现)
      - [4.3 模型训练和评估](#43-模型训练和评估)
      - [4.4 作业结果录入](#44-作业结果录入-1)
    - [5.结果展示](#5结果展示)



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

在本次任务中，数据集已经提前处理好，留下了需要的特征并存储于csv文件中，只需要读取并做简单的预处理即可。之后自行实现一个高斯贝叶斯分类器，训练并评估即可
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
#### 4.1 数据集读取与划分
使用`sklearn.datasets`中的`load_iris()`加载鸢尾花数据集，将特征数据`X`和目标标签`Y`分离，并利用`sklearn.model_selection`中的`train_test_split`进行训练集和测试集的划分。

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
Y = iris.target

# 划分数据集
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=1919
)

```

#### 4.2 定义高斯朴素贝叶斯分类器类
定义一个高斯朴素贝叶斯分类器作为模型，其中`fit()`方法进行了每个类均值，方差，先验概率的计算以作为预测依据。`_calculate_likehlihood()`方法中计算了类的特征的似然，假设它符合高斯分布，`_calculate_posterior()`方法对输入数据进行后验概率的计算，并且返回概率最高的预测类。最后定义了`predict()`方法接收数据并返回预测结果。

```python
import numpy as np

class GaussianNaiveBayes:
    def fit(self, x, y):
        # 所有target
        self.classes = np.unique(y)

        # 初始化均值，方差，先验概率
        self.mean = np.zeros((len(self.classes), x.shape[1]), dtype=np.float64)
        self.var = np.zeros((len(self.classes), x.shape[1]), dtype=np.float64)
        self.priors = np.zeros(len(self.classes), dtype=np.float64)

        # 计算每个类的均值，方差和先验概率
        for idx, c in enumerate(self.classes):
            x_c = x[y == c]
            self.mean[idx, :] = x_c.mean(axis=0)
            self.var[idx, :] = x_c.var(axis=0)
            self.priors[idx] = x_c.shape[0] / float(x.shape[0])

    def _calculate_likehlihood(self, class_idx, x):
        # 计算给定类下特征的似然（假设似然服从高斯分布)
        mean = self.mean[class_idx]
        var = self.mean[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))  # 高斯分布的分子部分
        denominator = np.sqrt(2 * np.pi * var)  # 分母部分
        return numerator / denominator

    def _calculate_posterior(self, x):
        # 计算后验概率，并返回具有最高后验概率的类
        posteriors = []
        for idx, c in enumerate(self.classes):
            piror = np.log(self.priors[idx])  # 先验概率取对数
            conditional = np.sum(
                np.log(self._calculate_likehlihood(idx, x))
            )  # 条件概率取对数并求和
            posterior = piror + conditional  # 计算后验概率
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]  # 返回后验概率最高的类

    def predict(self, X):
        y_pred = [self._calculate_posterior(x) for x in X]
        return np.array(y_pred)
```

#### 4.3 训练模型并评估
对自定义的高斯朴素贝叶斯分类器进行训练，并用测试集数据对其进行评估，采用`sklearn.metrics`中的`accuracy_score`方法计算评估的accuracy。

```python
from sklearn.metrics import accuracy_score

model = GaussianNaiveBayes()
model.fit(x_train,y_train)
y_pred = model.predict(x_val)
accuracy=accuracy_score(y_val,y_pred)
print(f"模型准确率:{accuracy}")

```

#### 4.4 测试模型并保存结果
用训练好的模型对作业数据进行预测，并保存预测结果到原文件中
```python
import pandas as pd

# 测试环节
test_iris = pd.read_csv("hmwk2/iris_test.csv", header=0)

test_data = np.array(test_iris[test_iris.columns[:4]].values).reshape(-1, 4)
# print(test_data)
pred_test_target = model.predict(test_data)
#print(pred_test_target)
test_iris["species"]=pred_test_target

test_iris.to_csv("hmwk2/homework.csv",index=False)
```

### 5.运行结果展示
最终评估的正确率为`0.96`，根据助教反馈信息，对作业集的正确率为`0.867`
![alt text](image-4.png)

---

## 任务3：支持向量机
### 1.任务描述
基于SVM构建机器学习模型, 对Omniglot中的字符分类
### 2.数据集描述
Omniglot Dataset是全语言文字数据集，包含各种语言的不同字母表，如日语的平假名和片假名，韩语的元音和辅音，最常见的拉丁字母abcd等。Omniglot Dataset共包含50个不同语言的字母表，每个字母表中包含不同的字符，共1623种字符，每种字符由20个不同的人书写。本次实验选取其中200种字符，每种字符的15张图片作为训练集，5张图片作为测试集。
### 3.任务分析
本次任务要求使用SVM对Omniglot数据集中的字符进行分类。
#### SVM
SVM，也就是支持向量机，是一种监督学习模型，用于分类和回归分析。SVM的目标是找到一个能够最好分离不同类别的超平面。
数据中最重要的部分是支持向量，也就是距离目标超平面最近的那些数据点。这些数据对确定超平面有着决定性的作用。
SVM不仅要找到一个超平面，还要使这个超平面与支持向量的间隔最大化以提高模型的泛化能力。
当数据是线性可分的，也就是可以通过一个线性超平面完全分开，那么SVM可以直接找到一个合适的超平面。但是如果数据线性不可分，就要通过核`kernel`来解决这个问题。在本次任务中计划使用RBF核矩阵，也就是高斯核，它具有较强的灵活性和适应性，常用于非线性数据。其公式如下
<div align=center>

$K(x,x')=\exp(-\gamma\|x-x'\|^2)$

</div>

$x$和$x'$是两个样本点，$\|x-x'\|$是两个样本间的欧几里得距离，$\gamma$是用于控制核函数宽度的超参数
在本次任务中，先进行数据读取与预处理，之后自定义一个SVM，由于本次是多类分类任务，而SVM本质是一个二分类模型，因此本次准备采用“一对一”策略以使SVM可以解决多分类问题。
所谓“一对一”是指对于每两个不同的类别都训练一个SVM分类器，最终预测类别由所有的分类器投票决定。这种方式更加准确，但是计算开销比较大
### 4.代码实现与分析
#### 4.1 数据读取，预处理与数据集划分
通过`os.path`中的多个方法确定数据所在的路径以保证在不同环境下程序都可以正常读取数据。由于数据存储在mat文件中，故使用`scipy.io`中的`loadmat`方法进行数据加载。
预处理环节中，先将数据从28*28展平为长784的一维向量，之后根据数据集介绍，手动创建标签1~200，每个标签包含15个样本。最后借助`sklearn.preprocessing`中的`MinMaxScaler`方法进行数据标准化以增强模型训练效果。最后采用`sklearn.model_selection`中的`train_test_split`方法进行数据集划分。

```python
import scipy.io as sio
import numpy as np
import os
import pandas as pd
from  import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 加载数据
current_dir = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(current_dir, "train_data.mat")
test_data_path = os.path.join(current_dir, "test_data.mat")

train_data = sio.loadmat(train_data_path)
test_data = sio.loadmat(test_data_path)

# 将训练数据展开为 28*28 = 784 的一维向量
x_train = train_data["train"].reshape(-1, 28 * 28)
# 创建标签 1-200, 每个标签15个样本
y_train = np.repeat(np.arange(1, 201), 15)
# 将测试数据展开为 28*28 = 784 的一维向量
x_test = test_data["test"].reshape(-1, 28 * 28)

# 数据标准化
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 划分数据集
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.25, stratify=y_train, random_state=1919
)
```
#### 4.2 SVM实现
实现一个一对一的SVM模型，核函数使用RBF。
在`rbf_kernel`方法中进行RBF核矩阵的计算。
`objective`方法计算了SVM的对偶形式，用于在训练过程中通过优化拉格朗日乘子$\alpha$以找到最优的分类决策边界，其公式为

<div align=center>

$ L(\alpha)=\sum_{i=1}^N\alpha_i-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_jK(x_i,x_j)$

</div>

`zerofun`方法定义了约束条件，使得拉格朗日乘子的加权和为0。`fit`方法使用二分类SVM采用一对一策略训练每对类的分类器。`predict`方法用于预测新的数据点的类别，通过投票机制确定最终分类结果
```python
from scipy.optimize import minimize
import itertools
# 实现SVM
class NonlinearSVM:
    def __init__(self, C=1.0, gamma=0.05):
        self.C = C# 正则化参数
        self.gamma = gamma# RBF核参数
        self.classifiers = {}# 存储每对类的分类器

    def rbf_kernel(self, X1, X2):#计算RBF核矩阵
        K = np.exp(-self.gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2)
        return K

    def objective(self, alpha, y, K):  # SVM对偶的目标函数。
        return 0.5 * np.sum(
            alpha * alpha[:, np.newaxis] * y * y[:, np.newaxis] * K
        ) - np.sum(alpha)

    def zerofun(self, alpha, y):  # 约束条件
        return np.dot(alpha, y)

    def fit(self, X, y):
        classes = np.unique(y)
        for i, j in itertools.combinations(classes, 2):# 对每对类别训练一个分类器
            print(f"Training for classes {i} vs {j}")
            idx = np.where((y == i) | (y == j))[0]# 获取属于这两个类别的样本
            X_ij, y_ij = X[idx], y[idx]
            y_ij = np.where(y_ij == i, 1, -1)# 将标签转换为+1和-1

            K = self.rbf_kernel(X_ij, X_ij)# 计算RBF核矩阵

            N = len(y_ij)
            alpha0 = np.zeros(N)# 初始化拉格朗日乘子
            B = [(0, self.C) for _ in range(N)]# 定义alpha的边界
            constraints = {"type": "eq", "fun": lambda alpha: self.zerofun(alpha, y_ij)}# 定义约束条件
            
            # 最小化目标函数
            res = minimize(
                self.objective,
                alpha0,
                args=(y_ij, K),
                bounds=B,
                constraints=constraints,
            )
            alpha = res.x

            sv_idx = np.where(alpha > 1e-5)[0]
            b = np.mean([y_ij[k] - np.sum(alpha * y_ij * K[k]) for k in sv_idx])

            self.classifiers[(i, j)] = (alpha, b, X_ij, y_ij)

    def predict(self, X):
        votes = np.zeros((len(X), len(self.classifiers)))
        for k, ((i, j), (alpha, b, X_train, y_train)) in enumerate(
            self.classifiers.items()
        ):
            print(f"Test for classes {i} vs {j}")
            K = self.rbf_kernel(X_train, X)
            predictions = K.T @ (alpha * y_train) + b
            votes[:, k] = np.where(predictions > 0, i, j)
        y_pred = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(), axis=1, arr=votes
        )
        return y_pred

```

#### 4.3 训练和评估模型
使用自定义的SVM模型的`fit`方法进行模型训练，并计算在测试集上的正确率
```python
from sklearn.metrics import accuracy_score

# 训练SVM模型
svm = NonlinearSVM()
print("开始训练")
svm.fit(x_train, y_train)

# 计算准确率
print("进行准确率计算")
y_val_pred = svm.predict(x_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("准确率:", accuracy)

```

#### 4.4 作业结果录入
使用模型的`predict`方法对作业数据集进行预测，并将预测结果存储回原文件。
```python
# 作业部分
print("进行作业结果计算与录入")
y_test_pred = svm.predict(x_test)
hmwk_csv = pd.read_csv(os.path.join(current_dir, "submission.csv"), header=0)
hmwk_csv["预测结果"] = y_test_pred

hmwk_csv.to_csv(os.path.join(current_dir, "submission.csv"), index=False)
input()

```
### 5.运行结果展示
![alt text](image-5.png)

---

## 任务4：决策树
### 1.任务描述
基于决策树构建机器学习模型，根据乘客的特征预测其在 Titanic 沉船事件中是否幸存。
### 2.数据集描述
Titanic数据集中乘客的特征包含客舱等级、性别、年龄、在 Titanic 号上的同伴/配偶数量、船票编号、票价等。对于每一个乘客都包含了其是否在Titanic灾难中生还的信息（Survived），作为真实值标签。
### 3.任务分析

#### 决策树
决策树是一种监督学习算法，适用于分类和回归任务，其主要思想就是通过一系列的筛选条件，将数据集分割为更小的子集，形成一个类似于树状的结构。每个节点代表一个特征，分支为该特征的可能值，叶子节点则代表分类结果或回归值。
本次考虑建立一个简单的二叉决策树以完成任务，那么就需要找到每次分割的最佳分割点。寻找依据为信息增益，其公式如下

<div align = center>

$IG(Y,X) = H(Y)-\sum_{v\in Values(X)}P(v)·H(Y|X=v)$

</div>

$H(Y)$是分割前目标变量$Y$的熵，$Values(X)$ 是特征$X$的所有可能值.$P(v)$是特征$X$取值$v$的概率。$H(Y∣X=v)$是在特征$X$取值$v$时目标变量 $Y$的条件熵。
熵的公式如下：

<div align = center>

$H(Y)=-\sum_{i=1}^kp_i\log_2(p_i)$

</div>

之后通过方法递归建立完整的决策树。

#### 代码思路分析
本次任务需要基于决策树构建一个机器学习模型，使用Titanic数据集预测乘客在沉船事件中的生还情况。具体来说，以乘客的有效特征来预测他们是否生还，那么这里就需要对原始数据集进行数据填充并根据经验等依据筛选掉无效的特征，以提高决策树的效果。由于年龄特征为连续数据，而决策树更擅长处理离散数据，因此这里还需要将年龄数据分块，将其构造为离散数据。并将字符数据转换为数值数据以便于模型训练。
在处理完数据之后，就需要进行决策树的构建，具体需要的部件上文已经提过，这里不再赘述。
最后就是模型的训练，评估以及测试。
### 4.代码实现和分析
#### 4.1 数据加载和预处理
定义`preprocess_data`方法进行数据的读取和预处理，先通过`pandas`中的`read_csv`方法读取存储在csv文件中的数据，之后进行预处理，根据经验删除一些无关的特征，用中位数填充缺失值，并将字符特征编码为数值特征。最后对年龄进行分块，使其从连续数据变为离散数据，便于模型训练。
利用`os`的相关方法获取数据存储路径，使用定义的`preprocess_data`方法对数据进行预处理。之后分离数据特征和标签便于后续训练和评估
```python
import os
import pandas as pd
import numpy as np

# 读取和预处理数据
def preprocess_data(file_path):
    data = pd.read_csv(file_path)

    # 删除无关特征
    data.drop(columns=["PassengerId", "Name", "Ticket", "Fare", "Cabin"], inplace=True)

    # 填充缺失值
    data["Age"].fillna(data["Age"].mean(), inplace=True)
    data["Embarked"].fillna("S", inplace=True)

    # 编码分类特征
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})
    data["Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    # 分箱处理连续数据Age
    # 如果年龄小于1，设为0-1岁区间；其他年龄分成多个区间
    bins = [0, 1, 12, 20, 40, 60, 80]
    labels = [0, 1, 2, 3, 4, 5]
    data["Age"] = pd.cut(data["Age"], bins=bins, labels=labels, right=False)

    return data

# 定义路径
current_dir = os.path.dirname(os.path.abspath(__file__))
train_file_path = os.path.join(current_dir, "train.csv")
test_file_path = os.path.join(current_dir, "test.csv")

# 预处理数据
train_data = preprocess_data(train_file_path)
test_data = preprocess_data(test_file_path)

# 分离特征和标签
X_train = train_data.drop(columns=["Survived"])
y_train = train_data["Survived"]
```

#### 4.2 决策树实现
实现一个简单的二叉决策树以完成这次任务，`fit`方法用于树的训练，具体的逻辑在`_grow_tree`方法中。预定义了最大深度和最小样本数作为剪枝处理，但由于后续训练效果不错没有用到。之后调用`_best_split`方法寻找最佳分割点。
在该方法中，通过计算特征值的所有阈值和其对应的类别来找出最佳分割点。具体逻辑为对其划分后左右子树的数量进行遍历，并依次通过`_information_gain`计算信息增益。信息增益最大的划分方式即为最佳分割点。
之后根据找到的最佳分割点开始子树的生长，这里是一个递归的过程。
`predict`方法用于训练结束后的预测，逻辑为根据数据进行树查找，判断该数据应该属于哪一个叶节点并给出预测结果
```python
# 决策树实现
class SimpleDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        # 超深度或者样本数小于等于1
        if depth >= self.max_depth or num_samples <= 1:
            return np.round(np.mean(y))  # 取平均值并四舍五入

        # 寻找最佳分割点
        best_feature, best_threshold = self._best_split(X, y, num_samples, num_features)
        if best_feature is None:
            return np.round(np.mean(y))

        indices_left = X[:, best_feature] < best_threshold
        left_subtree = self._grow_tree(X[indices_left], y[indices_left], depth + 1)
        right_subtree = self._grow_tree(X[~indices_left], y[~indices_left], depth + 1)
        return (best_feature, best_threshold, left_subtree, right_subtree)

    def _best_split(self, X, y, num_samples, num_features):
        best_gain = -1
        split_idx, split_threshold = None, None
        # 计算特征值的所有可能阈值及其对应的类别
        for feature_idx in range(num_features):
            thresholds, classes = zip(
                *sorted(zip(X[:, feature_idx], y))
            )  # 对特征值和目标变量排序
            # 初始化左右子树的样本数量
            num_left = [0] * 2
            num_right = [np.sum(y == 0), np.sum(y == 1)]
            for i in range(1, num_samples):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gain = self._information_gain(
                    y, classes, num_left, num_right
                )  # 信息增益
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feature_idx
                    split_threshold = (thresholds[i] + thresholds[i - 1]) / 2
        return split_idx, split_threshold

    def _information_gain(self, y, classes, num_left, num_right):
        p = len(classes)
        p_left = sum(num_left) / p
        p_right = sum(num_right) / p

        if p_left == 0 or p_right == 0:
            return 0

        h = self._entropy(classes)
        h_left = self._entropy(classes[: sum(num_left)])
        h_right = self._entropy(classes[sum(num_left) :])

        return h - (p_left * h_left + p_right * h_right)

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)  # 各类出现概率
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            if inputs[node[0]] < node[1]:
                node = node[2]
            else:
                node = node[3]
        return node

```
#### 4.3 模型训练和评估
调用自定义的决策树模型，并使用处理好的训练数据进行训练。训练完成后直接在训练集上进行模型评估。

```python
# 构建模型
model = SimpleDecisionTree()
model.fit(X_train.values, y_train.values)

# 在训练集上评估模型
train_predictions = model.predict(X_train.values)
accuracy = np.mean(train_predictions == y_train.values)
print(f"Train Accuracy: {accuracy:.2f}")
```
#### 4.4 作业结果录入
对作业集数据进行预测，将预测结果转为整数形式并保存到原文件中
```python
# 对测试集进行预测
test_predictions = model.predict(test_data.values)

# 将预测结果转换为整数形式
test_predictions = [int(prediction) for prediction in test_predictions]

# 保存结果到submission.csv
submission = pd.read_csv(test_file_path)[["PassengerId"]].copy()
submission["Survived"] = test_predictions
submission.to_csv(os.path.join(current_dir, "submission.csv"), index=False)
```
### 5.结果展示
![alt text](image-6.png)
据助教反馈，测试集正确率为0.774
