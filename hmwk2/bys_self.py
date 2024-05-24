from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris = load_iris()
X = iris.data
Y = iris.target

# 划分数据集
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=1919
)


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


model = GaussianNaiveBayes()

model.fit(x_train,y_train)

y_pred = model.predict(x_val)

accuracy=accuracy_score(y_val,y_pred)

print(f"模型准确率:{accuracy}")


# 测试环节
test_iris = pd.read_csv("hmwk2/iris_test.csv", header=0)

test_data = np.array(test_iris[test_iris.columns[:4]].values).reshape(-1, 4)
# print(test_data)
pred_test_target = model.predict(test_data)
#print(pred_test_target)
test_iris["species"]=pred_test_target

test_iris.to_csv("hmwk2/homework.csv",index=False)
