import scipy.io as sio
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import os
import pandas as pd
from scipy.optimize import minimize
import itertools

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


# 训练SVM模型
svm = NonlinearSVM()
print("开始训练")
svm.fit(x_train, y_train)

# 计算准确率
print("进行准确率计算")
y_val_pred = svm.predict(x_val)
accuracy = accuracy_score(y_val, y_val_pred)
print("准确率:", accuracy)
input('a')
# 作业部分
print("进行作业结果计算与录入")
y_test_pred = svm.predict(x_test)
hmwk_csv = pd.read_csv(os.path.join(current_dir, "submission.csv"), header=0)
hmwk_csv["预测结果"] = y_test_pred

hmwk_csv.to_csv(os.path.join(current_dir, "submission.csv"), index=False)
input()