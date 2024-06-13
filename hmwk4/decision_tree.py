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

# 构建模型
model = SimpleDecisionTree()
model.fit(X_train.values, y_train.values)

# 在训练集上评估模型
train_predictions = model.predict(X_train.values)
accuracy = np.mean(train_predictions == y_train.values)
print(f"Train Accuracy: {accuracy:.2f}")

# 对测试集进行预测
test_predictions = model.predict(test_data.values)

# 将预测结果转换为整数形式
test_predictions = [int(prediction) for prediction in test_predictions]

# 保存结果到submission.csv
submission = pd.read_csv(test_file_path)[["PassengerId"]].copy()
submission["Survived"] = test_predictions
submission.to_csv(os.path.join(current_dir, "submission.csv"), index=False)
