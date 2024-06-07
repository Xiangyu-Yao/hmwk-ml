import numpy as np
import pandas as pd
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
tree_path = os.path.join(current_dir, "decisionTree.npy")


# 读取数据以及预处理
def get_data():
    train_data = pd.read_csv(os.path.join(current_dir, "train.csv"))
    test_data = pd.read_csv(os.path.join(current_dir, "test.csv"))
    # 删除不必要特征
    train_data = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)
    test_data = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

    # 填充可能有的缺失值
    train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
    train_data["Fare"] = train_data["Fare"].fillna(train_data["Fare"].median())
    train_data["Embarked"] = train_data["Embarked"].fillna("S")

    test_data["Age"] = test_data["Age"].fillna(test_data["Age"].median())
    test_data["Fare"] = test_data["Fare"].fillna(test_data["Fare"].median())
    test_data["Embarked"] = test_data["Embarked"].fillna("S")

    # 分类变量转换
    train_data["Sex"] = train_data["Sex"].map({"male": 0, "female": 1})
    train_data["Embarked"] = train_data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    test_data["Sex"] = test_data["Sex"].map({"male": 0, "female": 1})
    test_data["Embarked"] = test_data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    return train_data, test_data


# 计算香农熵
def calEnt(dataSet):
    n = dataSet.shape[0]  # 总样本数目
    iset = dataSet["Survived"].value_counts()  # 每个类别的数目
    p = iset / n  # 概率
    ent = (-p * np.log2(p)).sum()  # 香农熵
    return ent


def bestFeatureSplit(dataSet):
    n_feature = dataSet.shape[1]
    # 计算原始熵
    cur_entropy = calEnt(dataSet)
    # 寻找特征划分
    best_InfoGain = 0  # 最大信息增益
    best_feature_idx = -1  # 最好特征索引

    for i in range(1, n_feature):
        featList = dataSet.iloc[:, i].value_counts().index
        new_entropy = 0
        for j in featList:
            child_dataSet = splitDataset(dataSet, i, j)
            # 加权概率
            weight = len(child_dataSet) / len(dataSet)
            # 计算熵
            new_entropy += calEnt(child_dataSet) * weight
        infoGain = cur_entropy - new_entropy  # 信息增益
        if best_InfoGain < infoGain:
            best_InfoGain = infoGain
            best_feature_idx = i
            
    return best_feature_idx


# 根据选中的特征值和唯一值划分数据集
def splitDataset(dataSet, featureIndex, value):
    col = dataSet.columns[featureIndex]
    redataSet = dataSet.loc[dataSet[col] == value, :].drop(col, axis=1)
    return redataSet

#递归生成树
def createTree(dataSet,labels,feat_labels):
    #当前节点的样本标签
    cur_label_list = dataSet.iloc[:,0].value_counts()
    
    
train_data, test_data = get_data()
a=train_data.iloc[:,0].value_counts()
print(len(a))
print(a.count(a[0]))