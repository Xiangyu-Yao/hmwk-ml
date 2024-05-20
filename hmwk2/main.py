from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris = load_iris()
target = iris["target"]
data = iris["data"]

train_x, val_x, train_y, val_y = train_test_split(
    data, target, test_size=0.2, random_state=19198
)

bys = GaussianNB()

bys.fit(train_x, train_y)

y_pred = bys.predict(val_x)

auc = accuracy_score(val_y, y_pred)
print(auc)

# 测试环节
test_iris = pd.read_csv("hmwk2/iris_test.csv", header=0)

test_data = np.array(test_iris[test_iris.columns[:4]].values).reshape(-1, 4)
# print(test_data)
pred_test_target = bys.predict(test_data)
print(pred_test_target)
test_iris["species"]=pred_test_target

test_iris.to_csv("hmwk2/homework.csv",index=False)
