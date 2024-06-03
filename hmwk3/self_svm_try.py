import os
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import scipy.io as sio
import pandas as pd

# 加载数据
current_dir = os.path.dirname(os.path.abspath(__file__))
train_data_path = os.path.join(current_dir, "train_data.mat")
test_data_path = os.path.join(current_dir, "test_data.mat")

train_data = sio.loadmat(train_data_path)
test_data = sio.loadmat(test_data_path)

x_train = train_data["train"].reshape(-1, 28, 28)
y_train = np.repeat(np.arange(1, 201), 15)
x_test = test_data["test"].reshape(-1, 28, 28)

"""
# 提取 HOG 特征
def extract_hog_features(images):
    hog_features = []
    for image in images:
        #print(image.shape)
        hog_feature = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
        hog_features.append(hog_feature)
    return np.array(hog_features)

x_train_hog = extract_hog_features(x_train)
x_test_hog = extract_hog_features(x_test)

"""
# 数据标准化

scaler = MinMaxScaler()
x_train = x_train.reshape(-1, 28 * 28)
x_test = x_test.reshape(-1, 28 * 28)
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# 划分数据集
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.25, stratify=y_train, random_state=1919
)

# 训练SVM模型
svm_model = SVC(kernel="rbf")
svm_model.fit(x_train, y_train)

# 预测
y_pred = svm_model.predict(x_val)

print(y_pred[:20])
print(y_val[:20])
# 计算准确率
accuracy = accuracy_score(y_val, y_pred)
print("准确率:", accuracy)
# 作业部分
y_test_pred = svm_model.predict(x_test)
hmwk_csv = pd.read_csv(os.path.join(current_dir, "submission.csv"), header=0)
hmwk_csv["预测结果"] = y_test_pred

hmwk_csv.to_csv(os.path.join(current_dir, "submission.csv"), index=False)
