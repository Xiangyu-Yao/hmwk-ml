import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, path):
        self.transform = transforms.ToTensor()
        self.root = path
        self.images = []
        self.labels = []
        self.get_files()

    def get_files(self):
        p = os.walk(self.root)
        count = 0
        for filepath, dirnames, filenames in p:
            if len(filenames) != 20:
                continue
            if count == 200:
                break
            count += 1
            for filename in filenames:
                self.images.append(os.path.join(filepath, filename))
                label = filename.split("_")[0]
                self.labels.append(int(label))
                
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = cv.imread(self.images[idx])
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        image = self.transform(image)
        label = self.labels[idx]
        return image.flatten(), label

# 加载数据集
dataset = MyDataset(path='hmwk3/omniglot-py/images_background')

# 处理数据集：标准化和归一化
X = []
y = []
print(len(dataset))
for i in tqdm(range(len(dataset))):
    image, label = dataset[i]
    X.append(image)
    y.append(label)

X = np.array(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

min_max_scaler = MinMaxScaler()
X_normalized = min_max_scaler.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.25,stratify=y,random_state=1919)

# 训练SVM模型
svm_model = SVC(kernel='poly')
svm_model.fit(X_train, y_train)

# 预测
y_pred = svm_model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print("准确率:", accuracy)
