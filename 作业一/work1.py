import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
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

degree = 8

# 多项式特征
poly_features = PolynomialFeatures(degree=degree)
x_train_poly = poly_features.fit_transform(x_train)
x_test_poly = poly_features.transform(x_test)
new_x_test_poly=poly_features.transform(new_x_test)

# 模型训练
model = LinearRegression()
model.fit(x_train_poly, y_train)

#测试
train_predictions = model.predict(x_train_poly)
test_predictions = model.predict(x_test_poly)

#方差
train_mse= mean_squared_error(y_train,train_predictions)
test_mse = mean_squared_error(y_test,test_predictions)

print("训练集方差:",train_mse)
print("测试集方差",test_mse)

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


#作业提交部分
new_test_prediction=model.predict(new_x_test_poly)
#print(new_test_prediction)
new_test_data['y']=new_test_prediction
new_test_data.to_csv("作业一/homework.csv",index=False)