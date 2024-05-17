from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

iris=load_iris()
target=iris['target']
data=iris['data']

train_x,val_x,train_y,val_y= train_test_split(data,target,test_size=0.2,random_state=19198)

bys=GaussianNB()

bys.fit(train_x,train_y)

y_pred=bys.predict(val_x)

auc=accuracy_score(val_y,y_pred)
print(auc)