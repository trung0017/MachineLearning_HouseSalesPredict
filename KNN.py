import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("kc_house_data.csv") ##index_colum là bỏ cột số 1 không phải bỏ dòng đầu tiên

## Hiển thị thông tin các cột
data.info()

data.shape

data.head()

data.dtypes

data.describe()

for i in range(0, 20):
  print(data.iloc[:,i:(i+1)].isnull().sum())
  
X_train, X_test, y_train, y_test = \
  train_test_split(data.iloc[:,3:], data['price'], test_size=0.2, random_state=21)

from sklearn.neighbors import KNeighborsRegressor
model_KNN = KNeighborsRegressor(n_neighbors=17)
model_KNN.fit(X_train, y_train)

y_predict = model_KNN.predict(X_test)

from sklearn.metrics import mean_squared_error
err = mean_squared_error(y_test, y_predict)
rmse_err = np.sqrt(err)

print("err: {}".format(round(err,3)))
print("rmse_err: {}".format(round(rmse_err,3)))

y_predict

for k in range(20):
  k = k + 1
  KNeighborsRegressor(n_neighbors=k)