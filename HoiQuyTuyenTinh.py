import numpy as np
import pandas as pd
import seaborn as sns

from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import train_test_split

data = pd.read_csv("kc_house_data.csv")

"""## Giải thích các thuộc tính


*   id: ký hiệu của ngôi nhà
*   date: ngày bán ngôi nhà
*   price: giá nhà
*   bedrooms: số phòng ngủ
*   bathrooms: số phòng tắm
*   sqft_living: diện tích ngôi nhà
*   sqft_lot: diện tích lô đất
*   floors: số tầng
*   waterfront: nhà gần sông
*   view: có view
*   condition: tình trạng
*   grade: cấp của ngôi nhà
*   sqft_above: diện tích nhà ngoài tầng hầm
*   sqft_basement: diện tích tầng hầm
*   yr_built: năm xây dựng
*   yr_renobvated: năm cải tạo
*   zipcode: mã vùng
*   lat: vĩ độ
*   long: kinh độ
*   sqft_living15: diện tích phòng khách 2015
*   sqft_lot15: diện tích lô đất 2015
"""
'''
## Sử dụng 3 thuộc tính để đào tạo mô hình
features = ['bedrooms','bathrooms','floors']
X_train, X_test, y_train, y_test = \
  train_test_split(data[features], data['price'], test_size=0.2, random_state=21)

model_1 = linear_model.LinearRegression()
model_1.fit(X_train, y_train)

print("Intercept: {}".format(model_1.intercept_))
print("Coefficients: {}".format(model_1.coef_))

from sklearn import metrics
y_predict = model_1.predict(X_test)
err = metrics.mean_squared_error(y_test, y_predict)
print("mse_err = {}".format(round(err, 1)))
print("rmse_err = {}".format(round(np.sqrt(err), 1)))
'''
## Sử dụng 10 thuộc tính để xây dựng
for i in range(0, 10):
  features = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms',\
            'view', 'sqft_basement', 'bedrooms', 'lat', 'waterfront']
  X_train, X_test, y_train, y_test = \
    train_test_split(data[features], data['price'], test_size=0.2, random_state=i)

  model_2 = linear_model.LinearRegression()
  model_2.fit(X_train, y_train)

  from sklearn import metrics
  y_predict = model_2.predict(X_test)
  err = metrics.mean_squared_error(y_test, y_predict)
  print("Random_state =  {}".format(i))
  ##print("mse_err = {}".format(round(err, 1)))
  print("rmse_err = {}".format(round(np.sqrt(err), 1)))


'''
## Sử dụng 10 thuộc tính KHÁC để xây dựng mô hình

features = ['zipcode', 'long', 'condition', 'yr_built', 'sqft_lot',\
            'bedrooms', 'lat', 'sqft_lot15', 'floors', 'yr_renovated']
X_train, X_test, y_train, y_test = \
  train_test_split(data[features], data['price'], test_size=0.2, random_state=21)

model_2_1 = linear_model.LinearRegression()
model_2_1.fit(X_train, y_train)

from sklearn import metrics
y_predict = model_2_1.predict(X_test)
err = metrics.mean_squared_error(y_test, y_predict)
print("mse_err = {}".format(round(err, 1)))
print("rmse_err = {}".format(round(np.sqrt(err), 1)))

## Sử dụng tất cả thuộc tính để xây dựng mô hình

X_train, X_test, y_train, y_test = \
  train_test_split(data.iloc[:,3:], data['price'], test_size=0.2, random_state=21) 
X_test.shape

model_3 = linear_model.LinearRegression()
model_3.fit(X_train, y_train)

from sklearn import metrics
y_predict = model_3.predict(X_test)
err = metrics.mean_squared_error(y_test, y_predict)
print("mse_err = {}".format(round(err, 1)))
print("rmse_err = {}".format(round(np.sqrt(err), 1)))


"""## Hàm để test ví dụ trong slides"""

class GDLinearRegression:
  def __init__(self, learning_rate, step, theta0, theta1, theta2,theta3):
    self.learning_rate = learning_rate
    self.step = step
    self.theta0 = theta0
    self.theta1 = theta1
    self.theta2 = theta2
    self.theta3 = theta3
  
  def fit(self, X, y):

    theta0 = self.theta0
    theta1 = self.theta1
    theta2 = self.theta2
    theta3 = self.theta3

    X = np.array(X)
    y = np.array(y)
    
    h = [0, 0, 0]
    for solan in range(0, self.step):
      for i in range(0, 3):
        h[i] = theta0 + theta1*X[i][0] + theta2*X[i][1] + theta3*X[i][2]

      y_h = [0, 0, 0]
      for i in range(0, 3):
        y_h[i] = y[i] - h[i]

      theta0 = theta0 + self.learning_rate*(y_h[0]*1 + y_h[1]*1 + y_h[2]*1)
      theta1 = theta1 + self.learning_rate*(y_h[0]*X[0][0] + y_h[1]*X[1][0] + y_h[2]*X[2][0])
      theta2 = theta2 + self.learning_rate*(y_h[0]*X[0][1] + y_h[1]*X[1][1] + y_h[2]*X[2][1])
      theta3 = theta3 + self.learning_rate*(y_h[0]*X[0][2] + y_h[1]*X[1][2] + y_h[2]*X[2][2])
    
    return theta0, theta1, theta2, theta3

  def predict(self, X, y):
    X = np.array(X)
    y = np.array(y)
    h = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(0, 10):
      h[i] = self.theta0 + self.theta1*X[i][0] + self.theta2*X[i][1] + self.theta3*X[i][2];
    return h

model_GD = GDLinearRegression(0.05, 5, 10, 15, 20, 25)
theta0, theta1, theta2, theta3 = model_GD.fit(X_train.head(3), y_train.head(3))
X = np.array(X_test.head(10))
y = np.array(y_test.head(10))
h = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

for i in range(0, 10):
  h[i] = theta0 + theta1*X[i][0] + theta2*X[i][1] + theta3*X[i][2]

for i in range(0, 10):
  tong = (y[i]-h[i])**2
MSE = 1/10*(tong)
RMSE = np.sqrt(MSE)
print(RMSE)

## Vẽ biểu đồ giữa giá và các thuộc tính trong tập dữ liệu

import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x="price", y='sqft_living', data=data)
plt.ylim(0,)

import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x="price", y='bedrooms', data=data)
plt.ylim(0,)

import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x="price", y='sqft_above', data=data)
plt.ylim(0,)
'''