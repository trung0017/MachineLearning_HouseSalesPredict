import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("kc_house_data.csv", index_col=0)
data.info()
data.shape

X = data.iloc[:,3:20]
y = data.price
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = "entropy", random_state = 0)
regressor.fit(X_train, y_train)

y_model_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error
err = mean_squared_error(y_test, y_model_pred)
rmse_err = np.sqrt(err)
rmse_err