import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.externals import joblib



data = pd.read_csv("traindata.csv")
all_data = data[data["store_code"] == "all"]
rest_data = data[data["store_code"] != "all"]

rest_data["store_code"].astype("int32")

condition = ((rest_data["date"] >= "2014-10-10") & (rest_data["date"] <= "2014-11-01")) | ((rest_data["date"] >= "2014-11-20") & (rest_data["date"] <= "2014-12-05")) | ((rest_data["date"] >= "2014-12-20") & (rest_data["date"] <= "2015-06-10")) | ((rest_data["date"] >= "2015-06-20") & (rest_data["date"] <= "2015-11-01")) | ((rest_data["date"] >= "2015-11-16") & (rest_data["date"] <= "2015-12-05"))

rest_data = rest_data[condition]

rest_data_y = rest_data["qty_alipay_njhs"]
rest_data.drop(["qty_alipay_njhs", "date"], inplace=True, axis=1)


scaler = StandardScaler()
rest_data = scaler.fit_transform(rest_data)


model = MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
             beta_2=0.999, early_stopping=False, epsilon=1e-08,
             hidden_layer_sizes=(120, 60), learning_rate='constant',
             learning_rate_init=0.001, max_fun=15000, max_iter=200,
             momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
             power_t=0.5, random_state=None, shuffle=True, solver='adam',
             tol=0.0001, validation_fraction=0.1, verbose=False,
             warm_start=False)


model.fit(rest_data, rest_data_y)

'''
x = [(i * 20, j * 10, k * 5) * l  for i in range(1, 10) for j in range(2, 7) for k in range(7, 9) for l in range(1, 3)]
y = [(i * 20, j * 10) * l  for i in range(1, 10) for j in range(2, 7) for l in range(1, 3)]

parameters = {"activation":["relu"], "hidden_layer_sizes":x + y}
grid_search = GridSearchCV(MLPRegressor(), parameters, cv=10, n_jobs=-1, scoring='neg_mean_squared_error')
grid_search.fit(X=rest_data, y=rest_data_y)


print(grid_search.best_params_)
'''
joblib.dump(model, 'fitted_model.model')



