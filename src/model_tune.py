import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# TODO 根据kernel传入不同的weight
def model_fit(train_x, train_y, train_weight, kernel):
    if kernel == 'gdbt':
        model = GradientBoostingRegressor(loss='lad', learning_rate=0.01, n_estimators=300, subsample=0.75,
                                          max_depth=5, random_state=1024, max_features=0.75)

        model.fit(train_x, train_y, sample_weight=train_weight)

    elif kernel == 'rf':
        model = RandomForestRegressor()
        # model = RandomForestRegressor(n_estimators=500,max_depth=7,max_features=0.8,n_jobs=-1,random_state=1024)
        model.fit(train_x, train_y, sample_weight=train_weight)

    # TODO 还没在本机实现
    elif kernel == 'svr_linear':
        model = SVR(kernel='linear')
        # model = SVR(kernel='linear',cache_size=2000)
        model.fit(train_x, train_y, sample_weight=train_weight)

    # TODO 还没在本机实现
    elif kernel == 'svr_rbf':
        model = SVR(kernel='rbf', cache_size=2000, gamma=0.01, C=3.5)

        model.fit(train_x, train_y, sample_weight=train_weight)

    elif kernel == 'xgb':
        dtrain = xgb.DMatrix(train_x, label=train_y, weight=train_weight)
        params = {
            'booster': 'gbtree',
            'objective': 'reg:linear',
            'eval_metric': 'rmse',
            'max_depth': 7,
            'lambda': 100,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'eta': 0.008,
            'seed': 1024,
            'nthread': 4
        }
        watchlist = [(dtrain, 'train')]
        model = xgb.train(params, dtrain, num_boost_round=500, evals=watchlist)
    # TODO if xgb, fit 特殊处理
    return model