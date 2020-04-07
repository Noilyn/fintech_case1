import os
import joblib
import numpy as np
import pandas as pd
from src.utils import addtwodimdict
from src.model_tune import model_fit


def model_train(train_list, config_list, kernel, abs_path):
    model_assemble = {}
    # TODO 可以用prange并行加速
    for i in range(len(train_list)):
        train_data = joblib.load(os.path.join(abs_path, 'datasets', train_list[i]))
        config_data = joblib.load(os.path.join(abs_path, 'datasets', config_list[i]))
        train_data = pd.merge(train_data, config_data[['item_id', 'a', 'b']], on='item_id', how='left')
        # TODO can change into 'cate_level_id' here // 要改动需将ensemble部分也做改动
        cate_id = train_data['cate_id'].unique()
        # TODO 可以用prange并行加速
        for j in range(len(cate_id)):
            train_cate = train_data[train_data['cate_id'] == cate_id[j]]
            train_weight = train_cate[['a', 'b']].to_numpy()

            if kernel in ['gdbt', 'rf', 'xgb']:
                train_weight = np.minimum(train_weight[:, 0], train_weight[:, 1])
            else:
                train_weight = (train_weight[:, 0] + train_weight[:, 1])
            train_y = train_cate['label'].to_numpy()
            train_cate = train_cate.drop(columns=['item_id', 'cate_id', 'cate_level_id', 'a', 'b', 'label']).to_numpy()

            model = model_fit(train_cate, train_y, train_weight, kernel)
            model.fit(train_cate, train_y, sample_weight=train_weight)
            model_assemble = addtwodimdict(model_assemble, 'train_' + str(i), 'cate_id_' + str(cate_id[j]), model)
    joblib.dump(model_assemble, os.path.join(abs_path, 'results', 'model', '%s.m' % kernel))


# FIXME maybe not good for dict manipulate
if __name__ == "__main__":
    abs_path = '/Users/noilyn/course/20spring/fintech/assignment/fintech_case1'
    train_list = ['train_1.pkl', 'train_2.pkl', 'train_3.pkl', 'train_4.pkl', 'train_5.pkl', 'train_all.pkl']
    config_list = ['config_1.pkl', 'config_2.pkl', 'config_3.pkl', 'config_4.pkl', 'config_5.pkl', 'config_all.pkl']
    estimator_list = ['gdbt', 'rf', 'svr_linear', 'svr_rbf', 'xgb']
    for item in estimator_list:
        model_train(train_list, config_list, item, abs_path)