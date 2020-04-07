import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from src.feature_engineering import generate_window, processing_config


def pred_sub(df, config, df_str, solver_lst, abs_path):
    train_x = generate_window(df, (366-13))
    # TODO can set cate level id here
    cate_id = train_x['cate_id'].unique()
    item_id = []
    pred = []
    for i in cate_id:
        train_x_sub = train_x[train_x['cate_id'] == i]
        config_sub = pd.merge(train_x_sub['item_id'], config, on='item_id', how='left')
        a, b = config_sub['a'].to_numpy(), config_sub['b'].to_numpy()
        train_x_sub_data = train_x_sub.drop(columns=['item_id', 'cate_id', 'cate_level_id']).to_numpy()
        item_id.append(train_x_sub['item_id'].to_numpy())
        pred_sub = np.empty(len(train_x_sub))
        for j, item in enumerate(solver_lst):
            model = joblib.load(os.path.join(abs_path, 'results', 'model', '%s.m' % item))
            model = model[df_str]
            model_sub = model['cate_id_' + str(i)]
            if item == 'xgb':
                tmp = model_sub.predict(xgb.DMatrix(train_x_sub_data))
            tmp = model_sub.predict(train_x_sub_data)
            if j == 0:
                pred_sub = tmp
            pred_sub = 1.1 * np.sign(np.maximum(a-b, 0)) * np.maximum(pred_sub, tmp) + \
                        0.9 * np.sign(np.maximum(b-a, 0)) * np.minimum(pred_sub, tmp)
        pred.append(pred_sub)
    item_id = np.hstack(item_id)
    pred = np.hstack(pred)
    return item_id, pred


if __name__ == "__main__":
    abs_path = '/Users/noilyn/course/20spring/fintech/assignment/fintech_case1'
    item_id = []
    store_code = []
    pred_model = []
    solver_list = ['gdbt', 'rf', 'svr_linear', 'svr_rbf', 'xgb']
    df = pd.read_csv(os.path.join(abs_path, 'datasets', 'modified_traindata.csv'))
    df = df[df['date'] > (366 - 14)]

    df_1 = df[df['store_code'] == '1']
    config_1 = joblib.load(os.path.join(abs_path, 'datasets', 'config_1.pkl'))
    config_1 = processing_config(config_1)
    tmp_1, tmp_2 = pred_sub(df_1, config_1, 'train_1', solver_list, abs_path)
    item_id.append(tmp_1)
    pred_model.append(tmp_2)
    store_code.append(np.array(['1'] * len(tmp_1)))
    del df_1
    del config_1

    df_2 = df[df['store_code'] == '2']
    config_2 = joblib.load(os.path.join(abs_path, 'datasets', 'config_2.pkl'))
    config_2 = processing_config(config_2)
    tmp_1, tmp_2 = pred_sub(df_2, config_2, 'train_2', solver_list, abs_path)
    item_id.append(tmp_1)
    pred_model.append(tmp_2)
    store_code.append(np.array(['2'] * len(tmp_1)))
    del df_2
    del config_2

    df_3 = df[df['store_code'] == '3']
    config_3 = joblib.load(os.path.join(abs_path, 'datasets', 'config_3.pkl'))
    config_3 = processing_config(config_3)
    tmp_1, tmp_2 = pred_sub(df_3, config_3, 'train_3', solver_list, abs_path)
    item_id.append(tmp_1)
    pred_model.append(tmp_2)
    store_code.append(np.array(['3'] * len(tmp_1)))
    del df_3
    del config_3

    df_4 = df[df['store_code'] == '4']
    config_4 = joblib.load(os.path.join(abs_path, 'datasets', 'config_4.pkl'))
    config_4 = processing_config(config_4)
    tmp_1, tmp_2 = pred_sub(df_4, config_4, 'train_4', solver_list, abs_path)
    item_id.append(tmp_1)
    pred_model.append(tmp_2)
    store_code.append(np.array(['4'] * len(tmp_1)))
    del df_4
    del config_4

    df_5 = df[df['store_code'] == '5']
    config_5 = joblib.load(os.path.join(abs_path, 'datasets', 'config_5.pkl'))
    config_5 = processing_config(config_5)
    tmp_1, tmp_2 = pred_sub(df_5, config_5, 'train_5', solver_list, abs_path)
    item_id.append(tmp_1)
    pred_model.append(tmp_2)
    store_code.append(np.array(['5'] * len(tmp_1)))
    del df_5
    del config_5

    df_all = df[df['store_code'] == 'all']
    config_all = joblib.load(os.path.join(abs_path, 'datasets', 'config_all.pkl'))
    config_all = processing_config(config_all)
    tmp_1, tmp_2 = pred_sub(df_all, config_all, 'train_all', solver_list, abs_path)
    item_id.append(tmp_1)
    pred_model.append(tmp_2)
    store_code.append(np.array(['all'] * len(tmp_1)))
    del df_all
    del config_all

    item_id = np.hstack(item_id)
    pred_model = np.hstack(pred_model)
    store_code = np.hstack(store_code)

    pred_ = np.empty(len(item_id), 3)
    pred_[:, 0], pred_[:, 1], pred_[:, 2] = item_id, store_code, pred_model
    pred_model = pd.DataFrame(data=pred_, columns=['item_id', 'store_code', 'pred_model'])
    pred_rule = joblib.load(os.path.join(abs_path, 'datasets', 'pred_rule.pkl'))
    pred_ensemble = pd.merge(pred_model, pred_rule, on=['item_id', 'store_code'])
    del pred_model
    del pred_rule
    pred_ensemble['pred'] = (0.25 * pred_ensemble['pred_rule'] + 0.75 * pred_model['pred_model'])
    pred_ensemble.drop(columns=['pred_rule', 'pred_model'], inplace=True)
    pred_ensemble.to_csv(os.path.join(abs_path, 'results', 'results.csv'))


