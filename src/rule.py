import os
import joblib
import numpy as np
import pandas as pd
from src.feature_engineering import processing_config


# TODO Having known that the max day is 366
def rule_predict(df, config_df):
    df = df[df['date'] > (366 - 14)]
    df_1 = df[df['date'] <= (366 - 7)]
    df_2 = df[df['date'] > (366 - 7)]
    df_1 = df_1[['item_id', 'store_code', 'qty_alipay_njhs']]
    df_2 = df_2[['item_id', 'store_code', 'qty_alipay_njhs']]
    grouped = df_1.groupby(['item_id', 'store_code'])
    sum_week1 = grouped.sum()
    sum_week1.rename(columns={'qty_alipay_njhs': 'pred_week1'}, inplace=True)
    grouped = df_2.groupby(['item_id', 'store_code'])
    sum_week2 = grouped.sum()
    sum_week2.rename(columns={'qty_alipay_njhs': 'pred_week2'}, inplace=True)
    sum_week = pd.merge(sum_week1, sum_week2, on=['item_id', 'store_code'], how='outer')
    del sum_week1
    del sum_week2
    sum_week.fillna(0, inplace=True)

    config_df = pd.merge(config_df, sum_week, on=['item_id', 'store_code'], how='outer')
    config_df.fillna(0, inplace=True)

    a, b, pred_week1, pred_week2 = config_df['a'].to_numpy(), config_df['b'].to_numpy(), \
                                   config_df['pred_week1'].to_numpy(), config_df['pred_week2'].to_numpy()
    pred_rule = 2 * (np.sign(np.maximum(a-b, 0)) * np.maximum(pred_week1, pred_week2) +
                     np.sign(np.maximum(b-a, 0)) * np.minimum(pred_week1, pred_week2))
    config_df['pred_rule'] = pd.Series(pred_rule)
    pred_rule = config_df[['item_id', 'store_code', 'pred_rule']]
    return pred_rule


if __name__ == "__main__":
    abs_path = '/Users/noilyn/course/20spring/fintech/assignment/fintech_case1'
    df = pd.read_csv(os.path.join(abs_path, 'datasets', 'modified_traindata.csv'))
    config_df = pd.read_csv(os.path.join(abs_path, 'datasets', 'configData.csv'))
    config_df = processing_config(config_df)
    pred_rule = rule_predict(df, config_df)
    joblib.dump(pred_rule, os.path.join(abs_path, 'datasets', 'pred_rule.pkl'))