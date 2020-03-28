from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import joblib
import os

# need to change path here
abs_path='/Users/noilyn/course/20spring/fintech/assignment/fintech_case1'
df = pd.read_csv(os.path.join(abs_path, 'datasets/traindata.csv'))
df['date'] = pd.to_datetime(df['date'])
df = df[df['date']!=pd.datetime(2015,11,11)]
df = df[df['date']!=pd.datetime(2015,12,12)]
df['store_code'] = df['store_code'].astype('str')


def sub_window(df_sub, start_date, time_delta):
    end_date = start_date + relativedelta(days=time_delta)
    df_tmp = df_sub[df_sub['date'] <= end_date]
    df_tmp = df_tmp[df_tmp['date'] >= start_date]
    df_tmp_amount = df_tmp[['pv_ipv', 'pv_uv', 'cart_ipv', 'cart_uv', 'collect_uv', 'num_gmv', 'amt_gmv',
                            'qty_gmv', 'unum_gmv', 'amt_alipay', 'num_alipay', 'qty_alipay', 'unum_alipay',
                            'ztc_pv_ipv', 'tbk_pv_ipv', 'ss_pv_ipv', 'jhs_pv_ipv', 'ztc_pv_uv', 'tbk_pv_uv',
                            'ss_pv_uv', 'jhs_pv_uv', 'num_alipay_njhs', 'amt_alipay_njhs', 'qty_alipay_njhs',
                            'unum_alipay_njhs', 'item_id']]
    grouped = df_tmp_amount.groupby(['item_id'])
    sum_ = grouped.sum()
    sum_.columns = map(lambda x: str(x) + '_%s' % str(time_delta) + '_%s' % 'sum_', sum_.columns)
    mean_ = grouped.mean()
    mean_.columns = map(lambda x: str(x) + '_%s' % str(time_delta) + '_%s' % 'mean_', mean_.columns)
    sum_mean_ = pd.merge(sum_, mean_, on='item_id')
    return sum_mean_


def generate_window(df_sub, start_date):
    window_sub = [1, 2, 3, 5, 7, 9, 11, 14]
    sum_mean = sub_window(df_sub, start_date, window_sub[0])
    for i in range(1, 8):
        tmp = sub_window(df_sub, start_date, window_sub[i])
        sum_mean = pd.merge(sum_mean, tmp, on='item_id')
    end_date = start_date + relativedelta(days=14)

    tmp = df_sub[['item_id', 'qty_alipay_njhs']]
    grouped = tmp.groupby(['item_id'])
    sum_mean = pd.merge(sum_mean,
                        pd.merge(pd.merge(grouped.max(), grouped.min(), on='item_id'), grouped.std(), on='item_id'),
                        on='item_id')
    tmp_ = sum_mean.columns.values
    sum_mean.rename(columns={tmp_[-3]: 'item_njhs_max', tmp_[-2]: 'item_njhs_min', tmp_[-1]: 'item_njhs_std'},
                    inplace=True)

    tmp = df_sub[['item_id', 'cate_id', 'qty_alipay_njhs']]
    grouped = tmp.groupby(['item_id', 'cate_id'])
    sum_mean = pd.merge(sum_mean,
                        pd.merge(pd.merge(grouped.max(), grouped.min(), on=['item_id', 'cate_id']), grouped.std(),
                                 on=['item_id', 'cate_id']), on='item_id')
    tmp_ = sum_mean.columns.values
    sum_mean.rename(columns={tmp_[-3]: 'cate_njhs_max', tmp_[-2]: 'cate_njhs_min', tmp_[-1]: 'cate_njhs_std'},
                    inplace=True)

    tmp = df_sub[['item_id', 'cate_level_id', 'qty_alipay_njhs']]
    grouped = tmp.groupby(['item_id', 'cate_level_id'])
    sum_mean = pd.merge(sum_mean,
                        pd.merge(pd.merge(grouped.max(), grouped.min(), on=['item_id', 'cate_level_id']), grouped.std(),
                                 on=['item_id', 'cate_level_id']), on='item_id')
    tmp_ = sum_mean.columns.values
    sum_mean.rename(
        columns={tmp_[-3]: 'cate_level_njhs_max', tmp_[-2]: 'cate_level_njhs_min', tmp_[-1]: 'cate_level_njhs_std'},
        inplace=True)

    tmp = df_sub[['item_id', 'brand_id', 'qty_alipay_njhs']]
    grouped = tmp.groupby(['item_id', 'brand_id'])
    sum_mean = pd.merge(sum_mean,
                        pd.merge(pd.merge(grouped.max(), grouped.min(), on=['item_id', 'brand_id']), grouped.std(),
                                 on=['item_id', 'brand_id']), on='item_id')
    tmp_ = sum_mean.columns.values
    sum_mean.rename(columns={tmp_[-3]: 'brand_njhs_max', tmp_[-2]: 'brand_njhs_min', tmp_[-1]: 'brand_njhs_std'},
                    inplace=True)

    tmp = df_sub[['item_id', 'supplier_id', 'qty_alipay_njhs']]
    grouped = tmp.groupby(['item_id', 'supplier_id'])
    sum_mean = pd.merge(sum_mean,
                        pd.merge(pd.merge(grouped.max(), grouped.min(), on=['item_id', 'supplier_id']), grouped.std(),
                                 on=['item_id', 'supplier_id']), on='item_id')
    tmp_ = sum_mean.columns.values
    sum_mean.rename(
        columns={tmp_[-3]: 'supplier_njhs_max', tmp_[-2]: 'supplier_njhs_min', tmp_[-1]: 'supplier_njhs_std'},
        inplace=True)
    return sum_mean


def store_code_train(df_sub, start_date, interval):
    window_sub = generate_window(df_sub, start_date)
    for i in range(interval.days // 14):
        start_date += relativedelta(days=14)
        window_sub = pd.concat([window_sub, generate_window(df_1, start_date)])
    return window_sub

df_1 = df[df['store_code']=='1']
df_2 = df[df['store_code']=='2']
df_3 = df[df['store_code']=='3']
df_4 = df[df['store_code']=='4']
df_5 = df[df['store_code']=='5']
df_all = df[df['store_code']=='all']
interval = df['date'].max() - df['date'].min()
start_date = df['date'].min()
store_code_1 = store_code_train(df_1, start_date, interval)
joblib.dump(store_code_1, os.path.join(abs_path, 'datasets/store_code_1.pkl'))
store_code_2 = store_code_train(df_2, start_date, interval)
joblib.dump(store_code_2, os.path.join(abs_path, 'datasets/store_code_2.pkl'))
store_code_3 = store_code_train(df_3, start_date, interval)
joblib.dump(store_code_3, os.path.join(abs_path, 'datasets/store_code_3.pkl'))
store_code_4 = store_code_train(df_4, start_date, interval)
joblib.dump(store_code_4, os.path.join(abs_path, 'datasets/store_code_4.pkl'))
store_code_5 = store_code_train(df_5, start_date, interval)
joblib.dump(store_code_5, os.path.join(abs_path, 'datasets/store_code_5.pkl'))
store_code_all = store_code_train(df_all, start_date, interval)
joblib.dump(store_code_all, os.path.join(abs_path, 'datasets/store_code_all.pkl'))

