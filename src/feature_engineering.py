import os
import joblib
import pandas as pd
#from dateutil.relativedelta import relativedelta


def sub_window(df_sub, end_date, time_delta):
    """ Given the df, end_date and time_delta,
            return sum and avg of numerical feature during the time_delta before end date"""
    #start_date = end_date - relativedelta(days=time_delta)
    start_date = end_date - time_delta
    df_tmp = df_sub[df_sub['date'] < end_date]
    df_tmp = df_tmp[df_tmp['date'] >= start_date]
    df_tmp_amount = df_tmp[['pv_ipv', 'pv_uv', 'cart_ipv', 'cart_uv', 'collect_uv', 'num_gmv', 'amt_gmv',
                            'qty_gmv', 'unum_gmv', 'amt_alipay', 'num_alipay', 'qty_alipay', 'unum_alipay',
                            'ztc_pv_ipv', 'tbk_pv_ipv', 'ss_pv_ipv', 'jhs_pv_ipv', 'ztc_pv_uv', 'tbk_pv_uv',
                            'ss_pv_uv', 'jhs_pv_uv', 'num_alipay_njhs', 'amt_alipay_njhs', 'qty_alipay_njhs',
                            'unum_alipay_njhs', 'item_id']]
    # 因为数据是分仓做的所以不 group by store code
    grouped = df_tmp_amount.groupby(['item_id'])
    sum_ = grouped.sum()
    sum_.columns = map(lambda x: str(x) + '_%s' % str(time_delta) + '_%s' % 'sum_', sum_.columns)
    mean_ = grouped.mean()
    mean_.columns = map(lambda x: str(x) + '_%s' % str(time_delta) + '_%s' % 'mean_', mean_.columns)
    sum_mean_ = pd.merge(sum_, mean_, on='item_id')
    return sum_mean_


def generate_window(df_sub, end_date):
    """
    :param df_sub: 14 days data
    :return: a 14 days' training data window
    """
    window_sub = [1, 2, 3, 5, 7, 9, 11, 14]
    window_sub.reverse()
    sum_mean = sub_window(df_sub, end_date, window_sub[0])

    for i in range(1, 8):
        tmp = sub_window(df_sub, end_date, window_sub[i])
        sum_mean = pd.merge(sum_mean, tmp, on='item_id', how='left')

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

    tmp = df_sub[['item_id', 'cate_id', 'cate_level_id']]

    # 均值填充
    sum_mean.fillna(sum_mean.mean(), inplace=True)
    sum_mean = pd.merge(sum_mean, tmp, on='item_id', how='left')

    return sum_mean


def train_gen(df):
    start_date = df['date'].min() + 14
    end_date = df['date'].max() - 14
    slide_date = df['date'].min()
    interval = end_date - start_date

    df_sub = df[df['date']>=slide_date]
    df_sub = df_sub[df_sub['date']<start_date]
    train_x = generate_window(df_sub, start_date)

    df_sub_after = df[df['date']>=start_date]
    df_sub_after = df_sub_after[df_sub_after['date']<(start_date+14)]
    tmp = df_sub_after[['item_id', 'store_code', 'qty_alipay_njhs']]
    tmp.rename(columns={'qty_alipay_njhs': 'label'}, inplace=True)
    grouped = tmp.groupby(['item_id', 'store_code'])
    train = pd.merge(train_x, grouped.sum(), on='item_id', how='outer')

    # TODO change the train data size here
    #for i in range(0, interval.days, 7):
    for i in range(0, interval, 7):
        slide_date += 7
        start_date += 7
        df_sub = df[df['date']>=slide_date]
        df_sub = df_sub[df_sub['date']<start_date]
        train_x = generate_window(df_sub, start_date)
        df_sub_after = df[df['date'] >= start_date]
        df_sub_after = df_sub_after[df_sub_after['date'] < (start_date + 14)]
        tmp = df_sub_after[['item_id', 'store_code', 'qty_alipay_njhs']]
        tmp.rename(columns={'qty_alipay_njhs': 'label'}, inplace=True)
        grouped = tmp.groupby(['item_id', 'store_code'])
        train = pd.concat([train, pd.merge(train_x, grouped.sum(), on='item_id', how='outer')])
    # label 空缺值处理
    train['label'].fillna(0, inplace=True)
    train.fillna(0, inplace=True)
    return train


def processing_config(df):
    df['a_b'] = df['a_b'].str.split('_')
    df['a'] = df['a_b'].str[0].astype('float64')
    df['b'] = df['a_b'].str[1].astype('float64')
    df.drop(columns=['a_b'], inplace=True)
    return df


if __name__ == "__main__":
    # need to change path here
    abs_path = '/Users/noilyn/course/20spring/fintech/assignment/fintech_case1'
    df = pd.read_csv(os.path.join(abs_path, 'datasets', 'modified_traindata.csv'))
    # df['date'] = pd.to_datetime(df['date'])
    df['store_code'] = df['store_code'].astype('str')

    # TODO 仅仅是删掉了双十一双十二，没有将之后的日期往前平移， 未来可以去均值模拟双十一双十二
    # TODO 可以观察一下不同id产品随时间的销量波动情况，局部日期没有的另做处理（删掉某些无数据时间点窗口）
    #df = df[df['date'] != pd.datetime(2014, 11, 11)]
    #df = df[df['date'] != pd.datetime(2014, 12, 12)]
    #df = df[df['date'] != pd.datetime(2015, 11, 11)]
    #df = df[df['date'] != pd.datetime(2015, 12, 12)]

    df_1 = df[df['store_code'] == '1']
    df_2 = df[df['store_code'] == '2']
    df_3 = df[df['store_code'] == '3']
    df_4 = df[df['store_code'] == '4']
    df_5 = df[df['store_code'] == '5']
    df_all = df[df['store_code'] == 'all']

    train_1 = train_gen(df_1)
    joblib.dump(train_1, os.path.join(abs_path, 'datasets', 'train_1.pkl'))
    del train_1

    train_2 = train_gen(df_2)
    joblib.dump(train_2, os.path.join(abs_path, 'datasets', 'train_2.pkl'))
    del train_2

    train_3 = train_gen(df_3)
    joblib.dump(train_3, os.path.join(abs_path, 'datasets', 'train_3.pkl'))
    del train_3

    train_4 = train_gen(df_4)
    joblib.dump(train_4, os.path.join(abs_path, 'datasets', 'train_4.pkl'))
    del train_4

    train_5 = train_gen(df_5)
    joblib.dump(train_5, os.path.join(abs_path, 'datasets', 'train_5.pkl'))
    del train_5

    train_all = train_gen(df_all)
    joblib.dump(train_all, os.path.join(abs_path, 'datasets', 'train_all.pkl'))
    del train_all

    config = pd.read_csv(os.path.join(abs_path, 'datasets', 'configData.csv'))
    config = processing_config(config)

    config_1 = config[config['store_code'] == '1']
    config_2 = config[config['store_code'] == '2']
    config_3 = config[config['store_code'] == '3']
    config_4 = config[config['store_code'] == '4']
    config_5 = config[config['store_code'] == '5']
    config_all = config[config['store_code'] == 'all']

    joblib.dump(config_1, os.path.join(abs_path, 'datasets', 'config_1.pkl'))
    joblib.dump(config_2, os.path.join(abs_path, 'datasets', 'config_2.pkl'))
    joblib.dump(config_3, os.path.join(abs_path, 'datasets', 'config_3.pkl'))
    joblib.dump(config_4, os.path.join(abs_path, 'datasets', 'config_4.pkl'))
    joblib.dump(config_5, os.path.join(abs_path, 'datasets', 'config_5.pkl'))
    joblib.dump(config_all, os.path.join(abs_path, 'datasets', 'config_all.pkl'))


