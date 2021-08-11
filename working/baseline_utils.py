import pandas as pd
import numpy as np
import os
from multiprocessing import Pool, cpu_count
from sklearn import model_selection
from tqdm import tqdm
import lightgbm as lgb
import joblib
import copy as cp


def load_book(stock_id=0, data_type='train'):
    """加载 book 数据
    """
    book_df = pd.read_parquet(
        os.path.join('../data',
                     'book_{}.parquet/stock_id={}'.format(data_type,
                                                          stock_id)))
    book_df['stock_id'] = stock_id
    book_df['stock_id'] = book_df['stock_id'].astype(np.int8)
    book_df['seconds_in_bucket'] = book_df['seconds_in_bucket'].astype(
        np.int32)

    return book_df


def load_trade(stock_id=0, data_type='train'):
    """加载 trade 数据
    """
    trade_df = pd.read_parquet(
        os.path.join(
            '../data',
            'trade_{}.parquet/stock_id={}'.format(data_type, stock_id)))
    trade_df['stock_id'] = stock_id
    trade_df['stock_id'] = trade_df['stock_id'].astype(np.int8)
    trade_df['order_count'] = trade_df['order_count'].astype(np.int32)
    trade_df['seconds_in_bucket'] = trade_df['seconds_in_bucket'].astype(
        np.int32)

    return trade_df


def log_return(list_stock_prices):
    """收益率
    """
    return np.log(list_stock_prices).diff()


def realized_volatility(series_log_return):
    """波动率
    """
    return np.sqrt(np.sum(series_log_return**2))


def fix_jsonerr(df):
    """
    """
    df.columns = [
        "".join(c if c.isalnum() else "_" for c in str(x)) for x in df.columns
    ]
    return df


def feature_row(book):
    """
    """
    # book_wap1 生成标签
    for i in [
            1,
            2,
    ]:
        # wap
        book[f'book_wap{i}'] = (book[f'bid_price{i}'] * book[f'ask_size{i}'] +
                                book[f'ask_price{i}'] *
                                book[f'bid_size{i}']) / (book[f'bid_size{i}'] +
                                                         book[f'ask_size{i}'])

    # mean wap
    book['book_wap_mean'] = (book['book_wap1'] + book['book_wap2']) / 2

    # wap diff
    book['book_wap_diff'] = book['book_wap1'] - book['book_wap2']

    # other orderbook features
    book['book_price_spread'] = (book['ask_price1'] - book['bid_price1']) / (
        book['ask_price1'] + book['bid_price1'])
    book['book_bid_spread'] = book['bid_price1'] - book['bid_price2']
    book['book_ask_spread'] = book['ask_price1'] - book['ask_price2']
    book['book_total_volume'] = book['ask_size1'] + book['ask_size2'] + book[
        'bid_size1'] + book['bid_size2']
    book['book_volume_imbalance'] = (book['ask_size1'] + book['ask_size2']) - (
        book['bid_size1'] + book['bid_size2'])
    return book


def feature_agg(book, trade):
    """
    """
    # 聚合生成特征
    book_feats = book.columns[book.columns.str.startswith('book_')].tolist()
    trade_feats = ['price', 'size', 'order_count', 'seconds_in_bucket']

    trade = trade.groupby(['time_id', 'stock_id'])[trade_feats].agg(
        ['sum', 'mean', 'std', 'max', 'min', 'median']).reset_index()

    book = book.groupby(['time_id', 'stock_id'])[book_feats].agg(
        [lambda x: realized_volatility(log_return(x))]).reset_index()

    # 修改特征名称
    book.columns = ["".join(col).strip() for col in book.columns.values]
    trade.columns = ["".join(col).strip() for col in trade.columns.values]
    df_ret = book.merge(trade, how='left', on=['time_id', 'stock_id'])
    return df_ret


def gen_data_train(stock_id=0):
    """
    """
    book = load_book(stock_id, 'train')
    trade = load_trade(stock_id, 'train')

    book = book.sort_values(by=['time_id', 'seconds_in_bucket'])
    trade = trade.sort_values(by=['time_id', 'seconds_in_bucket'])

    book = feature_row(book)

    df_ret1 = feature_agg(book, trade)

    df_ret2 = calculate_features2(book, trade)

    return df_ret2.merge(df_ret1, how='left', on='time_id')


def gen_data_test(stock_id=0):
    """
    """
    book = load_book(stock_id, 'test')
    trade = load_trade(stock_id, 'test')

    book = book.sort_values(by=['time_id', 'seconds_in_bucket'])
    trade = trade.sort_values(by=['time_id', 'seconds_in_bucket'])

    book = feature_row(book)

    df_ret1 = feature_agg(book, trade)

    df_ret2 = calculate_features2(book, trade)

    return df_ret2.merge(df_ret1, how='left', on='time_id')


def gen_data_multi(stock_lst, data_type='train'):
    """
    """
    with Pool(cpu_count()) as p:
        if data_type == 'train':
            feature_dfs = list(
                tqdm(p.imap(gen_data_train, stock_lst), total=len(stock_lst)))
        if data_type == 'test':
            feature_dfs = list(
                tqdm(p.imap(gen_data_test, stock_lst), total=len(stock_lst)))
    df_ret = pd.concat(feature_dfs)
    return df_ret


def gen_data_encoding(df_ret, df_label, data_type='train'):
    """
    test 不使用自己数据的 stock_id encoding
    """

    # 对 stock_id 进行 encoding
    vol_feats = [f for f in df_ret.columns if ('lambda' in f) & ('wap' in f)]
    if data_type == 'train':
        # agg
        stock_df = df_ret.groupby('stock_id')[vol_feats].agg([
            'mean',
            'std',
            'max',
            'min',
            'median',
        ]).reset_index()

        # fix column names
        stock_df.columns = ['stock_id'] + [
            f'{f}_stock' for f in stock_df.columns.values.tolist()[1:]
        ]
        stock_df = fix_jsonerr(stock_df)

    # 对 time_id 进行 encoding
    time_df = df_ret.groupby('time_id')[vol_feats].agg([
        'mean',
        'std',
        'max',
        'min',
        'median',
    ]).reset_index()
    time_df.columns = ['time_id'] + [
        f'{f}_time' for f in time_df.columns.values.tolist()[1:]
    ]

    # merge
    df_ret = df_ret.merge(time_df, how='left', on='time_id')

    # make sure to fix json error for lighgbm
    df_ret = fix_jsonerr(df_ret)

    # out
    if data_type == 'train':
        df_ret = df_ret.merge(stock_df, how='left', on='stock_id').merge(
            df_label, how='left',
            on=['stock_id', 'time_id']).replace([np.inf, -np.inf],
                                                np.nan).fillna(method='ffill')
        return df_ret
    if data_type == 'test':
        stock_df = pd.read_pickle(os.path.join(input_dir, '20210805.pkl'))
        df_ret = df_ret.merge(stock_df, how='left', on='stock_id').replace(
            [np.inf, -np.inf], np.nan).fillna(method='ffill')
        return df_ret


def calc_rollingstats(rolling_x, roll_name):
    #统计量
    if len(rolling_x) > 0:
        roll_autocorr = rolling_x.groupby("time_id")[[roll_name,
                                                      "xpre"]].corr()
        roll_autocorr.reset_index(inplace=True)
        roll_autocorr = roll_autocorr.groupby("time_id").head(1)
        roll_autocorr.index = roll_autocorr["time_id"]
        del roll_autocorr["time_id"]

        roll_autocorr = pd.DataFrame(
            {roll_name + "_autocorr": roll_autocorr["xpre"]})

        roll_mean = pd.DataFrame({
            roll_name + "_mean":
            rolling_x.groupby("time_id")[roll_name].mean()
        })
        roll_std = pd.DataFrame({
            roll_name + "_std":
            rolling_x.groupby("time_id")[roll_name].std()
        })
        roll_skew = pd.DataFrame({
            roll_name + "_skew":
            rolling_x.groupby("time_id")[roll_name].skew()
        })
        roll_median = pd.DataFrame({
            roll_name + "_median":
            rolling_x.groupby("time_id")[roll_name].median()
        })

        data_merge = pd.merge(roll_mean,
                              roll_std,
                              left_index=True,
                              right_index=True,
                              how="inner")
        data_merge = pd.merge(data_merge,
                              roll_skew,
                              left_index=True,
                              right_index=True,
                              how="inner")
        data_merge = pd.merge(data_merge,
                              roll_autocorr,
                              left_index=True,
                              right_index=True,
                              how="inner")
        data_merge = pd.merge(data_merge,
                              roll_median,
                              left_index=True,
                              right_index=True,
                              how="inner")

    else:

        data_merge = pd.DataFrame([[np.nan, np.nan, np.nan, np.nan]])
        data_merge.columns = [
            roll_name + "_mean", roll_name + "_std", roll_name + "_skew",
            roll_name + "_autocorr", roll_name + "_median"
        ]

    return data_merge


def make_candle(df_data, price_name, vol_name, amt_name):

    df_data["pre"] = df_data.groupby("time_id")[price_name].shift(1)
    df_data["ret"] = df_data[price_name] / df_data["pre"] - 1
    df_data["absret"] = abs(df_data["ret"])
    df_retsum = pd.DataFrame(
        {"retsum": df_data.groupby("time_id")["ret"].sum()})
    df_absretsum = pd.DataFrame(
        {"absretsum": df_data.groupby("time_id")["absret"].sum()})

    df_data["absobv"] = df_data["absret"] * df_data[vol_name]
    df_obvabs = pd.DataFrame(
        {"xf4_abs": df_data.groupby("time_id")["absobv"].sum()})

    df_data["obv"] = df_data["ret"] * df_data[vol_name]
    df_obv = pd.DataFrame({"xf4": df_data.groupby("time_id")["obv"].sum()})

    df_amt = pd.DataFrame(
        {amt_name + "sum": df_data.groupby("time_id")[amt_name].sum()})
    df_vol = pd.DataFrame(
        {vol_name + "sum": df_data.groupby("time_id")[vol_name].sum()})

    df_mean = pd.DataFrame(
        {price_name + "mean": df_data.groupby("time_id")[price_name].mean()})
    df_median = pd.DataFrame(
        {price_name + "median": df_data.groupby("time_id")[price_name].median()})
    df_high = pd.DataFrame(
        {price_name + "high": df_data.groupby("time_id")[price_name].max()})
    df_low = pd.DataFrame(
        {price_name + "low": df_data.groupby("time_id")[price_name].min()})

    df_open = df_data.groupby("time_id").head(1)
    df_open.index = df_open["time_id"]
    df_open = pd.DataFrame({price_name + "open": df_open[price_name]})

    df_close = df_data.groupby("time_id").tail(1)
    df_close.index = df_close["time_id"]
    df_close = pd.DataFrame({price_name + "close": df_close[price_name]})

    df_candle = pd.merge(df_high,
                         df_low,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_mean,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_median,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_open,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_close,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_vol,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_amt,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_retsum,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_absretsum,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_obvabs,
                         left_index=True,
                         right_index=True,
                         how="inner")
    df_candle = pd.merge(df_candle,
                         df_obv,
                         left_index=True,
                         right_index=True,
                         how="inner")

    return df_candle


def cal_candlefactor(df_candle, price_name, vol_name, amt_name):
    f_name = price_name + "candle"
    #f1:illiq
    df_candle[f_name + "f1"] = (
        2 * (df_candle[price_name + "high"] - df_candle[price_name + "low"]) -
        abs(df_candle[price_name + "open"] -
            df_candle[price_name + "close"])) / df_candle[amt_name + "sum"]
    #f2 strength
    df_candle[f_name + "f2"] = df_candle["retsum"] / df_candle["absretsum"]
    #f3:ad
    df_candle[f_name + "f3"] =  (2 *df_candle[price_name + "close"] - df_candle[price_name + "low"]\
                    - df_candle[price_name + "high"] )/(df_candle[price_name + "high"] - df_candle[price_name + "low"]) \
                    * df_candle[vol_name + "sum"]
    #f3: obv
    df_candle[f_name + "f41"] = df_candle["xf4"] / df_candle[vol_name + "sum"]
    df_candle[f_name +
              "f42"] = df_candle["xf4_abs"] / df_candle[vol_name + "sum"]
    return df_candle


def calculate_features2(book_df, trade_df):
    """
    df: book_train data for each stock_id
    """
    #calculate price for features

    book_df['wap'] = (book_df['bid_price1'] * book_df['ask_size1'] +
                      book_df['ask_price1'] * book_df['bid_size1']) / (
                          book_df['bid_size1'] + book_df['ask_size1'])

    book_df["vol_ab"] = book_df['bid_size1'] + book_df['ask_size1']
    book_df["amt_ab"] = book_df['bid_price1'] * book_df['ask_size1'] + book_df[
        'ask_price1'] * book_df['bid_size1']

    book_df["amt_a"] = book_df['ask_price1'] * book_df['ask_size1']
    book_df["amt_b"] = book_df['bid_price1'] * book_df['bid_size1']

    trade_df["amt"] = trade_df["price"] * trade_df["size"]

    #flag filter
    book_df["wap_pre"] = book_df.groupby("time_id")['wap'].shift(1)
    book_df["bid_ppre"] = book_df.groupby("time_id")['bid_price1'].shift(1)
    book_df["ask_ppre"] = book_df.groupby("time_id")['ask_price1'].shift(1)

    book_df["isBS"] = np.where(
        book_df["wap"] > book_df["wap_pre"], "B",
        np.where(book_df["wap"] < book_df["wap_pre"], "S", np.nan))
    book_df["isBS_big"] = np.where(
        book_df["wap"] > book_df["ask_ppre"], "supB",
        np.where(book_df["wap"] < book_df["bid_ppre"], "supS",
                 np.where(pd.notnull(book_df["wap"]), "midBS", np.nan)))

    ordersize50 = pd.DataFrame({
        "ordersize50":
        book_df.groupby("time_id")["amt_ab"].apply(lambda x: np.nanmedian(x))
    })
    ordersize50.reset_index(inplace=True)

    ordersize25 = pd.DataFrame({
        "ordersize25":
        book_df.groupby("time_id")["amt_ab"].apply(
            lambda x: np.nanpercentile(x, 75))
    })
    ordersize25.reset_index(inplace=True)

    ordersize75 = pd.DataFrame({
        "ordersize75":
        book_df.groupby("time_id")["amt_ab"].apply(
            lambda x: np.nanpercentile(x, 25))
    })
    ordersize75.reset_index(inplace=True)
    book_df1 = pd.merge(book_df, ordersize50, on="time_id", how="left")
    book_df1 = pd.merge(book_df1, ordersize25, on="time_id", how="left")
    book_df1 = pd.merge(book_df1, ordersize75, on="time_id", how="left")
    book_df1.loc[:, "isoversize50"] = np.where(
        book_df1["amt_ab"] > book_df1["ordersize50"], "up50",
        np.where(book_df1["amt_ab"] <= book_df1["ordersize50"], "down50",
                 np.nan))

    book_df1.loc[:, "isoversize75"] = np.where(
        book_df1["amt_ab"] > book_df1["ordersize75"], "up75",
        np.where(book_df1["amt_ab"] <= book_df1["ordersize75"], "down75",
                 np.nan))
    book_df1.loc[:, "isoversize25"] = np.where(
        book_df1["amt_ab"] > book_df1["ordersize25"], "up25",
        np.where(book_df1["amt_ab"] <= book_df1["ordersize25"], "down25",
                 np.nan))

    #不同波动率
    #calculate historical volatility
    vol = book_df1.groupby('time_id')['wap'].apply(
        lambda x: np.sqrt(np.sum(np.log(x).diff()**2)))
    vol_df = pd.DataFrame(vol)
    vol_df.rename(columns={'wap': 'vol_orig'}, inplace=True)
    data_merge_all = vol_df

    #修改波动率：
    #    rolling波动率均值，标准差，偏度，自相关
    #新指标,新指标均值，标准差，偏度，自相关

    #    roll_name0 = "roll_std"
    #    roll_window = 10
    #BS FLAG
    flagname = "B"
    filtername = "isBS"
    for filtername, flagname in [["isBS", "B"], ["isBS", "S"],
                                 ["isBS_big", "supB"], ["isBS_big", "supS"],
                                 ["isBS_big", "midBS"],
                                 ["isoversize50", "up50"],
                                 ["isoversize50", "down50"],
                                 ["isoversize25", "up25"],
                                 ["isoversize25", "down25"],
                                 ["isoversize75", "up75"],
                                 ["isoversize75", "down75"]]:
        print(filtername, flagname)
        book_df_new = book_df1[book_df1[filtername] == flagname]
        #个数
        df_fnum = pd.DataFrame({
            flagname + "num":
            book_df_new.groupby("time_id")["seconds_in_bucket"].count()
        })
        data_merge_all = pd.merge(data_merge_all,
                                  df_fnum,
                                  left_index=True,
                                  right_index=True,
                                  how="left")

        for roll_window in [5, 10]:
            #rolling指标
            price_name = "wap"
            roll_name0 = price_name + "roll_std"
            roll_name = roll_name0 + str(roll_window) + "_" + flagname

            rolling_x = pd.DataFrame({
                roll_name:
                book_df_new.groupby("time_id")[price_name].rolling(
                    roll_window).std()
            })
            rolling_x.reset_index(inplace=True)
            rolling_x.loc[:, "xpre"] = rolling_x.groupby(
                "time_id")[roll_name].shift(1)
            #计算统计量因子
            data_merge = calc_rollingstats(rolling_x, roll_name)
            data_merge_all = pd.merge(data_merge_all,
                                      data_merge,
                                      left_index=True,
                                      right_index=True,
                                      how="left")

        #全局做candle：wap，买盘，卖盘
        #candle因子
        price_name = "wap"
        vol_name = "vol_ab"
        amt_name = "amt_ab"
        df_data = cp.deepcopy(book_df_new)
        df_candle = make_candle(df_data, price_name, vol_name, amt_name)

        list_save = [
            price_name + "candlef1", price_name + "candlef2",
            price_name + "candlef3", price_name + "candlef41",
            price_name + "candlef42"
        ]

        list_save = [i + "_" + flagname for i in list_save]
        df_candle = cal_candlefactor(df_candle, price_name, vol_name, amt_name)

        col_orig = list(df_candle.columns)
        col_new = [i + "_" + flagname for i in col_orig]
        df_candle.columns = col_new

        data_merge_all = pd.merge(data_merge_all,
                                  df_candle[list_save],
                                  left_index=True,
                                  right_index=True,
                                  how="left")

    #加filter做candle
    #    切割，打flag，给权重， 加filter切历史，波动率，分买入卖出，大单小单，上行下行，主买主卖
    #    全天上行波动率/全天波动率
    #index 数据   #复杂
    #calculate max and min bid-ask spread
    del data_merge_all["vol_orig"]

    return data_merge_all


def RMSPEMetric(XGBoost=False):
    def RMSPE(yhat, dtrain, XGBoost=XGBoost):

        y = dtrain.get_label()
        elements = ((y - yhat) / y)**2
        if XGBoost:
            return 'RMSPE', float(np.sqrt(np.sum(elements) / len(y)))
        else:
            return 'RMSPE', float(np.sqrt(np.sum(elements) / len(y))), False

    return RMSPE


def rmspe(y_true, y_pred):
    return (np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def fit_model(params, X_train, y_train, features, cats=[], n_fold=10, seed=66):
    """
    模型训练
    """

    models = []
    oof_df = X_train[['time_id', 'stock_id']].copy()
    oof_df['target'] = y_train
    oof_df['pred'] = np.nan

    cv = model_selection.KFold(n_splits=n_fold,
                               shuffle=True,
                               random_state=seed)

    kf = cv.split(X_train, y_train)

    fi_df = pd.DataFrame()
    fi_df['features'] = features
    fi_df['importance'] = 0

    for fold_id, (train_index, valid_index) in tqdm(enumerate(kf)):
        # split
        X_tr = X_train.loc[train_index, features]
        X_val = X_train.loc[valid_index, features]
        y_tr = y_train.loc[train_index]
        y_val = y_train.loc[valid_index]

        # model (note inverse weighting)
        train_set = lgb.Dataset(X_tr,
                                y_tr,
                                categorical_feature=cats,
                                weight=1 / np.power(y_tr, 2))
        val_set = lgb.Dataset(X_val,
                              y_val,
                              categorical_feature=cats,
                              weight=1 / np.power(y_val, 2))
        model = lgb.train(params,
                          train_set,
                          valid_sets=[train_set, val_set],
                          feval=RMSPEMetric(),
                          verbose_eval=250)

        # save model
        joblib.dump(model, '../model/model_fold{}.pkl'.format(fold_id))
        oof_df['pred'].iloc[valid_index] = model.predict(X_val)

        print('model saved \n==================================')

    y_true = oof_df['target'].values
    y_pred = oof_df['pred'].values
    print(rmspe(y_true, y_pred))
    return oof_df


if __name__ == '__main__':
    pass