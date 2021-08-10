SEED = 42
import numpy as np
import pandas as pd
from utils import read_train_test, get_time_stock
from optiver_lofo import OptiverRecursiveLOFO, plot_importance
import itertools
import gc

train, _ = read_train_test()
df_book = pd.read_csv('../input/processed-book-ffill/df_book.csv')
df_trade = pd.read_csv('../input/processed-book-ffill/df_trade.csv')
train_ = df_book.merge(df_trade, on=['row_id'], how='left')
train = train.merge(train_, on=['row_id'], how='left')
del _, df_book, df_trade, train_
gc.collect()
# Get group stats of time_id and stock_id
train = get_time_stock(train)
train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)
print("train shape:", train.shape)

print("group level 1")
feature_cols = [c for c in train.columns if c not in [
    'row_id', 'target', 'time_id', 'stock_id', 'target', 'logtarget']]
# wap1_cols = [c for c in feature_cols if c.split('_')[0]=='wap1']
# feature_cols = [c for c in feature_cols if c not in wap1_cols]
print(f"# features: {len(feature_cols)}")
feature_groups = {
    'wap1': [c for c in feature_cols if c.split('_')[0] == 'wap1'],
    'wap2': [c for c in feature_cols if c.split('_')[0] == 'wap2'],
    'log_return1': [c for c in feature_cols if c.split('_')[0:2] == ['log', 'return1'] and c.split('_')[-1] not in ['time', 'stock']],
    'log_return2': [c for c in feature_cols if c.split('_')[0:2] == ['log', 'return2'] and c.split('_')[-1] not in ['time', 'stock']],
    'wap_balance': [c for c in feature_cols if c.split('_')[0:2] == ['wap', 'balance']],
    'price_spread': [c for c in feature_cols if c.split('_')[0:2] == ['price', 'spread']],
    'bid_spread': [c for c in feature_cols if c.split('_')[0:2] == ['bid', 'spread']],
    'ask_spread': [c for c in feature_cols if c.split('_')[0:2] == ['ask', 'spread']],
    'total_volume': [c for c in feature_cols if c.split('_')[0:2] == ['total', 'volume']],
    'volume_imbalance': [c for c in feature_cols if c.split('_')[0:2] == ['volume', 'imbalance']],
    'trade_log_return': [c for c in feature_cols if c.split('_')[0:3] == ['trade', 'log', 'return'] and c.split('_')[-1] not in ['time', 'stock']],
    'trade_seconds_in_bucket': [c for c in feature_cols if c.split('_')[0:4] == ['trade', 'seconds', 'in', 'bucket']],
    'trade_size': [c for c in feature_cols if c.split('_')[0:2] == ['trade', 'size']],
    'trade_order_count': [c for c in feature_cols if c.split('_')[0:3] == ['trade', 'order', 'count']],
    'timeagg': [c for c in feature_cols if c.split('_')[-1] == 'time'],
    'stockagg': [c for c in feature_cols if c.split('_')[-1] == 'stock'],

    #     '150': [c for c in feature_cols if '150' in c],
    #     '300': [c for c in feature_cols if '300' in c],
    #     '450': [c for c in feature_cols if '450' in c],
}
grouped_features = list(itertools.chain.from_iterable(feature_groups.values()))
print(len(grouped_features))
print(len(set(grouped_features)))


print("group level 2")
feature_groups_level2 = {}
lags = ['150', '300', '450']
for k in feature_groups:
    feature_groups_level2[k] = []
    for lag in lags:
        feature_groups_level2[k + '_' + lag] = []
    for v in feature_groups[k]:
        for lag in lags:
            islag = False
            if lag in v:
                feature_groups_level2[k + '_' + lag].append(v)
                islag = True
                break
        if not islag:
            feature_groups_level2[k].append(v)
grouped_features = list(itertools.chain.from_iterable(
    feature_groups_level2.values()))
print(len(grouped_features))
print(len(set(grouped_features)))

print("group level 3")
feature_groups_level3 = {}
cats = ['log_return1', 'log_return2', 'trade_log_return']
tosplit = ['timeagg', 'stockagg']
for k in feature_groups_level2:
    if k.split('_')[0] not in tosplit:
        feature_groups_level3[k] = feature_groups_level2[k]
    else:
        for cat in cats:
            feature_groups_level3[k + '_' +
                                  cat] = [c for c in feature_groups_level2[k] if cat in c]
grouped_features = list(itertools.chain.from_iterable(
    feature_groups_level3.values()))
print(len(grouped_features))
print(len(set(grouped_features)))
print(feature_groups_level3)

recur_lofo = OptiverRecursiveLOFO(
    train, feature_cols, group_dict=feature_groups_level3)
recur_lofo.recursive_select()
