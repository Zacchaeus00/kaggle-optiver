import numpy as np
import pandas as pd

# Function to read our base train and test set
def read_train_test():
    train = pd.read_csv('../input/optiver-realized-volatility-prediction/train.csv')
    test = pd.read_csv('../input/optiver-realized-volatility-prediction/test.csv')
    # Create a key to merge with book and trade data
    train['row_id'] = train['stock_id'].astype(str) + '-' + train['time_id'].astype(str)
    test['row_id'] = test['stock_id'].astype(str) + '-' + test['time_id'].astype(str)
    print(f'Our training set has {train.shape[0]} rows')
    return train, test

# Function to get group stats for the stock_id and time_id
def get_time_stock(df):
    # Get realized volatility columns
    vol_cols = ['log_return1_realized_volatility', 'log_return2_realized_volatility', 'log_return1_realized_volatility_450', 'log_return2_realized_volatility_450', 
                'log_return1_realized_volatility_300', 'log_return2_realized_volatility_300', 'log_return1_realized_volatility_150', 'log_return2_realized_volatility_150', 
                'trade_log_return_realized_volatility', 'trade_log_return_realized_volatility_450', 'trade_log_return_realized_volatility_300', 'trade_log_return_realized_volatility_150']

    # Group by the stock id
    df_stock_id = df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', 'median']).reset_index()
    # Rename columns joining suffix
    df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
    df_stock_id = df_stock_id.add_suffix('_' + 'stock')

    # Group by the time id
    df_time_id = df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min', 'median']).reset_index()
    # Rename columns joining suffix
    df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
    df_time_id = df_time_id.add_suffix('_' + 'time')
    
    # Merge with original dataframe
    df = df.merge(df_stock_id, how = 'left', left_on = ['stock_id'], right_on = ['stock_id__stock'])
    df = df.merge(df_time_id, how = 'left', left_on = ['time_id'], right_on = ['time_id__time'])
    df.drop(['stock_id__stock', 'time_id__time'], axis = 1, inplace = True)
    return df

# Function to calculate the root mean squared percentage error
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))

def get_feature_groups(train):
    feature_cols = [c for c in train.columns if c not in ['row_id', 'target', 'time_id', 'stock_id', 'target', 'logtarget']]
    # wap1_cols = [c for c in feature_cols if c.split('_')[0]=='wap1']
    # feature_cols = [c for c in feature_cols if c not in wap1_cols]
    print(f"# features: {len(feature_cols)}")
    feature_groups = {
        'wap1': [c for c in feature_cols if c.split('_')[0]=='wap1'],
        'wap2': [c for c in feature_cols if c.split('_')[0]=='wap2'],
        'log_return1': [c for c in feature_cols if c.split('_')[0:2]==['log', 'return1'] and c.split('_')[-1] not in ['time', 'stock']],
        'log_return2': [c for c in feature_cols if c.split('_')[0:2]==['log', 'return2'] and c.split('_')[-1] not in ['time', 'stock']],
        'wap_balance': [c for c in feature_cols if c.split('_')[0:2]==['wap', 'balance']],
        'price_spread': [c for c in feature_cols if c.split('_')[0:2]==['price', 'spread']],
        'bid_spread': [c for c in feature_cols if c.split('_')[0:2]==['bid', 'spread']],
        'ask_spread': [c for c in feature_cols if c.split('_')[0:2]==['ask', 'spread']],
        'total_volume': [c for c in feature_cols if c.split('_')[0:2]==['total', 'volume']],
        'volume_imbalance': [c for c in feature_cols if c.split('_')[0:2]==['volume', 'imbalance']],
        'trade_log_return': [c for c in feature_cols if c.split('_')[0:3]==['trade', 'log', 'return'] and c.split('_')[-1] not in ['time', 'stock']],
        'trade_seconds_in_bucket': [c for c in feature_cols if c.split('_')[0:4]==['trade', 'seconds', 'in', 'bucket']],    
        'trade_size': [c for c in feature_cols if c.split('_')[0:2]==['trade', 'size']],
        'trade_order_count': [c for c in feature_cols if c.split('_')[0:3]==['trade', 'order', 'count']],
        'timeagg': [c for c in feature_cols if c.split('_')[-1]=='time'],
        'stockagg': [c for c in feature_cols if c.split('_')[-1]=='stock'],

    #     '150': [c for c in feature_cols if '150' in c],
    #     '300': [c for c in feature_cols if '300' in c],
    #     '450': [c for c in feature_cols if '450' in c],
    }
    feature_groups_level2 = {}
    lags = ['150', '300', '450']
    for k in feature_groups:
        feature_groups_level2[k] = []
        for lag in lags:
            feature_groups_level2[k+'_'+lag] = []
        for v in feature_groups[k]:
            for lag in lags:
                islag = False
                if lag in v:
                    feature_groups_level2[k+'_'+lag].append(v)
                    islag = True
                    break
            if not islag:
                feature_groups_level2[k].append(v)

    feature_groups_level3 = {}
    cats = ['log_return1', 'log_return2', 'trade_log_return']
    tosplit = ['timeagg', 'stockagg']
    for k in feature_groups_level2:
        if k.split('_')[0] not in tosplit:
            feature_groups_level3[k] = feature_groups_level2[k]
        else:
            for cat in cats:
                feature_groups_level3[k+'_'+cat] = [c for c in feature_groups_level2[k] if cat in c]
                
    return feature_groups_level3