SEED = 2000

import os
import glob
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import scipy as sc
from sklearn.model_selection import KFold, GroupKFold
import lightgbm as lgb
import warnings
from utils import get_feature_groups
import itertools
import optuna
from utils import read_train_test, get_time_stock, rmspe
warnings.filterwarnings('ignore')
pd.set_option('max_columns', 300)
pd.set_option('max_rows', 300)

train, _ = read_train_test()
df_book = pd.read_csv('../input/processed-book-ffill/df_book.csv')
df_trade = pd.read_csv('../input/processed-book-ffill/df_trade.csv')
train_ = df_book.merge(df_trade, on = ['row_id'], how = 'left')
train = train.merge(train_, on = ['row_id'], how = 'left')
train = get_time_stock(train)
train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)

feature_groups = get_feature_groups(train)
pruned_groups = ["timeagg_450_log_return1",
    "trade_seconds_in_bucket_450",
    "wap1_300",
    "timeagg_450_trade_log_return",
    "trade_seconds_in_bucket",
    "ask_spread_450"]
feature_cols = list(itertools.chain.from_iterable([c for g, c in feature_groups.items() if g not in pruned_groups]))
print(f"# features: {len(feature_cols)}")

kfold = GroupKFold(n_splits=5)
cv = list(kfold.split(train, groups=train.time_id))
# print(cv)
print(len(cv[0][0]), len(cv[0][1]))

def run_train(param, fold):
    trn_ind, val_ind = cv[fold][0], cv[fold][1]
    x_train, x_val = train.loc[trn_ind, feature_cols], train.loc[val_ind, feature_cols]
    y_train, y_val = train.loc[trn_ind, 'target'], train.loc[val_ind, 'target']

    # Root mean squared percentage error weights
    train_weights = 1 / np.square(y_train)
    val_weights = 1 / np.square(y_val)

    # Fit with sklearn API
    model = lgb.LGBMRegressor(**param)
    model.fit(x_train, 
              y_train, 
              sample_weight=train_weights,
              eval_set=[(x_val, y_val)],
              eval_sample_weight=[val_weights],
              eval_metric='rmse',
              early_stopping_rounds=100,
              verbose=100)

    return rmspe(y_val, model.predict(x_val))

def objective(trial):
    param = {
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 10000),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 2, 2048),
        "colsample_bytree": trial.suggest_float("feature_fraction", 0.2, 1.0), # feature_fraction
        "subsample": trial.suggest_float("bagging_fraction", 0.2, 1.0), # bagging_fraction
        "subsample_freq": trial.suggest_int("bagging_freq", 1, 50), # bagging_freq
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 10000, log=True),
#         "max_depth": trial.suggest_int("min_child_samples", 5, 10000),
        "objective": "rmse",
        "verbose": -1,
        "random_state": SEED,
    }
    cv_scores = []
    for fold in range(5):
        score = run_train(param, fold)
        cv_scores.append(score)
        trial.report(score, fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return np.mean(cv_scores)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1000)

print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

with open(ROOT / "best_params.json", "w") as f:
    json.dump({"rmspe":trial.value, "params":trial.params}, f, indent=4)
