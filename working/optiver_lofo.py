# modified from: https://github.com/aerdem4/lofo-importance/blob/master/lofo/lofo_importance.py

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from utils import rmspe
import warnings

def lofo_to_df(lofo_scores, feature_list):
    importance_df = pd.DataFrame()
    importance_df["feature"] = feature_list
    importance_df["importance_mean"] = lofo_scores.mean(axis=1)
    importance_df["importance_std"] = lofo_scores.std(axis=1)

    for val_score in range(lofo_scores.shape[1]):
        importance_df["val_imp_{}".format(val_score)] = lofo_scores[:, val_score]

    return importance_df.sort_values("importance_mean", ascending=False)


def plot_importance(importance_df, figsize=(8, 8), kind="default"):
    """Plot feature importance
    Parameters
    ----------
    importance_df : pandas dataframe
        Output dataframe from LOFO/FLOFO get_importance
    kind : string
        plot type can be default or box
    figsize : tuple
    """
    importance_df = importance_df.copy()
    importance_df["color"] = (importance_df["importance_mean"] > 0).map({True: 'g', False: 'r'})
    importance_df.sort_values("importance_mean", inplace=True)

    available_kinds = {"default", "box"}
    if kind not in available_kinds:
        warnings.warn("{kind} not in {ak}. Setting to default".format(kind=kind, ak=available_kinds))

    if kind == "default":
        importance_df.plot(x="feature", y="importance_mean", xerr="importance_std",
                           kind='barh', color=importance_df["color"], figsize=figsize)
    elif kind == "box":
        lofo_score_cols = [col for col in importance_df.columns if col.startswith("val_imp")]
        features = importance_df["feature"].values.tolist()
        importance_df.set_index("feature")[lofo_score_cols].T.boxplot(column=features, vert=False, figsize=figsize)

class OptiverLOFO:
    def __init__(self, train, feature_cols, group_dict=None):
        self.train = train
        self.feature_cols = feature_cols
        if group_dict is None:
            self.group_dict = {f: [f] for f in feature_cols}
        else:
            self.group_dict = group_dict
        self.gkf = GroupKFold(5)
        
        self.base_cv_score = None
        self.lofo_cv_scores = None
        self.lofo_df = None
        
    def _get_cv_score(self, features_to_remove):
        lofo_cols = [f for f in self.feature_cols if f not in features_to_remove]
        cv_scores = np.zeros(5)
        for fold, (trn_ind, val_ind) in enumerate(self.gkf.split(self.train, groups=self.train.time_id)):
            x_train, x_val = self.train.loc[trn_ind, lofo_cols], self.train.loc[val_ind, lofo_cols]
            y_train, y_val = self.train.loc[trn_ind, 'target'], self.train.loc[val_ind, 'target']

            # Root mean squared percentage error weights
            train_weights = 1 / np.square(y_train)
            val_weights = 1 / np.square(y_val)

            # Fit with sklearn API
            model = lgb.LGBMRegressor(random_state=42, min_child_samples=int(0.01*self.train.shape[0]), objective='rmse')
            model.fit(x_train, 
                      y_train, 
                      sample_weight=train_weights,
                      eval_set=[(x_val, y_val)],
                      eval_sample_weight=[val_weights],
                      eval_metric='rmse',
                      early_stopping_rounds=100,
                      verbose=False)
            cv_scores[fold] = rmspe(y_val, model.predict(x_val))
        return cv_scores


    def get_importance(self):
        base_cv_score = self._get_cv_score(features_to_remove=[None])
        self.base_cv_score = base_cv_score
        print(f"base cv mean: {np.mean(base_cv_score)}")

        lofo_cv_scores = []
        feature_list = list(self.group_dict.keys())
        for group in tqdm(feature_list):
            lofo_cv_scores.append(self._get_cv_score(features_to_remove=self.group_dict[group]))
        self.lofo_cv_scores = lofo_cv_scores
        lofo_cv_scores_normalized = np.array([lofo_cv_score-base_cv_score for lofo_cv_score in lofo_cv_scores])
        self.lofo_df = lofo_to_df(lofo_cv_scores_normalized, feature_list)
        return self.lofo_df
    
class OptiverRecursiveLOFO:
    def __init__(self, train, feature_cols, group_dict=None, log_dir='recusive_lofo_log/'):
        self.train = train
        self.feature_cols = feature_cols
        if group_dict is None:
            self.group_dict = {f: [f] for f in feature_cols}
        else:
            self.group_dict = group_dict
        self.feature_cols_selected = self.feature_cols
        self.group_dict_selected = self.group_dict
        self.step = 0
        self.lofo_dfs = []
        
    def _select_one_round(self):    
        lofo = OptiverLOFO(self.train, self.feature_cols_selected, group_dict=self.group_dict_selected)
        lofo_df = lofo.get_importance()
        self.lofo_dfs.append(lofo_df)
        self.step += 1
    def recursive_select(self):
        lofo = OptiverLOFO(train, feature_cols, group_dict=feature_groups)