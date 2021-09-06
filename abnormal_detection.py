# -*- coding: UTF-8 -*-
"""
Project -> File: tempe -> abnormal_detection
Author: PMR
Date: 2021/9/1 9:22
Desc: 内测版本
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from prediction_model import PredictionModel

model = PredictionModel()


class AbnormalDetection:
    """
        异常检测
    """
    def __init__(self, name_map=None):
        if name_map is None:
            self.feature_name_map = dict(time_col="记录时间", tempe_col="主轴承温度1")
        else:
            self.name_map = name_map

    def diff_feature_generation(self, test_out_df, krr_pre, frequency):
        """残差特征提取

        Parameter
        ---------
        krr_pre_ser：Series
            岭回归预测值

        test_out_df：DataFrame
            验证数据集/真实值

        frequency:str
            采样频率

        Return
        ------
        diff_feature_df：DataFrame
            残差特征矩阵

        """
        tempe_df = test_out_df.reset_index().drop(['index'], axis=1)
        pre_df = pd.DataFrame(krr_pre, columns=['krr_pre'])
        diff_df = pd.concat([tempe_df, pre_df], axis=1)
        diff_df['diff'] = diff_df[self.name_map['tempe_col'] +
                                  '_mean'] - diff_df['krr_pre']
        max_df = diff_df['diff'].resample(frequency).max()
        min_df = diff_df['diff'].resample(frequency).min()
        mean_df = diff_df['diff'].resample(frequency).mean()
        std_df = diff_df['diff'].resample(frequency).std()
        mid_df = diff_df['diff'].resample(frequency).median()
        resample_df = pd.concat(
            [max_df, min_df, mean_df, std_df, mid_df], axis=1
        )
        resample_df.columns = ['max', 'min', 'mean', 'std', 'mid']
        diff_feature_df = model.feature_handle(resample_df)
        return diff_feature_df

    @staticmethod
    def isolation_forest(diff_feature_df, state):
        """孤立森林异常检测模型

        Parameter
        ---------
        diff_feature_df: DataFrame
            输入残差特征
        state:int
            初始状态值

        Return
        ------
        pred_labels:Series
            异常标签值

        """
        diff_feature_df["state"] = state
        x = diff_feature_df.drop(["state"], axis=1).values
        iforest = IsolationForest(
            n_estimators=100,
            max_samples="auto",
            contamination=0.05,
            max_features=5,
            bootstrap=False,
            n_jobs=-1,
            random_state=1,
        )
        pred_labels = iforest.fit_predict(x)
        diff_feature_df["scores"] = iforest.decision_function(x)
        diff_feature_df["anomaly_label"] = pred_labels
        return pred_labels
