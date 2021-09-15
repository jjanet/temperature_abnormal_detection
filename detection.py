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
    def __init__(self, contamination=0.05, max_features=5, random_state=1, name_map=None):
        if name_map is None:
            self.feature_name_map = dict(time_col="记录时间", tempe_col="主轴承温度1")
        else:
            self.name_map = name_map
        self.contamination = contamination  # 数据集中异常值的比例
        self.max_feature = max_features  # 从输入中提取的特征数量来训练每个基础估计器
        self.random_state = random_state  # 控制每个分支步骤和森林中每棵树的特征和分裂值选择的伪随机性

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

    def isolation_forest(self, diff_feature_df, state):
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
        iforest = IsolationForest(contamination=self.contamination,
                                  max_features=self.max_feature, random_state=self.random_state)
        pred_labels = iforest.fit_predict(x)
        diff_feature_df["scores"] = iforest.decision_function(x)
        diff_feature_df["anomaly_label"] = pred_labels
        return pred_labels
