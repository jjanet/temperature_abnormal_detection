# -*- coding: UTF-8 -*-
"""
Project -> File: tempe -> prediction_model
Author: PMR
Date: 2021/9/1 9:22
Desc: 内测版本
"""

import math
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from sklearn.linear_model import Ridge
from condition_category import ConditionCategory as Condition
from sklearn.preprocessing import MinMaxScaler


class PredictionModel:
    """
        预测模型
    """
    def __init__(self, split_size=0.8, feature_name_map=None):
        if feature_name_map is None:
            self.feature_name_map = dict(time_col="记录时间", wind_speed="机舱气象站风速",
                                         wheel_speed="轮毂转速", power="变频器发电机测功率",
                                         agl_to_wind="偏航对风", environment_tempe="测风塔环境温度",
                                         gen_tempe="发电机定子温度1", tempe_col="主轴承温度1")
        else:
            self.feature_name_map = feature_name_map
        feature_col = list(self.feature_name_map.values())[1:]
        self.feature_col = feature_col  # 特征字段
        self.split_size = split_size  # 训练集测试集划分比例

    def condition_split(self, lof_df, wheel_speed_min, wind_speed_min, power_min,
                        wheel_speed_max=math.inf, wind_speed_max=math.inf, power_max=math.inf):
        """工况分类

        Parameter
        ---------
        lof_df: DataFrame
            去噪后数据

        wheel_speed_min: float
            子工况轮毂转速最小值

        wind_speed_min: float
            子工况风速最小值

        power_min: float
            子工况功率最小值

        wheel_speed_max: float
            子工况轮毂转速最大值

        wind_speed_max: float
            子工况轮毂转速最大值

        power_max: float
            子工况轮毂转速最大值

        Return
        ------
        lof_df：DataFrame
            处理好的子工况数据

        """

        if self.feature_name_map['wheel_speed']:
            lof_df = lof_df[(lof_df[self.feature_name_map['wheel_speed']] >= wheel_speed_min)
                            & (lof_df[self.feature_name_map['wheel_speed']] < wheel_speed_max)].copy()
        if self.feature_name_map['wind_speed']:
            lof_df = lof_df[(lof_df[self.feature_name_map['wind_speed']] >= wind_speed_min)
                            & (lof_df[self.feature_name_map['wind_speed']] < wind_speed_max)].copy()
        if self.feature_name_map['power_speed']:
            lof_df = lof_df[(lof_df[self.feature_name_map['power_speed']] >= power_min)
                            & (lof_df[self.feature_name_map['power_speed']] < power_max)].copy()
        return lof_df

    def condition_model_construct(self, condition_df, frequency):
        """分工况建模

        Parameter
        ---------
        condition_df: DataFrame
            输入初步机理过滤并去噪后的子工况数据

        frequency:str
            重采样频率

        Return
        ------
        krr_pre：Series
            岭回归预测结果

        test_out_df：DataFrame
            验证数据集

        evaluate_res：list
            预测评估指标结果

        time_index：DataFrame
            时间索引
        """

        feature_df = self.feature_generate(condition_df, frequency)
        split_num = int(feature_df.shape[0] * self.split_size)
        train_data_df, train_data_out_df, test_df, test_out_df = self.train_test_split(
            feature_df, split_num)
        krr_pre = self.krr(train_data_df, train_data_out_df, test_df)
        evaluate_res = self.evaluate_model_result(krr_pre, test_out_df)
        model_construct_result_list = [test_out_df, krr_pre, evaluate_res]
        return model_construct_result_list

    def feature_generate(self, condition_df, frequency):
        """分工况建模

        Parameter
        ---------
        condition_df: DataFrame
            输入初步机理过滤并去噪后的子工况数据

        frequency:str
            重采样频率

        Return
        ------
        feature_df：DataFrame
            生成的特征数据

        time_index：DataFrame
            时间索引
        """
        condition_df[self.feature_name_map['time_col']] = pd.to_datetime(
            condition_df[self.feature_name_map['time_col']], format="%Y-%m-%d %H:%M:%S")
        condition_df.index = condition_df[self.feature_name_map['time_col']].values
        resample_df_max = condition_df[self.feature_col].resample(
            frequency).max()
        resample_df_min = condition_df[self.feature_col].resample(
            frequency).min()
        resample_df_mean = condition_df[self.feature_col].resample(
            frequency).mean()
        feature_df = pd.concat(
            [resample_df_max, resample_df_min, resample_df_mean], axis=1
        )
        feature_df = self.feature_handle(feature_df)
        return feature_df

    def feature_handle(self, resample_df):
        """特征数据处理，包括去除极值，去除空值，归一化

        Parameter
        ---------
        resample_df: DataFrame
            输入采样完毕的数据

        Return
        ------
        std_df：DataFrame
            经剔除异常值、归一化之后的特征矩阵

        time_index：DataFrame
            时间索引

        """
        resample_df = Condition.drop_inf_and_nan(resample_df)
        time_index = resample_df.index
        tmp_df = resample_df.reset_index().drop(["index"], axis=1)
        std_df = MinMaxScaler().fit_transform(tmp_df)
        std_df = pd.DataFrame(std_df)
        std_df.columns = ([i + "_max" for i in self.feature_col] + [i + "_min" for i in self.feature_col]
                          + [i + "_mean" for i in self.feature_col])
        std_df.index = time_index
        return std_df

    def train_test_split(self, feature_df, split_num):
        """划分训练集和测试集

        Parameter
        ---------
        feature_df: DataFrame
            输入处理好的特征数据

        split_num: int
            划分数据集标准

        Return
        ------
        train_data_df: DataFrame
            训练集
        train_data_out_df: DataFrame
            训练集待预测数据列
        test_data_df: DataFrame
            测试集
        test_out_df: DataFrame
            测试集待预测数据列

        """
        feature_df = feature_df.drop(
            [
                self.feature_name_map['tempe_col'] +
                '_max',
                self.feature_name_map['tempe_col'] +
                '_min'],
            axis=1)
        train_data_df = feature_df.drop(
            [self.feature_name_map['tempe_col'] + '_mean'], axis=1).iloc[:split_num]
        train_data_out_df = feature_df[self.feature_name_map['tempe_col'] +
                                       '_mean'].iloc[:split_num]
        test_data_df = feature_df.drop(
            [self.feature_name_map['tempe_col'] + '_mean'], axis=1).iloc[split_num:]
        test_out_df = feature_df[self.feature_name_map['tempe_col'] +
                                 '_mean'].iloc[split_num:]
        return train_data_df, train_data_out_df, test_data_df, test_out_df

    @staticmethod
    def krr(train_df, train_out_df, test_df):
        """岭核回归

        Parameter
        ---------
        train_df: DataFrame
            输入训练数据集
        train_df_out: DataFrame
            输出训练数据集

        test_df: DataFrame
            输入测试数据

        Return
        ------
        krr_pre_ser：Series
            岭回归预测值

        """
        clf = Ridge(alpha=1.0)
        model = clf.fit(np.array(train_df), np.array(
            train_out_df).reshape(-1, 1))
        krr_pre_ser = model.predict(np.array(test_df))
        return krr_pre_ser

    @staticmethod
    def evaluate_model_result(krr_pre_ser, tes_out_df):
        """预测结果评估指标结果计算

        Parameter
        ---------
        krr_pre_ser：Series
            岭回归预测值

        test_out_df：DataFrame
            验证数据集/真实值

        Return
        ------
        evaluate_res：list
            预测评估指标结果

        """
        rmse = round(metrics.mean_absolute_error(krr_pre_ser, tes_out_df), 2)
        mse = round(metrics.mean_squared_error(krr_pre_ser, tes_out_df), 2)
        r2 = round(
            metrics.explained_variance_score(
                krr_pre_ser, tes_out_df), 2)
        # 解释方差分
        evs = round(
            metrics.explained_variance_score(
                krr_pre_ser, tes_out_df), 2)
        evaluate_res = [rmse, mse, r2, evs]
        return evaluate_res
