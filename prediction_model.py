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
import itertools
import sklearn.metrics as metrics
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import LocalOutlierFactor
from condition_category import ConditionCategory

condition = ConditionCategory()


class PredictionModel:
    """
        预测模型
    """
    def __init__(self, lof_metric="euclidean", split_size=0.8, feature_name_map=None):
        if feature_name_map is None:
            self.feature_name_map = dict(time_col="记录时间", wind_speed="机舱气象站风速",
                                         wheel_speed="轮毂转速", power="变频器发电机侧功率",
                                         agl_to_wind="5秒偏航对风平均值", environment_tempe="测风塔环境温度",
                                         gen_tempe="发电机定子温度1", tempe_col="主轴承温度1")
        else:
            self.feature_name_map = feature_name_map
        feature_col = list(self.feature_name_map.values())[1:]
        self.lof_metric = lof_metric  # lof距离计算度量
        self.feature_col = feature_col  # 特征字段
        self.split_size = split_size  # 训练集测试集划分比例

    def algorithm_denoising(self, filter_df):
        """lof去噪

        Parameters
        ----------
        filter_df:DataFrame
            机理过滤后的数据矩阵

        Returns
        -------
        lof_df：DataFrame
            lof算法去噪后的数据矩阵

        """
        filter_df = filter_df[list(self.feature_name_map.values())]
        col_list = [self.feature_name_map["wind_speed"],
                    self.feature_name_map['wheel_speed'], self.feature_name_map['power']]
        combination = list(itertools.combinations(col_list, 2))
        col_name_list = []
        for index, col in enumerate(combination):
            clf = LocalOutlierFactor(metric=self.lof_metric)
            x = filter_df[list(col)].values
            col_name = "label" + str(index)
            col_name_list.append(col_name)
            filter_df[col_name] = clf.fit_predict(x)
        lof_df = filter_df[(filter_df[col_name_list[0]] >= 0)
                           & (filter_df[col_name_list[1]] >= 0) & (filter_df[col_name_list[2]] >= 0)]
        lof_df = lof_df.drop(col_name_list, axis=1)
        return lof_df

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
        if self.feature_name_map['power']:
            lof_df = lof_df[(lof_df[self.feature_name_map['power']] >= power_min)
                            & (lof_df[self.feature_name_map['power']] < power_max)].copy()
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
        model_construct_result_list : list
            建模后结果，包括测试集真实值、预测值、评价标准列表
        """

        feature_df = self.feature_generate(condition_df, frequency)
        split_num = int(feature_df.shape[0] * self.split_size)
        train_data_df, train_data_out_df, test_df, test_out_df = self.train_test_split(feature_df, split_num)
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

        """
        condition_df[self.feature_name_map['time_col']] \
            = pd.to_datetime(condition_df[self.feature_name_map['time_col']], format="%Y-%m-%d %H:%M:%S")
        condition_df.index = condition_df[self.feature_name_map['time_col']]
        condition_df = condition_df.drop(self.feature_name_map['time_col'], axis=1)
        resample_df_max = condition_df.resample(frequency).max()
        resample_df_min = condition_df.resample(frequency).min()
        resample_df_mean = condition_df.resample(frequency).mean()
        feature_df = pd.concat([resample_df_max, resample_df_min, resample_df_mean], axis=1)
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

        """
        resample_df = condition.drop_inf_and_nan(resample_df)
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
        x_train ： DataFrame
            训练集

        y_train: DataFrame
            训练集待预测数据列

        x_test: DataFrame
            测试集

        y_test: DataFrame
            测试集待预测数据列

        """
        feature_df = feature_df.drop([self.feature_name_map['tempe_col'] + '_max',
                                      self.feature_name_map['tempe_col'] + '_min'], axis=1)
        x_df = feature_df.drop([self.feature_name_map['tempe_col'] + '_mean'], axis=1)
        y_df = feature_df[self.feature_name_map['tempe_col'] + '_mean']

        x_train = x_df.iloc[:split_num]
        y_train = y_df.iloc[:split_num]
        x_test = x_df.iloc[split_num:]
        y_test = y_df.iloc[split_num:]

        return x_train, y_train, x_test, y_test

    @staticmethod
    def krr(x_train, y_train, x_test):
        """核岭回归

        Parameter
        ---------
        x_train: DataFrame
            输入训练数据集

        y_train: DataFrame
            输出训练数据集

        x_test: DataFrame
            输入测试数据

        Return
        ------
        krr_pre_ser：Series
            岭回归预测值

        """
        clf = KernelRidge(alpha=1.0)
        model = clf.fit(np.array(x_train), np.array(y_train).reshape(-1, 1))
        krr_pre_ser = model.predict(np.array(x_test))
        return krr_pre_ser

    @staticmethod
    def evaluate_model_result(krr_pre_ser, y_test):
        """预测结果评估指标结果计算

        Parameter
        ---------
        krr_pre_ser：Series
            核岭回归预测值

        test_out_df：DataFrame
            验证数据集/真实值

        Return
        ------
        evaluate_res：list
            预测评估指标结果

        """
        rmse = round(metrics.mean_absolute_error(krr_pre_ser, y_test), 2)
        mse = round(metrics.mean_squared_error(krr_pre_ser, y_test), 2)
        r2 = round(metrics.explained_variance_score(krr_pre_ser, y_test), 2)
        # 解释方差分
        evs = round(metrics.explained_variance_score(krr_pre_ser, y_test), 2)
        evaluate_res = [rmse, mse, r2, evs]
        return evaluate_res
