# -*- coding: UTF-8 -*-
"""
Project -> File: tempe -> condition_category
Author: PMR
Date: 2021/9/1 9:22
Desc: 内测版本
"""

import numpy as np
import pandas as pd
import itertools
from sklearn import mixture
from sklearn.neighbors import LocalOutlierFactor


class ConditionCategory:
    """
        分工况建模
    """
    def __init__(self, wheel_speed_threshold=8, wind_turbine_current_state=6, gen_power_threshold=2000,
                 power_threshold=2000, lof_metric="euclidean", gmm_n_components=3, gmm_random_state=0,
                 category_str="2D_GMM", col_name_map=None):
        if col_name_map is None:
            self.col_name_map = dict(time_col="记录时间", wind_speed="机舱气象站风速",
                                     wheel_speed="轮毂转速", power="变频器发电机测功率",
                                     turbine_state="风机当前状态值", gen_power_threshold="发电机功率限幅值",
                                     power_threshold="风机功率限制设定值")
        else:
            self.col_name_map = col_name_map
        self.wheel_speed_threshold = wheel_speed_threshold  # 轮毂转速限定值
        self.wind_turbine_current_state = wind_turbine_current_state  # 风机当前状态值
        self.gen_power_threshold = gen_power_threshold  # 发电机功率限幅值
        self.power_threshold = power_threshold  # 风机功率限制设定值
        category_list = [self.col_name_map['wheel_speed'], self.col_name_map['power']]
        self.category_list = category_list  # 分类依据数据列
        self.lof_metric = lof_metric  # lof距离计算度量
        self.gmm_n_components = gmm_n_components  # 混合高斯模型的个数
        self.gmm_random_state = gmm_random_state  # 随机数发生器
        self.category_str = category_str  # 分类命名

    def mechanism_filter(self, df):
        """基于机理的数据筛选

        Parameters
        ----------
        df : DataFrame
            原始数据矩阵

        Returns
        -------
        data : DataFrame
            过滤掉风机限电、轮毂转速与液压制动力过小的数据，并仅保留风机正在发电的数据
        """

        # 过滤掉风机限电,风机正在发电且不偏航的数据为必要过滤条件
        data = df[(df[self.col_name_map['turbine_statue']] ==
                   self.wind_turbine_current_state)].copy()

        # 剩余过滤条件视风场字段情况
        if self.col_name_map['wheel_speed']:
            data = data[data[self.col_name_map['wheel_speed']] >= self.wheel_speed_threshold].copy()
        if self.col_name_map['power_threshold']:
            data = data[data[self.col_name_map['power_threshold']] == self.power_threshold].copy()
        if self.col_name_map['gen_power_threshold']:
            data = data[data[self.col_name_map['gen_power_threshold']] > self.gen_power_threshold].copy()
        return data

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
        col_list = [
            self.col_name_map["wind_speed"],
            self.col_name_map['wheel_speed'],
            self.col_name_map['power']]
        combination = list(itertools.combinations(col_list, 2))
        col_name_list = []
        for index, col in enumerate(combination):
            clf = LocalOutlierFactor(metric=self.lof_metric)
            x = filter_df[list(col)].values
            col_name = "label" + str(index)
            col_name_list.append(col_name)
            filter_df[col_name] = clf.fit_predict(x)
        lof_df = filter_df[filter_df[col_name_list] >= 0]
        return lof_df

    def gmm_raw_data_handing(self, lof_df, frequency):
        """高斯混合模型特征提取，去除极值、空值、重采样

        Parameters
        ----------
        lof_df:DataFrame
            lof算法去噪后的数据矩阵

        frequency:str
            采样频率

        Returns
        -------
        handled_df：DataFrame
            预处理完毕的数据矩阵
        """
        lof_df[self.col_name_map['time_col']] = pd.to_datetime(lof_df[self.col_name_map['time_col']],
                                                               format="%Y-%m-%d %H:%M:%S")
        lof_df.index = lof_df[self.col_name_map['time_col']].values
        resample_df = lof_df.resample(frequency).mean()
        handled_df = self.drop_inf_and_nan(resample_df)
        return handled_df

    @staticmethod
    def drop_inf_and_nan(has_inf_and_nan_df):
        """去除极值、空值

        Parameters
        ----------
        has_inf_and_nan_df:DataFrame
            可能有空值或极值的数据

        Returns
        -------
        has_inf_and_nan_df：DataFrame
            去除极值、空值之后的数据
        """
        has_inf_and_nan_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        drop_inf_and_nan_df = has_inf_and_nan_df.dropna(axis=0, how="any", inplace=True)
        return drop_inf_and_nan_df

    def gaussian_mixture_model(self, handled_df):
        """高斯混合模型预测

        Parameter
        ---------
        handled_df: DataFrame
            去噪、重采样后的数据

        Return
        ------
        labels: Series
            输出预测的样本标签值

        """
        x = handled_df[self.category_list].values
        gmm = mixture.GaussianMixture(n_components=self.gmm_n_components, random_state=self.gmm_random_state).fit(x)
        labels = gmm.predict(x)
        return labels

    def condition_category_result(self, handled_df, labels, wind_turbine):
        """提取风机各工况均值结果

        Parameter
        ---------
        handled_df: DataFrame
            数据预处理后的数据

        labels: Series
            预测的样本标签值

        wind_turbine:str
            风机号

        Return
        ------
        res_df：DataFrame
            工况平均值特征

        """
        feature_cols = list(self.col_name_map.values())[1:]
        handled_df["label"] = labels
        tmp_df = handled_df.groupby("label").describe()
        rename_feature_cols = [wind_turbine + col for col in feature_cols]
        res_df = tmp_df.loc["mean", feature_cols[:3]]
        res_df.columns = rename_feature_cols
        return res_df
