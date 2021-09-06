# -*- coding: UTF-8 -*-
"""
Project -> File: tempe -> all_dataset_condition_category_mean
Author: PMR
Date: 2021/9/1 9:22
Desc: 内测版本
"""

import pandas as pd


class AllDatasetConditionCategoryMean:
    """
        所有数据集分工况建模处理结果
    """
    def __init__(self, category_str="2D_GMM", col_name_map=None):
        if col_name_map is None:
            self.col_name_map = dict(time_col="记录时间", wind_speed="机舱气象站风速",
                                     wheel_speed="轮毂转速", power="变频器发电机测功率", tempe_col="主轴承温度1")
        else:
            self.col_name_map = col_name_map
        self.category_str = category_str  # 结果列表命名

    @staticmethod
    def sort_category_result(condition_df, col_name, sort_df):
        """对工况分类结果进行排序
         Parameter
        ---------
        condition_df: DataFrame
            全部风机工况分类结果数据

        col_name: str
            待排序字段名

        sort_df:DataFrame
            待拼接排序好的数据

        Return
        ------
        sorted_df：DataFrame
            排序好工况分类结果

        """
        sorted_df = pd.concat(
            [pd.DataFrame(sort_df.values), pd.DataFrame(
                condition_df[col_name].sort_values().values)],
            axis=1,
        )
        return sorted_df

    def category_result_mean_sum(self, sorted_df, wind_turbine_id_list):
        """不同字段工况分类均值汇总
        Parameter
        ---------
        sorted_df: DataFrame
            按工况分类结果排序好的数据

        wind_turbine_id_list: DataFrame:
            风机编号列表

        Return
        ------
        category_df：DataFrame
            按字段工况分类均值汇总结果

        split_wind_df:DataFrame
            按风速分类汇总

        split_wheel_df:DataFrame
            按轮毂转速分类汇总

        split_power_df:DataFrame
            按有功功率分类汇总
        """
        para_list = list(self.col_name_map.values())[1:4]
        split_wind_df = self.category_split(sorted_df, para_list[0], wind_turbine_id_list)
        split_wheel_df = self.category_split(sorted_df, para_list[1], wind_turbine_id_list)
        split_power_df = self.category_split(sorted_df, para_list[2], wind_turbine_id_list)
        wind_tmp_df = split_wind_df.to_numpy().mean()
        wheel_tmp_df = split_wheel_df.to_numpy().mean()
        power_tmp_df = split_power_df.to_numpy().mean()
        category_df = pd.concat([wind_tmp_df, wheel_tmp_df, power_tmp_df], axis=1)
        category_df.columns = para_list
        category_df.index = ["工况1", "工况2", "工况3"]
        category_df = round(category_df, 2)
        category_df.columns.name = self.category_str
        return category_df, split_wind_df, split_wheel_df, split_power_df

    @staticmethod
    def category_split(sorted_df, para, wind_turbine_id_list):
        """根据分类结果将矩阵命名转置，根据字段分成不同的dataframe

         Parameter
        ---------
        df: DataFrame
            已排序全部风机工况分类结果数据

        para: str
            分类字段名

        wind_turbine_id_cols:list
            风机编号列表

        Return
        ------
        split_df：DataFrame
            按字段划分完毕的工况分类结果

        """
        col_name = [i + para for i in wind_turbine_id_list]
        split_df = sorted_df[col_name].transpose()
        return split_df

    def condition_over_percentage(self, condition2_df, condition3_df):
        """计算工况2中超过工况3温度均值的数据个数占工况2总数的占比

        Parameter
        ---------
        df_condition2: DataFrame
            工况2数据
        df_condition3: DataFrame
            工况3数据

        Return
        ------
        percentage：float
            工况2中超过工况3温度均值的数据个数占工况2总数的占比

        """
        over_count = len(condition2_df[condition2_df[self.col_name_map["tempe_col"]] > condition3_df[
            self.col_name_map["tempe_col"]].mean()][self.col_name_map["tempe_col"]])
        percentage = round(
            over_count / len(condition2_df[self.col_name_map["tempe_col"]]), 3) * 100
        return percentage
