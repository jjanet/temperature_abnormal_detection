# -*- coding: UTF-8 -*-
"""
Project -> File: tempe -> plt_all
Author: PMR
Date: 2021/9/1 9:22
Desc: 内测版本
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


class PltAll:
    """
        绘图结果并保存
    """
    def __init__(self, col_name_map=None):
        if col_name_map is None:
            self.name_map = dict(time_col="记录时间", wind_speed="机舱气象站风速",
                                 wheel_speed="轮毂转速", power="变频器发电机侧功率")
        else:
            self.col_name_map = col_name_map
        self.name_map = col_name_map
        category_list = [self.col_name_map["wind_speed"], self.col_name_map['wheel_speed'], self.col_name_map['power']]
        self.category_list = category_list  # 工况分类字段名

    def gmm_scatter(self, handled_df, labels, wind_turbine_id, save_path):
        """高斯混合模型预测3D散点图

        Parameter
        ---------
        handled_df: DataFrame
            去噪、重采样后的数据

        labels:Series
            类别标签数据

        wind_turbine_id:str
            风机号

        save_path : Path
            存储路径

        """
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(handled_df[self.category_list[0]], handled_df[self.category_list[1]], s=15, c=labels)
        ax.set_xlabel(self.category_list[0], linespacing=4)
        ax.set_ylabel(self.category_list[1], linespacing=2.5)
        ax.set_zlabel(self.category_list[2], linespacing=2.5)
        plt.title("TB" + wind_turbine_id + "GMM_3d图", fontsize=25)
        plt.savefig(str(save_path.joinpath(wind_turbine_id, "_gmm3D.png")), transparent=True)

    @staticmethod
    def category_plot(split_df, plt_name, save_path, wind_turbine_id):
        """工况分类折线图

        Parameter
        ---------
        split_df: DataFrame
            根据字段划分后的数据矩阵

        plt_name:str
            分类字段名

        wind_turbine_id:str
            风机号

        save_path:Path
            存储路径

        """
        mean1 = split_df[0].to_numpy().mean()
        mean2 = split_df[1].to_numpy().mean()
        mean3 = split_df[2].to_numpy().mean()
        a = [i for i in range(22)]
        plt.figure(figsize=(20, 8))
        plt.plot(split_df[0].values, color="#778899", linestyle="-.")
        plt.plot(split_df[1].values, color="#778899", linestyle="-.")
        plt.plot(split_df[2].values, color="#778899", linestyle="-.")

        plt.title("各风机" + plt_name + "工况划分")
        plt.grid(1)
        plt.xticks(a, list("TB" + split_df.index.values), rotation=45)
        plt.axhline(mean1, color="#FFB6C1", label="工况1")
        plt.axhline(mean2, color="#FFB6C1", label="工况2")
        plt.axhline(mean3, color="#FFB6C1", label="工况3")
        b, c = 0, 21
        x = np.linspace(0, 22)
        xf = x[np.where((x >= b) & (x < c))]
        plt.fill_between(xf, mean1, mean3 + (mean3 - mean2), color="#B0C4DE", alpha=0.25)
        plt.legend(loc="best")
        plt.xlabel("风机")
        plt.ylabel(plt_name)
        plt.savefig(str(save_path.joinpath(wind_turbine_id, "_gmm3D.png")), transparent=True)

    @staticmethod
    def condition_krr_pre_result_plot(test_out, krr_pre, wind_turbine, condition_num, save_path):
        """子工况核岭回归建模预测效果图

        Parameter
        ---------
        krr_pre_ser：Series
            岭回归预测值

        test_out_df：DataFrame
            验证数据集/真实值

        wind_turbine:str
            风机号

        condition_num:str
            子工况编号

        save_path:Path
            存储路径

        """
        plt.figure(figsize=(20, 8))
        plt.plot(test_out.values, label='主轴承温度1真实值')
        plt.plot(krr_pre, label='krr预测值')
        plt.legend(loc="best")
        plt.title(wind_turbine + "工况" + condition_num + "krr_pre")
        plt.grid()
        plt.savefig(str(save_path.joinpath(wind_turbine, "工况", condition_num, "krr_pre.png")), transparent=True)
