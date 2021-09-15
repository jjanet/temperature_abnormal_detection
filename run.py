# -*- coding: UTF-8 -*-
"""
Project -> File: tempe -> run
Author: PMR
Date: 2021/9/1 9:22
Desc: 内测版本
"""

import gc
import pandas as pd
from pathlib import Path
from condition_category import ConditionCategory
from all_dataset_condition_category_mean import AllDatasetConditionCategoryMean
from prediction_model import PredictionModel
from plt_all import PltAll
from abnormal_detection import AbnormalDetection

condition = ConditionCategory()
all_condition = AllDatasetConditionCategoryMean()
model = PredictionModel()
all_plt = PltAll()
detection = AbnormalDetection()


def get_data_name_list(base_path):
    """读取数据集目录所有哦文件并生成风机数据集列表

    Parameters
    ----------
    base_path : Path
        存放数据文件的文件夹路径

    Returns
    -------
    data_list : list
        返回指定文件夹下文件夹列表

    """
    file_list = base_path.glob("*.csv")
    data_list = [file.stem for file in file_list]
    return data_list


def join_save_path(sub_path, base_path):
    """拼接生成存储路径

    Parameters
    ----------
    sub_path : str
        文件命名子路径

    base_path : Path
        文件根路径

    Returns
    -------
    save_path : Path
        文件存储路径

    """
    base_path.joinpath(sub_path).mkdir(exist_ok=True)  # return void
    save_path = base_path.joinpath(sub_path)
    return save_path


def data_pre_processing(original_df):
    """数据预处理

    Parameters
    ----------
    original_df : DataFrame

    Returns
    -------
    lof_df : DataFrame
        去噪后数据

    """
    filter_df = condition.mechanism_filter(original_df)
    lof_df = model.algorithm_denoising(filter_df)
    return lof_df


def condition_category_result_split(sorted_df, data_list_reset, gmm_plot_picture_path):
    """划分工况结果

    Parameters
    ----------
    sorted_df : DataFrame
        排序好的工况分类数据

    data_list_reset : list
        重置后的不为空的风机数据集列表

    gmm_plot_picture_path : Path
        所有的风机数据折线图存储路径

    Returns
    -------
    condition_category_threshold_list : list
        按字段划分结果阈值
    """

    category_df, split_wind_df, split_wheel_df, split_power_df \
        = all_condition.category_result_mean_sum(sorted_df, data_list_reset)
    # 按工况划分分类结果
    condition1_res = category_df.iloc[0]
    condition2_res = category_df.iloc[1]
    condition3_res = category_df.iloc[2]
    # 按字段划分分类结果[[w,wh,p],[],[]]
    condition_category_threshold_list = [condition1_res, condition2_res, condition3_res]
    all_plt.category_plot(split_wind_df, '风速', gmm_plot_picture_path, data_list_reset)
    all_plt.category_plot(split_wheel_df, '轮毂转速', gmm_plot_picture_path, data_list_reset)
    all_plt.category_plot(split_power_df, '发电机侧功率', gmm_plot_picture_path, data_list_reset)
    return condition_category_threshold_list


def dataset_split(lof_df, condition_category_threshold_list):
    """按工况划分结果阈值划分子工况

    Parameters
    ----------
    lof_df : DataFrame
        去噪后数据集

    condition_category_threshold_list : list
        按字段划分结果阈值

    Returns
    -------
    condition_df_list : list
        子工况数据集合列表

    """
    condition1_df = model.condition_split(
        lof_df, condition_category_threshold_list[1][0], condition_category_threshold_list[0][0],
        condition_category_threshold_list[2][0], condition_category_threshold_list[1][1],
        condition_category_threshold_list[0][1], condition_category_threshold_list[2][1])
    condition2_df = model.condition_split(
        lof_df, condition_category_threshold_list[1][1], condition_category_threshold_list[0][1],
        condition_category_threshold_list[2][1], condition_category_threshold_list[1][2],
        condition_category_threshold_list[0][2], condition_category_threshold_list[2][2])
    condition3_df = model.condition_split(
        lof_df, condition_category_threshold_list[1][2], condition_category_threshold_list[0][2],
        condition_category_threshold_list[2][2])
    condition_df_list = [condition1_df, condition2_df, condition3_df]
    return condition_df_list


def model_construct(condition_df, frequency, condition_krr_pre_result_plot_save_path, wind_turbine, num):
    """划分工况结果

    Parameters
    ----------
    condition_df : DataFrame
        子工况数据

    frequency : str
        采样频率

    condition_krr_pre_result_plot_save_path : Path
        分工况建模预测结果折线图存储路径

    wind_turbine : str
        风机编号

    num : str
        工况编号

    Returns
    -------
    model_construct_result_list : list
        分工况预测建模结果集合列表

    """
    model_construct_result_list = model.condition_model_construct(condition_df, frequency)
    all_plt.condition_krr_pre_result_plot(model_construct_result_list[0], model_construct_result_list[1],
                                          wind_turbine, num, condition_krr_pre_result_plot_save_path)
    return model_construct_result_list


def main():
    base_path = Path("/三十六湾2020S3")
    data_list = get_data_name_list(base_path)
    # 生成存储路径
    gmm_scatter_picture_path = join_save_path("gmm_scatter", base_path)
    condition_result_path = join_save_path("condition_result", base_path)
    gmm_plot_picture_path = join_save_path("gmm_result_plot", base_path)
    condition_krr_pre_result_plot_path = join_save_path("krr_pre_res_plot", base_path)

    for wind_turbine in data_list:

        original_df = pd.read_csv(str(base_path.joinpath(wind_turbine, ".csv")), encoding="utf-8")
        lof_df = data_pre_processing(original_df)
        if lof_df.empty:
            print(wind_turbine + "is null after filter")
            continue

        handled_df = condition.gmm_raw_data_handing(lof_df, '50s')

        # 高斯混合模型分类工况
        labels = condition.gaussian_mixture_model(handled_df)

        # 绘制工况分类散点图，并保存
        all_plt.gmm_scatter(handled_df, labels, wind_turbine, gmm_scatter_picture_path)

        # 保存分类结果并存入csv
        result_df = condition.condition_category_result(handled_df, labels, wind_turbine)
        condition_category_final_result = pd.DataFrame()
        condition_category_final_result = pd.concat([condition_category_final_result, result_df], axis=1)
        pd.to_csv(condition_category_final_result, str(condition_result_path.joinpath("result.csv")))  # 目录名

        del original_df, handled_df, result_df
        gc.collect()

    # 读取分类结果文件并处理分类结果
    data = pd.read_csv("category_result_dataset_path", encoding="utf-8")

    # 排序，转置，划分数据矩阵
    sorted_df = data.apply(lambda col: col.sort_values(ignore_index=True), axis=0)

    # nums可能有缺失，手动调整
    data_list_reset = data_list

    # 按工况分类结果提取，并绘制所有数据集的工况分类结果
    condition_category_threshold_list \
        = condition_category_result_split(sorted_df, data_list_reset, gmm_plot_picture_path)

    # 分工况建模
    for wind_turbine in data_list_reset:
        original_df = pd.read_csv(str(base_path.joinpath(wind_turbine, ".csv")), encoding="utf-8")
        lof_df = data_pre_processing(original_df)
        if lof_df.empty:
            print(wind_turbine + "is null after filter")
            continue
        condition_df_list = dataset_split(lof_df, condition_category_threshold_list)
        sub_condition_model_construct_result_list = []
        for index, condition_df in enumerate(condition_df_list):
            if condition_df.empty:
                print(wind_turbine + "工况"+str(index+1)+"is null after filter")
            else:
                model_construct_result_list \
                    = model_construct(condition_df, "50s", condition_krr_pre_result_plot_path, wind_turbine, str(index))
                sub_condition_model_construct_result_list.append(model_construct_result_list)

        # 计算工况2中超过工况3温度均值的数据个数占工况2总数的占比
        percentage = all_condition.condition_over_percentage(condition_df_list[1], condition_df_list[2])

        # 提取残差特征
        if sub_condition_model_construct_result_list[-1]:
            diff_feature_df = \
                detection.diff_feature_generation(sub_condition_model_construct_result_list[-1][0],
                                                  sub_condition_model_construct_result_list[-1][1], "1D")
            # 孤立森林异常检测，得到异常标签值
            abnormal_pred_labels = detection.isolation_forest(diff_feature_df, state=1)


if __name__ == '__main__':
    main()
