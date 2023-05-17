# -*- coding: utf-8 -*-
"""
@function: GB pipeline
@version: python 3.9
"""
import time
import re
import numpy as np
import pandas as pd
import os
import json
import pickle
from scipy.stats import *
import seaborn as sns
import palettable
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.utils import check_array
# from appStore.data.manage.pyopls.opls import OPLS
from PIL import Image
from pyopls import OPLS
from statsmodels.formula.api import ols
from scipy.stats import hypergeom
from scipy.cluster import hierarchy
import requests
from lxml import html
from lxml import etree
# from bs4 import BeautifulSoup
import statistics
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import squarify
from sklearn.metrics import roc_curve, auc
# matplotlib.rc("font", family='YouYuan') # 饼图所需

class Public(object):
    def __init__(self, descNum, sample, file_name, path, filterMethod, fillMethod, normMethod, group):
        self.descNum = descNum
        self.sample = sample
        self.group = group
        self.input_file = os.path.join(path, file_name)
        self.path = path
        self.filterMethod = filterMethod
        self.fillMethod = fillMethod
        self.normMethod = normMethod
        self.raw_data = self.read_raw_file()
        self.QCnum = self.cal_qc_num()

    def df_to_json(self, df, orient='split'):
        """
        将读入的excel表数据转换成json格式的数据
        返回json格式数据
        """
        return df.to_json(orient=orient, force_ascii=False)  # pd.to_json用于将python对象转换成json字符串

    def read_raw_file(self):
        """
        读取原始数据表，根据NAME_EN列删除重复项，默认保留重复项第一个，并将表里面的0值转换成NaN
        返回json格式数据和读入的dataframe原始数据
        """
        RawFileExcel = pd.read_excel(self.input_file)
        # 将NAME_EN所有字母变为小写后去重
        RawFileExcel['name'] = RawFileExcel['NAME_EN'].str.lower()
        RawFileExcel = RawFileExcel.drop_duplicates(subset='name')
        RawFileExcel = RawFileExcel.drop('name', axis=1)
        RawFileExcel = RawFileExcel.reset_index(drop=True)
        # 将json格式数据解码成python对象
        RawFileJson = json.loads(self.df_to_json(RawFileExcel))
        RawFileExcel_exp = RawFileExcel.iloc[:, :self.descNum]
        RawFileExcel_data = RawFileExcel.iloc[:, self.descNum:]
        RawFileExcel_data[RawFileExcel_data == 0] = np.nan
        RawFileExcel = pd.concat([RawFileExcel_exp, RawFileExcel_data], axis=1)
        return RawFileJson, RawFileExcel

    def cal_qc_num(self):
        """
        计算QC的个数
        返回QC个数的number
        """
        RawFile = self.raw_data[1]
        QCnum = str(RawFile.columns.str.startswith('QC')).count("True")
        return QCnum

    def sfw_filter(self):
        """
        对原始数据进行四分位过滤，过滤可能的异常数据
        返回过滤后的dataframe数据
        """
        RawFile = self.raw_data[1]
        QCnum = str(RawFile.columns.str.startswith('QC')).count("True")  # QC的数量，区分大小写
        ISnum = [
            RawFile.loc[RawFile['NAME_EN'] == '2-Chloro-DL-Phenylalanine'].empty
        ].count(False)  # 只有一个内标
        # 过滤时内标行不参与，需要拆分内标行和非内标行，过滤后再合并
        if ISnum == 1:
            lineIs = int(RawFile.loc[RawFile['NAME_EN'] == '2-Chloro-DL-Phenylalanine'].index[0])
            lineIsData = RawFile.loc[[lineIs], ]  # 内标行数据
            dropIsData = RawFile.drop([lineIs])  # 非内标行数据
            dropIsData = dropIsData.reset_index(drop=True)
        else:
            lineIsData = pd.DataFrame()
            dropIsData = RawFile
        if QCnum > 0:
            df_exp_QC = dropIsData.loc[:, dropIsData.columns.str.startswith("QC")]  # 提取QC样本数据，四分位法QC不参与过滤
            df_exp_drop_QC = dropIsData.drop(list(df_exp_QC.columns), axis=1)  # 获取除QC样本外的其他样本的数据
            df_exp_desc = dropIsData.loc[:, df_exp_drop_QC.columns[0:self.descNum]]  # 获取前8列非样本含量信息
            df_exp_in = df_exp_drop_QC.drop(list(df_exp_desc.columns), axis=1)  # 获取样本含量信息
        else:
            df_exp_QC = pd.DataFrame()
            df_exp_desc = dropIsData.loc[:, dropIsData.columns[0:self.descNum]]
            df_exp_in = dropIsData.drop(list(df_exp_desc.columns), axis=1)

        """数据过滤"""
        # 此处为四分位过滤，四分位检验法是认为大于line_max和小于line_min的值为异常值，具体计算公式百度
        names = df_exp_in.columns
        quan = df_exp_in.quantile([.25, .75], numeric_only=True, axis=1).T
        line_max = quan[0.75] + 1.5 * (quan[0.75] - quan[0.25])
        line_max[pd.isnull(line_max)] = 0
        line_min = quan[0.25] - 1.5 * (quan[0.75] - quan[0.25])
        line_min[pd.isnull(line_min)] = 0
        line_max = list(line_max)
        line_min = list(line_min)

        df_exp_in_arr = np.array(df_exp_in)
        df_exp_in_arr = df_exp_in_arr.flatten()
        j = 1
        length = len(df_exp_in.columns)
        for i in range(len(df_exp_in_arr)):
            if i < j * length:
                j = j
            else:
                j = j + 1
            if not (np.isnan(df_exp_in_arr[i])) and (
                    df_exp_in_arr[i] > line_max[j - 1] or df_exp_in_arr[i] < line_min[j - 1]):
                df_exp_in_arr[i] = None
        df_exp_in = df_exp_in_arr.reshape(len(df_exp_in.index), len(df_exp_in.columns))
        df_exp_in = pd.DataFrame(df_exp_in)
        df_exp_in.columns = names

        df_exp_in_Excel1 = pd.concat([df_exp_desc, df_exp_in, df_exp_QC], axis=1)  # 按列合并
        lineIsData_QC = lineIsData.loc[:, lineIsData.columns.str.startswith("QC")]
        lineIsData_drop_QC = lineIsData.drop(list(lineIsData_QC.columns), axis=1)
        lineIsData = pd.concat([lineIsData_drop_QC, lineIsData_QC], axis=1)
        df_exp_in_Excel2 = pd.concat([df_exp_in_Excel1, lineIsData], axis=0)  # 按行合并
        df_exp_in_Excel2 = df_exp_in_Excel2.reset_index(drop=True)
        df_exp_in_Excel2.to_excel(r"C:\Users\Administrator\Desktop\GB_cloud\sfw.xlsx")

        rawDataByFilter_Excel_temp = df_exp_in_Excel2
        rawDataByFilter_Json_temp = json.loads(self.df_to_json(rawDataByFilter_Excel_temp))
        return rawDataByFilter_Json_temp, rawDataByFilter_Excel_temp  # 返回经过四分位过滤后的数据

    def rsd_filter(self):
        """
        根据rsd方法对原始数据进行过滤
        返回rsd法过滤后的dataframe数据
        """
        RawFile = self.raw_data[1]
        QCnum = str(RawFile.columns.str.startswith('QC')).count("True")  # QC的数量，区分大小写
        ISnum = [
            RawFile.loc[RawFile['NAME_EN'] == '2-Chloro-DL-Phenylalanine'].empty
        ].count(False)
        # 过滤时内标行不参与，需要拆分内标行和非内标行，过滤后再合并
        if ISnum == 1:
            lineIs = int(RawFile.loc[RawFile['NAME_EN'] == '2-Chloro-DL-Phenylalanine'].index[0])  # 内标行索引
            lineIsData = RawFile.loc[[lineIs],]  # 内标行数据
            dropIsData = RawFile.drop([lineIs])  # 非内标行数据
            dropIsData = dropIsData.reset_index(drop=True)
        else:
            lineIsData = pd.DataFrame()
            dropIsData = RawFile
        if QCnum > 0:
            df_exp_QC = dropIsData.loc[:, dropIsData.columns.str.startswith("QC")]
            df_exp_drop_QC = dropIsData.drop(list(df_exp_QC.columns), axis=1)
            df_exp_desc = dropIsData.loc[:, df_exp_drop_QC.columns[0:self.descNum]]
            df_exp_in = df_exp_drop_QC.drop(list(df_exp_desc.columns), axis=1)
        else:
            df_exp_QC = pd.DataFrame()
            df_exp_desc = dropIsData.loc[:, dropIsData.columns[0:self.descNum]]
            df_exp_in = dropIsData.drop(list(df_exp_desc.columns), axis=1)

        # rsd法qc不能为NA
        df_exp_QC_mean = df_exp_QC.apply(lambda x: np.mean(x), axis=1)
        df_exp_QC_sd = df_exp_QC.apply(lambda x: np.std(x, ddof=1), axis=1)
        # 计算该物质的QC组rsd
        df_exp_QC_rsd = df_exp_QC_sd / df_exp_QC_mean
        df_exp_QC_rsd_filter = df_exp_QC_rsd[df_exp_QC_rsd < 1]
        df_exp_QC_rsd_filter_index = list(df_exp_QC_rsd_filter.index)

        df_exp_in_filter = df_exp_in.loc[df_exp_QC_rsd_filter_index, ]
        df_exp_QC_filter = df_exp_QC.loc[df_exp_QC_rsd_filter_index, ]
        df_exp_desc_filter = df_exp_desc.loc[df_exp_QC_rsd_filter_index, ]
        df_exp_in_Excel1 = pd.concat([df_exp_desc_filter, df_exp_in_filter, df_exp_QC_filter], axis=1)  ##按列合并
        lineIsData_QC = lineIsData.loc[:, lineIsData.columns.str.startswith("QC")]
        lineIsData_drop_QC = lineIsData.drop(list(lineIsData_QC.columns), axis=1)
        lineIsData = pd.concat([lineIsData_drop_QC,lineIsData_QC], axis=1)
        df_exp_in_Excel2 = pd.concat([df_exp_in_Excel1, lineIsData], axis=0)  ##按行合并
        df_exp_in_Excel2 = df_exp_in_Excel2.reset_index(drop=True)

        rawDataByFilter_Excel_temp = df_exp_in_Excel2
        rawDataByFilter_Json_temp = json.loads(self.df_to_json(rawDataByFilter_Excel_temp))
        return rawDataByFilter_Json_temp, rawDataByFilter_Excel_temp

    def no_filter(self):
        """
        不对数据进行过滤处理，只改变数据的格式，样本顺序等
        返回Reshape后的dataframe
        """
        RawFile = self.raw_data[1]
        QCnum = str(RawFile.columns.str.startswith('QC')).count("True")  ## QC的数量，区分大小写
        ISnum = [
            RawFile.loc[RawFile['NAME_EN'] == '2-Chloro-DL-Phenylalanine'].empty
        ].count(False)
        if ISnum == 1:
            lineIs = int(RawFile.loc[RawFile['NAME_EN'] == '2-Chloro-DL-Phenylalanine'].index[0])  ##内标行索引
            lineIsData = RawFile.loc[[lineIs], ]  # 内标行数据
            dropIsData = RawFile.drop([lineIs])  # 非内标行数据
        else:
            lineIsData = pd.DataFrame()
            dropIsData = RawFile
        if QCnum > 0:
            df_exp_QC = dropIsData.loc[:, dropIsData.columns.str.startswith("QC")]
            df_exp_drop_QC = dropIsData.drop(list(df_exp_QC.columns), axis=1)
            df_exp_desc = dropIsData.loc[:, df_exp_drop_QC.columns[0:self.descNum]]
            df_exp_in = df_exp_drop_QC.drop(list(df_exp_desc.columns), axis=1)
        else:
            df_exp_QC = pd.DataFrame()
            df_exp_desc = dropIsData.loc[:, dropIsData.columns[0:self.descNum]]
            df_exp_in = dropIsData.drop(list(df_exp_desc.columns), axis=1)

        df_exp_in_Excel1 = pd.concat([df_exp_desc, df_exp_in, df_exp_QC], axis=1)  ##按列合并
        lineIsData_QC = lineIsData.loc[:, lineIsData.columns.str.startswith("QC")]
        lineIsData_drop_QC = lineIsData.drop(list(lineIsData_QC.columns), axis=1)
        lineIsData = pd.concat([lineIsData_drop_QC, lineIsData_QC], axis=1)
        df_exp_in_Excel2 = pd.concat([df_exp_in_Excel1, lineIsData], axis=0)  ##按行合并
        df_exp_in_Excel2 = df_exp_in_Excel2.reset_index(drop=True)

        rawDataByFilter_Excel_temp = df_exp_in_Excel2
        rawDataByFilter_Json_temp = json.loads(self.df_to_json(rawDataByFilter_Excel_temp))
        return rawDataByFilter_Json_temp, rawDataByFilter_Excel_temp

    def recode_na(self, filterData):
        """
        对过滤后的dataframe格式数据的空值进行补值，
        空值行的保留标准：每行QC样本空值个数要小于一半，否则删除改行数据，满足该条件后，若每组样本空值个数都超过一半，则删除该行
        补值方法有最小值法、中位值法或不补值
        返回补值后的dataframe
        """
        df_desc = filterData.loc[:, filterData.columns[0:self.descNum]]
        df_input = filterData.drop(list(filterData.columns[0:self.descNum]), axis=1)
        sub_na_index_list = []
        col = 0
        for i in self.sample.split(" "):
            col = col + int(i)
            df_input_subgroup = df_input.loc[:, df_input.columns[(col - int(i)):col]]
            subgroup_na = [x for x in df_input_subgroup.isna().sum(axis=1) <= df_input_subgroup.shape[1] / 2]
            sub_na_index_list.append([i for i, n in (enumerate(subgroup_na)) if n == True])
        total_na_index_list = []
        for i in range(len(sub_na_index_list)):
            total_na_index_list = total_na_index_list + sub_na_index_list[i]
        total_na_index_list = set(total_na_index_list)
        if self.QCnum > 0:
            df_qc = df_input.iloc[:, sum([int(x) for x in self.sample.split(" ")]):]
            qc_na = [x for x in df_qc.isna().sum(axis=1) <= df_qc.shape[1] / 2]
            qc_na_index_list = [i for i, n in (enumerate(qc_na)) if n == True]
            na_index_list = list(total_na_index_list.intersection(set(qc_na_index_list)))
        else:
            na_index_list = list(total_na_index_list)
        if len(na_index_list) > 0:
            df_desc = df_desc.iloc[na_index_list, :]
            df_input = df_input.iloc[na_index_list, :]
        df_desc = df_desc.reset_index(drop=True)
        df_input = df_input.reset_index(drop=True)

        df_input_temp = df_input
        if self.fillMethod == "median":
            medianSelect = df_input.median()
            for i in range(len(medianSelect)):
                df_input_temp.loc[:, df_input_temp.columns[i]] = df_input_temp.loc[:, df_input_temp.columns[i]].fillna(
                    medianSelect[i])
        if self.fillMethod == "min":
            df_data11 = df_input.iloc[ :, :(df_input.shape[1] - self.QCnum)]
            minSelect = df_data11.min().min() / 2
            df_input_temp = df_input_temp.fillna(minSelect)
        if self.fillMethod == "none":
            df_input_temp = df_input
        df_input_temp_Excel2 = pd.concat([df_desc, df_input_temp], axis=1)  ##按列合并
        df_input_temp_Excel2.to_excel(self.path + "/recode_na.xlsx", index=False)
        return df_input_temp_Excel2

    def normalize(self):
        """
        对补值后的数据进行归一化处理
        归一化方式有：内标归一法、面积归一法、中位值归一法及不进行归一处理
        返回归一化处理后的dataframe,也即mean表里的数据
        """
        RawFile = self.raw_data[1]
        if self.filterMethod == "sfw":
            filterdata = self.sfw_filter()[1]
        elif self.filterMethod == "rsd":
            filterdata = self.rsd_filter()[1]
        else:
            filterdata = self.no_filter()[1]
        filterData = self.recode_na(filterData=filterdata)  # 获取过滤、补值后的数据进行后续分析
        ISnum = [
            RawFile.loc[RawFile['NAME_EN'] == '2-Chloro-DL-Phenylalanine'].empty
        ].count(False)

        if ISnum == 1:
            IS_line = int(filterData.loc[filterData["NAME_EN"] == "2-Chloro-DL-Phenylalanine"].index[0])
            target_is_name = filterData["NAME_EN"][IS_line]
        df_desc = filterData.loc[:, filterData.columns[0:self.descNum]]
        df_input = filterData.drop(list(filterData.columns[0:self.descNum]), axis=1)
        # 内标归一化
        if self.normMethod == "neibiao":
            is_line_num = int(df_desc[df_desc["NAME_EN"] == target_is_name].index[0])
            # print(is_line_num)
            for i in range(len(df_input.columns)):
                df_input.iloc[:, i] = df_input.iloc[:, i] / df_input.iloc[is_line_num, i]
            norm_data = pd.concat([df_desc, df_input], axis=1)
            norm_data = norm_data.drop(IS_line, axis=0)
            norm_data = norm_data.reset_index(drop=True)
        # 面积归一化
        elif self.normMethod == "mianji":
            if ISnum > 0:
                # is_line_name = list(df.loc[df["compound name"].str.startswith("IS")]["compound name"])
                # is_line_num = list(df_desc.loc[df_desc["MS2 name"].str.startswith("IS")].index)
                df_desc = df_desc.drop(IS_line, axis=0)
                df_desc = df_desc.reset_index(drop=True)
                df_input = df_input.drop(IS_line, axis=0)
                df_input = df_input.reset_index(drop=True)
                for i in range(len(df_input.columns)):
                    df_input.iloc[:, i] = df_input.iloc[:, i] / np.sum(df_input.iloc[:, i])
            else:
                for i in range(len(df_input.columns)):
                    df_input.iloc[:, i] = df_input.iloc[:, i] / np.sum(df_input.iloc[:, i])
            norm_data = pd.concat([df_desc, df_input], axis=1)
        # 中位值归一化
        elif self.normMethod == "median":
            if ISnum > 0:
                df_desc = df_desc.drop(IS_line, axis=0)
                df_desc = df_desc.reset_index(drop=True)
                df_input = df_input.drop(IS_line, axis=0)
                df_input = df_input.reset_index(drop=True)
                for i in range(len(df_input.columns)):
                    df_input.iloc[:, i] = df_input.iloc[:, i] / df_input.iloc[:, i].median()
            else:
                for i in range(len(df_input.columns)):
                    df_input.iloc[:, i] = df_input.iloc[:, i] / df_input.iloc[:, i].median()
            norm_data = pd.concat([df_desc, df_input], axis=1)
        # 不使用归一化方法
        elif self.normMethod == "none":
            if ISnum > 0:
                df_desc = df_desc.drop(IS_line, axis=0)
                df_desc = df_desc.reset_index(drop=True)
                df_input = df_input.drop(IS_line, axis=0)
                df_input = df_input.reset_index(drop=True)
                norm_data = pd.concat([df_desc, df_input], axis=1)
            else:
                norm_data = pd.concat([df_desc, df_input], axis=1)
        norm_data_desc = norm_data.iloc[:, :self.descNum]
        norm_data_input = norm_data.drop(norm_data_desc.columns, axis=1)
        sample_num = self.sample.split()
        sample_num = [int(x) for x in sample_num]
        sample_num = np.cumsum(sample_num)
        group = self.group.split()
        norm_data_input0 = norm_data_input.iloc[:, :sample_num[0]]
        norm_data_input0.insert(norm_data_input0.shape[1], "Mean "+group[0], norm_data_input0.mean(axis=1))
        norm_data_res = norm_data_input0
        for i in range(1,len(sample_num)):
            norm_data_tmp = norm_data_input.iloc[:, sample_num[i-1]:sample_num[i]]
            norm_data_tmp.insert(norm_data_tmp.shape[1], "Mean "+group[i], norm_data_tmp.mean(axis=1))
            norm_data_res = pd.concat([norm_data_res, norm_data_tmp], axis=1)
        if self.QCnum > 0:
            qc_data = norm_data_input.iloc[:, sample_num[-1]:]
            qc_data.insert(qc_data.shape[1], "Mean QC", qc_data.mean(axis=1))
            norm_data_res = pd.concat([norm_data_res, qc_data], axis=1)
        norm_data = pd.concat([norm_data_desc, norm_data_res], axis=1) #用来输出成mean表
        norm_data.to_excel(self.path + "/norm.xlsx", index=False)
        norm_data.replace(np.nan, 0)
        return norm_data

class ModelAnalysis(object):
    def __init__(self, path, species, descNum, sample, group, contrast, pvalue, FC_up, FC_down, mean_data, vip_cut, ci, kegghitmin, pca_scaling, opls_scaling, pca_log, opls_log):
        self.path = path
        self.species = species
        self.descNum = descNum
        self.sample = sample
        self.group = group
        self.pvalue = pvalue
        self.FC_up = FC_up
        self.FC_down = FC_down
        self.contrast = contrast
        self.mean_data = mean_data
        self.vip_cut = vip_cut
        self.ci = ci
        self.kegghitmin = kegghitmin
        self.pca_scaling = pca_scaling
        self.opls_scaling = opls_scaling
        self.pca_log = pca_log
        self.opls_log = opls_log
        self.vip = self.get_vip()
        self.difference = self.compare_filter_dec()
        self.kegg_pathway_mapping_list = self.mapping()
        self.kegg_results_list = None
        self.pca_model_param = None
        self.oplsda_model_param = None
        self.diff_com = None
        self.pathway_results_list = None

    def compare_analysis(self):
        """
        差异物质筛选，通过计算每个对比组之间的p值及FC值，根据p值和FC值筛选出差异物质
        返回差异物质表
        """
        filterData = self.mean_data
        filterData_name = filterData.columns[filterData.columns.str.startswith("Mean ")]
        filterData = filterData.drop(filterData_name, axis=1)
        df_desc = filterData.loc[:, filterData.columns[0:self.descNum]]
        df_input = filterData.drop(list(filterData.columns[0:self.descNum]), axis=1)
        col = 0
        Gseq = -1
        # 修改列名,把样本名(即列名)前面加上所属组名
        for i in self.sample.split(" "):
            Gseq = Gseq + 1
            col = col + int(i)
            df_input_subGroup = df_input.loc[:, df_input.columns[(col - int(i)):col]]
            for n in range(len(df_input_subGroup.columns)):  # 把每组的样本名前加上所属的组的信息，方便下面对比时进行索引
                new_name = self.group.split(" ")[Gseq] + "_" + df_input_subGroup.columns[n]
                df_input = df_input.rename(columns={df_input_subGroup.columns[n]: new_name})
        # 计算各对比的FC值和P值，并筛选出差异物质
        df_com_res = list()  # 用于保存各对比的处理后的数据
        for i in self.contrast.split(" "):
            g1 = i.split(":")[0]
            g2 = i.split(":")[1]
            df_compare1 = df_input.loc[:, df_input.columns.str.startswith(g1)]
            df_compare2 = df_input.loc[:, df_input.columns.str.startswith(g2)]
            df_compare1 = df_compare1.astype('float64')
            df_compare2 = df_compare2.astype('float64')
            FC = pd.DataFrame(df_compare1.mean(axis=1) / df_compare2.mean(axis=1)).rename(
                columns={0: "FC"})  # 计算各对比中每个物质的FC
            logFC = pd.DataFrame(np.log2(FC)).rename(columns={"FC": "logFC"})  # 计算各对比中每个物质的logFC
            len_a = int(len(df_compare1.columns))
            len_b = int(len(df_compare2.columns))
            F = np.var(df_compare1, axis=1, ddof=1) / np.var(df_compare2, axis=1, ddof=1)
            f_test_p = 1 - 2 * abs(0.5 - f.cdf(F, len_a-1, len_b-1))
            var_test = pd.DataFrame(f_test_p).rename(columns={0: "var_test"})
            df_compare = pd.concat([df_compare1,df_compare2,var_test], axis=1)
            df_compare["var_test"] = df_compare["var_test"] > 0.05
            P = df_compare.apply(
                lambda x: ttest_ind(x.values.tolist()[:len_a], x.values.tolist()[len_a:len_a+len_b], equal_var=x.values.tolist()[-1:][0])[
                    1], axis=1)
            P = pd.DataFrame(P).rename(columns={0:"Pvalue"})
            df_compare1.insert(df_compare1.shape[1], "Mean "+g1, df_compare1.mean(axis=1))
            df_compare2.insert(df_compare2.shape[1], "Mean "+g2, df_compare2.mean(axis=1))
            df_com = pd.concat([df_desc, df_compare1, df_compare2, FC, logFC, P], axis=1)  # 按列合并
            df_com_res.append(df_com)
        return df_com_res

    def compare_filter_dec(self):
        """
        compare_analysis所得的差异物质表加上VIP值并整理后续所需的差异物质表格信息形式
        返回所有peak后的数据并和对应vip合并后的结果，每个对比的差异物质表，差异物质表路径
        """
        df_com = self.compare_analysis()
        vip_tmp = self.vip
        df_com_analysis = []  # 保存所有peak后的数据并和对应vip合并后的结果
        df_com_dec = []  # 用于保存每个对比的差异物质表
        dec_path_list = []  # 用于保存差异物质表路径
        self.diff_com = []
        for i in range(len(self.contrast.split(" "))):
            g1 = self.contrast.split(" ")[i].split(":")[0]
            g2 = self.contrast.split(" ")[i].split(":")[1]
            df_com_analysis_tmp = pd.concat([df_com[i], vip_tmp[i]], axis=1)
            df_com_analysis.append(df_com_analysis_tmp)
            df_com_dec_tmp = df_com_analysis_tmp[df_com_analysis_tmp['Pvalue'] < self.pvalue]  # 保留Pvalue小于pvalue的行
            df_com_dec_tmp = df_com_dec_tmp[df_com_dec_tmp['vip'] > self.vip_cut]
            df_peak_dec_tmp = df_com_dec_tmp
            # df_peak_dec_tmp1 = df_peak_dec_tmp.iloc[:, 9:-4]
            df_com_dec_tmp = df_com_dec_tmp[df_com_dec_tmp["NAME_EN"].notnull()]
            df_com_dec_tmp2 = df_com_dec_tmp[["id", "NAME_EN", "NAME_CH", "CAS", "KEGG_ID", "FORMULA", "EXACT_MASS", "CLASS_CH", "q1", "q3",
                                              "ionmode", "rt", "Mean "+g1, "Mean "+g2, "FC", "logFC", "Pvalue", "vip"]]
            if df_com_dec_tmp2.shape[0] > 7:
                df_com_dec_tmp2 = df_com_dec_tmp2.iloc[:7, :]
            self.diff_com.append(df_com_dec_tmp2)
            df_com_dec_tmp_name = df_com_dec_tmp.columns[df_com_dec_tmp.columns.str.startswith("Mean ")]
            df_com_dec_tmp1 = df_com_dec_tmp.drop(df_com_dec_tmp_name, axis=1)
            df_com_dec.append(df_com_dec_tmp1)
            dec_path = self.path + g1 + "-" + g2 + "/DEC.xlsx"
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
                df_com_dec_tmp.to_excel(self.path + g1 + "-" + g2 + "/stats analysis.xlsx", index=False)  # 保存各对比中所有行的统计信息
            df_peak_dec_tmp.to_excel(dec_path, index=False)
            dec_path_list.append(g1 + "-" + g2 + "/DEC.xlsx")
        return df_com_analysis, df_com_dec, dec_path_list

    def total_pca(self):
        """
        使用mean表数据绘制所有组的PCA图，首先要建立PCA模型，再根据每个样本对第一、二主成分的贡献度绘制PCA散点图
        """
        filterData = self.mean_data
        filterData_name = filterData.columns[filterData.columns.str.startswith("Mean ")]
        filterData = filterData.drop(filterData_name, axis=1)
        path_list = []
        df_desc = filterData.loc[:, filterData.columns[0:self.descNum]]
        df_input = filterData.drop(list(filterData.columns[0:self.descNum]), axis=1)
        col = 0
        Gseq = -1
        df_com = pd.DataFrame()
        model_param = []
        # 修改列名,把样本名(即列名)前面加上所属组名
        for i in self.sample.split(" "):
            Gseq = Gseq + 1
            col = col + int(i)
            df_input_subGroup = df_input.loc[:, df_input.columns[(col - int(i)):col]]
            for n in range(len(df_input_subGroup.columns)):  # 把每组的样本名前加上所属的组的信息，方便下面对比时进行索引
                new_name = self.group.split(" ")[Gseq] + "_" + df_input_subGroup.columns[n]
                df_input = df_input.rename(columns={df_input_subGroup.columns[n]: new_name})
            df_com = pd.concat([df_com, df_input.loc[:, df_input.columns.str.startswith(self.group.split(" ")[Gseq])]], axis=1)
        df_com = pd.concat([df_com, df_input.loc[:,df_input.columns.str.startswith("QC")]], axis=1)
        QC_num = df_input.loc[:,df_input.columns.str.startswith("QC")].shape[1]
        X = df_com.T
        sample_len = X.shape[0]
        X = X.astype("float")
        if self.pca_log == True:
            X.replace(0, np.nan)
            X = np.log10(X)  # log对数转换
        if self.pca_scaling == "Ctr":
            X = X.sub(X.mean(axis=0), axis=1)  # scaling="center"
        elif self.pca_scaling == "UV":
            X = X.sub(X.mean(axis=0), axis=1) / X.std(axis=0)  # scaling="UV"
        else:
            X = X.sub(X.mean(axis=0), axis=1) / np.sqrt(X.std(axis=0))  # scaling="Par"
        X_index = X.index
        X[X.isnull()] = 0
        sample_num = self.sample + " " + str(QC_num)
        group_num = [int(i) for i in sample_num.split(" ")]
        y = np.repeat(range(len(sample_num.split(" "))), group_num)
        target_names = [i for i in self.group.split(" ")] + ["QC"]
        colors = ["navy", "turquoise", "red", "blue","green","yellow"]

        pca = PCA(min(10, min(X.shape)))  # n_components=2
        pca.fit(X)
        summary_df_res = pca.transform1(X)
        summary_df = summary_df_res[1]
        if summary_df.iloc[0, 1] < 3:
            pca = PCA(n_components=3)  # n_components=2
            pca.fit(X)
            summary_df_res = pca.transform1(X)
            summary_df = summary_df_res[1]
        model_param = model_param + ["Model1", "PCA"]
        model_param.append(summary_df.iloc[0, 1])
        model_param.append(sample_len)
        model_param.append(summary_df.iloc[0, 0])
        model_param.append("TOTAL")
        model_param = pd.DataFrame(model_param).T
        model_param.columns = ["Model", "Type", "A", "N", "R2X(cum)", "Title"]
        self.pca_model_param = model_param
        X_r = summary_df_res[0]
        X_score = X_r[:, [0, 1]]
        n = X_score.shape[0]
        hfn = 2 * (n - 1) * (n ** 2 - 1) / (n ** 2 * (n - 2)) * f.ppf(self.ci, 2, n - 2)
        rv = np.linspace(0, 2 * np.pi, 100)
        x1 = np.sqrt(np.var(X_r[:, 0], ddof=1) * hfn) * np.cos(rv)
        y1 = np.sqrt(np.var(X_r[:, 1], ddof=1) * hfn) * np.sin(rv)
        pc1_ratio = round(pca.explained_variance_ratio_[0] * 100, 1)
        pc2_ratio = round(pca.explained_variance_ratio_[1] * 100, 1)
        plt.figure()
        lw = 2
        for color, i, target_name in zip(colors, list(range(len(group_num))), target_names):
            plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
        plt.plot(x1, y1, c="k")
        plt.axhline(y=0, color="grey")
        plt.axvline(x=0, color="grey")  # linestyle="--"
        plt.xlabel('PC1[' + str(pc1_ratio) + "%]")
        plt.ylabel('PC2[' + str(pc2_ratio) + "%]")
        plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5), borderaxespad = 0., shadow=False, scatterpoints=1)
        if not os.path.exists(self.path + "total_pca"):
            os.makedirs(self.path + "total_pca")
        plt.savefig(self.path + "total_pca" + "/PCA.png", dpi=500, bbox_inches="tight")
        plt.close()

    def pca_model(self):
        """
        使用mean表数据绘制每组的PCA图，首先要建立PCA模型，再根据每个样本对第一、二主成分的贡献度绘制PCA散点图
        返回PCA散点图等
        """
        # August 4th pca作图的时候是用所有的数据，用筛选出的差异物质是不对的
        filterData = self.mean_data
        filterData_name = filterData.columns[filterData.columns.str.startswith("Mean ")]
        filterData = filterData.drop(filterData_name, axis=1)
        path_list = []
        df_desc = filterData.loc[:, filterData.columns[0:self.descNum]]
        df_input = filterData.drop(list(filterData.columns[0:self.descNum]), axis=1)
        col = 0
        Gseq = -1
        # 修改列名,把样本名(即列名)前面加上所属组名
        for i in self.sample.split(" "):
            Gseq = Gseq + 1
            col = col + int(i)
            df_input_subGroup = df_input.loc[:, df_input.columns[(col - int(i)):col]]
            for n in range(len(df_input_subGroup.columns)):  # 把每组的样本名前加上所属的组的信息，方便下面对比时进行索引
                new_name = self.group.split(" ")[Gseq] + "_" + df_input_subGroup.columns[n]
                df_input = df_input.rename(columns={df_input_subGroup.columns[n]: new_name})
        # 计算各对比的FC值和P值，并筛选出差异物质
        # df_com_res = list()  # 用于保存各对比的处理后的数据
        # df_com_dec_res = list()  # 用于保存处理后差异物质的数据
        # self.pca_model_param = pd.DataFrame()
        k = 2
        for i in self.contrast.split(" "):
            model = []
            model.append("model" + str(k))
            model.append("PCA")
            g1 = i.split(":")[0]
            g2 = i.split(":")[1]
            df_compare1 = df_input.loc[:, df_input.columns.str.startswith(g1)]
            df_compare2 = df_input.loc[:, df_input.columns.str.startswith(g2)]
            df_compare1 = df_compare1.astype('float64')
            df_compare2 = df_compare2.astype('float64')
            sample_len = int(df_compare1.shape[1]) + int(df_compare2.shape[1])
            df_com = pd.concat([df_compare1, df_compare2], axis=1)

            X = df_com.T
            if self.pca_log == True:
                X.replace(0, np.nan)
                X = np.log10(X)  # log对数转换
            if self.pca_scaling == "Ctr":
                X = X.sub(X.mean(axis=0), axis=1)  # scaling="center"
            elif self.pca_scaling == "UV":
                X = X.sub(X.mean(axis=0), axis=1) / X.std(axis=0)  # scaling="UV"
            else:
                X = X.sub(X.mean(axis=0), axis=1) / np.sqrt(X.std(axis=0))  # scaling="Par"
            #X_index = X.index
            X[X.isnull()] = 0
            g1_len = len(X.index[X.index.str.startswith(g1)])
            g2_len = len(X.index[X.index.str.startswith(g2)])
            y = [0 for i in range(g1_len)] + [1 for i in range(g2_len)]
            y = np.array(y)
            target_names = [g1, g2]
            colors = ["navy", "turquoise"]
            pca = PCA(min(10, min(X.shape)))  # n_components=2
            pca.fit(X)
            # X_r = pca.fit(X).transform(X)
            summary_df_res = pca.transform1(X)
            summary_df = summary_df_res[1]
            if summary_df.iloc[0, 1] < 3:
                pca = PCA(n_components=3)  # n_components=2
                pca.fit(X)
                # X_r = pca.fit(X).transform(X)
                summary_df_res = pca.transform1(X)
                summary_df = summary_df_res[1]
            model.append(summary_df.iloc[0, 1])
            model.append(sample_len)
            model.append(summary_df.iloc[0,0])
            model.append(g1+"-"+g2)
            model = pd.DataFrame(model).T
            model.columns = ["Model", "Type", "A", "N", "R2X(cum)", "Title"]
            k = k + 1
            self.pca_model_param = pd.concat([self.pca_model_param, model], axis=0)
            X_r = summary_df_res[0]
            X_score = X_r[:, [0,1]]
            n = X_score.shape[0]
            hfn = 2*(n-1)*(n**2-1)/(n**2*(n-2))*f.ppf(self.ci, 2, n-2)
            rv = np.linspace(0, 2*np.pi, 100)
            x1 = np.sqrt(np.var(X_r[:, 0], ddof=1)*hfn)*np.cos(rv)
            y1 = np.sqrt(np.var(X_r[:, 1], ddof=1)*hfn)*np.sin(rv)
            df_score = pd.DataFrame([x1, y1]).T
            pc1_ratio = round(pca.explained_variance_ratio_[0]*100, 1)
            pc2_ratio = round(pca.explained_variance_ratio_[1]*100, 1)
            plt.figure()
            lw = 2
            for color, i, target_name in zip(colors, [0, 1], target_names):
                plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
            plt.plot(x1, y1, c="k")
            plt.axhline(y= 0, color="grey")
            plt.axvline(x= 0, color="grey")#linestyle="--"
            plt.xlabel('PC1[' + str(pc1_ratio) + "%]")
            plt.ylabel('PC2[' + str(pc2_ratio) + "%]")
            plt.legend(loc="best", shadow=False, scatterpoints=1)
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            plt.savefig(self.path + g1 + "-" + g2 + "/PCA.png", dpi=500, bbox_inches="tight")
            plt.close()
            path_list.append(g1 + "-" + g2 + '/PCA.png')
        return path_list

    def opls(self):
        filterData = self.mean_data
        filterData_name = filterData.columns[filterData.columns.str.startswith("Mean ")]
        filterData = filterData.drop(filterData_name, axis=1)
        path_list = []
        df_desc = filterData.loc[:, filterData.columns[0:self.descNum]]
        df_input = filterData.drop(list(filterData.columns[0:self.descNum]), axis=1)
        col = 0
        Gseq = -1
        # 修改列名,把样本名(即列名)前面加上所属组名
        for i in self.sample.split(" "):
            Gseq = Gseq + 1
            col = col + int(i)
            df_input_subGroup = df_input.loc[:, df_input.columns[(col - int(i)):col]]
            for n in range(len(df_input_subGroup.columns)):  # 把每组的样本名前加上所属的组的信息，方便下面对比时进行索引
                new_name = self.group.split(" ")[Gseq] + "_" + df_input_subGroup.columns[n]
                df_input = df_input.rename(columns={df_input_subGroup.columns[n]: new_name})
        # 计算各对比的FC值和P值，并筛选出差异物质
        # df_com_res = list()  # 用于保存各对比的处理后的数据
        # df_com_dec_res = list()  # 用于保存处理后差异物质的数据
        self.oplsda_model_param = pd.DataFrame()
        k = 1
        for i in self.contrast.split(" "):
            model = []
            model.append("model"+str(k))
            model.append("OPLS-DA")
            model.append("1+1+0")
            g1 = i.split(":")[0]
            g2 = i.split(":")[1]
            df_compare1 = df_input.loc[:, df_input.columns.str.startswith(g1)]
            df_compare2 = df_input.loc[:, df_input.columns.str.startswith(g2)]
            df_compare1 = df_compare1.astype('float64').T
            df_compare2 = df_compare2.astype('float64').T
            g1_len = df_compare1.shape[0]
            g2_len = df_compare2.shape[0]
            sample_len = int(df_compare1.shape[0]) + int(df_compare2.shape[0])
            model.append(sample_len)
            df_com = pd.concat([df_compare1, df_compare2], axis=0)
            Y = np.array([0 for _ in range(df_compare1.shape[0])] + [1 for j in range(df_compare2.shape[0])])
            Y = check_array(Y, dtype=np.float64, copy=True, ensure_2d=False)
            X = df_com
            if self.opls_log == True:
                X.replace(0, np.nan)
                X = np.log10(X)  # log对数转换
            opls = OPLS(n_components=1, scale=self.opls_scaling)
            opls.fit(X, Y)
            model_ratio = opls.fit1(X, Y)[1]
            model.append(model_ratio.iloc[2, 1])
            model.append(model_ratio.iloc[2, 3])
            model.append(model_ratio.iloc[2, 5])
            model.append(g1+"-"+g2)
            model = pd.DataFrame(model)
            k = k + 1
            self.oplsda_model_param = pd.concat([self.oplsda_model_param, model.T], axis=0)
            P_ratio = round(model_ratio.iloc[0, 0] * 100, 1)
            O_ratio = round(model_ratio.iloc[1, 0] * 100, 1)
            vip_res = opls.vip
            self.vip.append(vip_res)
            df = pd.DataFrame(np.column_stack([opls.T[:, 0], opls.T_ortho_[:, 0]]),
                              index=X.index, columns=['t', 't_ortho'])
            name_list = list(df.index)
            sample_name_list = [name_list[i][len(g1) + 1:] for i in range(g1_len)] + [name_list[i][len(g2) + 1:] for i
                                                                                      in range(g1_len, g1_len + g2_len)]
            group1_df = df[Y == 0]
            group2_df = df[Y == 1]
            n = df.shape[0]
            hfn = 2 * (n - 1) * (n ** 2 - 1) / (n ** 2 * (n - 2)) * f.ppf(self.ci, 2, n - 2)
            rv = np.linspace(0, 2 * np.pi, 100)
            x = np.sqrt(np.var(df["t"], ddof=1) * hfn) * np.cos(rv)
            y = np.sqrt(np.var(df["t_ortho"], ddof=1) * hfn) * np.sin(rv)
            df_score = pd.DataFrame([x, y]).T
            plt.scatter(group1_df['t'], group1_df['t_ortho'], c='red', label= g1)
            plt.scatter(group2_df['t'], group2_df['t_ortho'], c='blue', label= g2)
            plt.plot(x, y, c="k")
            plt.axhline(y=0, color="black")
            plt.axvline(x=0, color="black")  # linestyle="--"
            plt.title('OPLS Scores')
            plt.xlabel('t[1]P['+str(P_ratio)+"%]")
            plt.ylabel('t[1]O['+str(O_ratio)+"%]")
            plt.legend(loc='upper right')
            plt.legend(loc="best", shadow=False, scatterpoints=1)
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            plt.savefig(self.path + g1 + "-" + g2 + "/opls.png", dpi=500, bbox_inches="tight")
            plt.close()


            ##做置换检验图
            df_p = opls.permutation(X, Y)
            df_p["persim"] = 1 - df_p["persim"]
            df_p["persim"].values[0] = 1
            df_p = df_p[df_p["persim"] > 0]
            df_p1 = df_p.copy()
            df_p1["R2Y_cum"] = df_p1["R2Y_cum"] - df_p1["R2Y_cum"].values[0]
            df_p1["Q2_cum"] = df_p1["Q2_cum"] - df_p1["Q2_cum"].values[0]
            df_p1["persim"] = df_p1["persim"] - 1
            fit_r2 = ols("R2Y_cum ~ persim+0", df_p1).fit()
            fit_q2 = ols("Q2_cum ~ persim+0", df_p1).fit()
            df_lm = pd.DataFrame(
                {"x1": [0], "x2": [1], "y1_r": [df_p["R2Y_cum"].values[0] - fit_r2.params[0]],
                 "y2_r": [df_p["R2Y_cum"][0]],
                 "y1_q": [df_p["Q2_cum"][0] - fit_q2.params[0]], "y2_q": [df_p["Q2_cum"][0]]})
            df_p2 = df_p[["R2Y_cum", "Q2_cum", "persim"]] #.iloc[1:, :]
            df_melt = pd.melt(df_p2, id_vars=["persim"])
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            ax1.plot([df_lm["x1"][0], df_lm["x2"][0]], [df_lm["y1_r"][0], df_lm["y2_r"][0]], "k--", lw=3, alpha=0.6)
            ax1.plot([df_lm["x1"][0], df_lm["x2"][0]], [df_lm["y1_q"][0], df_lm["y2_q"][0]], "k--", lw=3, alpha=0.6)
            ax1.scatter(df_melt[df_melt["variable"] == "R2Y_cum"]["persim"].values,
                        df_melt[df_melt["variable"] == "R2Y_cum"]["value"].values,
                        marker="o", color="g", alpha=0.8)
            ax1.scatter(df_melt[df_melt["variable"] == "Q2_cum"]["persim"].values,
                        df_melt[df_melt["variable"] == "Q2_cum"]["value"].values,
                        marker="s", color="b", alpha=0.8)
            ax1.set_title("Intercepts: " + r"$R^2$" + "Y(cum)=(0," + str(round(df_lm["y1_r"].values[0], 2)) + "), "
                          + r"$Q^2$" + "(cum)=(0," + str(round(df_lm["y1_q"].values[0], 2)) + ")", fontsize=13,
                          color="r")
            ax1.set_xlabel("Correlation Coefficient", fontsize=13, color="r")
            # ax1.set_ylabel("R2Y(cum)andQ2(cum)",fontsize=13, color="r")
            ax1.set_ylabel(r"$R^2$" + "Y(cum)and" + r"$Q^2$" + "(cum)", fontsize=13, color="r")
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            plt.savefig(self.path + g1 + "-" + g2 + "/opls_permutation.png", dpi=500, bbox_inches="tight")
            plt.close()

            # 置换检验柱状图
            dff_res = []
            dff_p = opls.permutation(X, Y)
            dff_p["persim"] = 1 - dff_p["persim"]
            dff_p["persim"].values[0] = 1

            R2Y1 = np.round(dff_p[dff_p.columns[1]], 3)
            R2Y = pd.cut(R2Y1, bins=np.round([(i + 0.02) for i in np.arange(-1.03, 1, 0.02)], 2))
            R2Y_data = []
            for i in range(len(R2Y)):
                R2Y_data.append((R2Y[i].left + R2Y[i].right) / 2)

            Q21 = np.round(dff_p[dff_p.columns[2]], 3)
            # Q2 = pd.cut(Q21, bins=np.round([(i + 0.02) for i in np.arange(-1.03, 1, 0.02)], 2))
            Q2 = pd.cut(Q21, bins=np.round([i for i in np.arange(np.round(np.min(Q21), 2) - 0.01, 1.02, 0.02)], 2))
            Q2_data = []
            for i in range(len(Q2)):
                Q2_data.append((Q2[i].left + Q2[i].right) / 2)

            dff_data = dff_p
            dff_data[dff_data.columns[0:3]] = np.round(dff_data[dff_data.columns[0:3]], 3)

            R2Y_len = sum(dff_data[dff_data.columns[1]] > dff_data[dff_data.columns[1]][0])
            Q2_len = sum(dff_data[dff_data.columns[2]] > dff_data[dff_data.columns[2]][0])
            P_R2Y = sum(dff_data[dff_data.columns[1]] > dff_data[dff_data.columns[1]][0]) / 200
            P_Q2 = sum(dff_data[dff_data.columns[2]] > dff_data[dff_data.columns[2]][0]) / 200

            # dff_all = pd.DataFrame({"R2Y": R2Y_data, "Q2": Q2_data,
            #                         "R2Y_Permutations": R2Y_data[0]-0.02, "R2Y_Frequency": 110,
            #                         "Q2_Permutations": Q2_data[0]-0.02, "Q2_Frequency": 160})
            R2Y_X = []
            R2Y_Y = []
            [R2Y_X.append(i) for i in R2Y_data if i not in R2Y_X]
            [R2Y_Y.append(R2Y_data.count(i)) for i in R2Y_X]

            Q2_X = []
            Q2_Y = []
            [Q2_X.append(i) for i in Q2_data if i not in Q2_X]
            [Q2_Y.append(Q2_data.count(i)) for i in Q2_X]


            name = ["R2Y"] * len(R2Y_X)
            name.extend(["Q2"] * len(Q2_X))
            # name.extend(["P_R2Y", "P_Q2"])
            R2Y_X.extend(Q2_X)
            # R2Y_X.extend([P_R2Y, P_Q2])
            R2Y_Y.extend(Q2_Y)
            # R2Y_Y.extend([110, 160])
            dict1 = {}
            if P_R2Y <= 0.05:
                dict1['P_R2Y'] = 'p<0.05' + '(' + str(R2Y_len) + '/' + '200)'
            else:
                dict1['P_R2Y'] = 'p>0.05' + '(' + str(R2Y_len) + '/' + '200)'
            if P_Q2 <= 0.05:
                dict1['P_Q2'] = 'p<0.05' + '(' + str(Q2_len) + '/' + '200)'
            else:
                dict1['P_Q2'] = 'p>0.05' + '(' + str(Q2_len) + '/' + '200)'
            dict1["Q2"] = df_p[0][2]
            dict1["R2Y"] = df_p[0][1]

            dff_all = pd.DataFrame({"name": name,
                                    "Permutations": R2Y_X,
                                    "Frequency": R2Y_Y}
                                   )
            dff_res.append(dff_all)
            dff_res.append(dict1)

        self.oplsda_model_param.columns = ["Model", "Type", "A", "N", "R^2X(cum)", "R^2Y(cum)", "Q^2(cum)", "Title"]
        return dff_res

    def volcanoplot(self):
        """
        使用每组的差异表根据logFC、Pvalue和vip划分上调下调和不显著，绘制火山图
        """
        contrast_list = self.contrast.split(" ")
        path_list = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            data = self.difference[0][i]
            tmp = pd.DataFrame()
            tmp["x"] = data["logFC"]
            tmp["y"] = data["Pvalue"]
            tmp["VIP"] = data["vip"]
            tmp["-log10(pvalue)"] = -data["Pvalue"].apply(np.log10)
            tmp.loc[(tmp.x > np.log2(self.FC_up)) & (tmp.y < self.pvalue) & (tmp.VIP > self.vip_cut), "Status"] = "up-regulated"
            tmp.loc[(tmp.x < np.log2(self.FC_down)) & (tmp.y < self.pvalue) & (tmp.VIP > self.vip_cut), "Status"] = "down-regulated"
            tmp.loc[tmp["Status"].isna(), "Status"] = "not significant"
            x_intercept = [np.log2(self.FC_down), np.log2(self.FC_up)]
            y_intercept = -np.log10(self.pvalue)
            sns.scatterplot(x="x", y="-log10(pvalue)", hue="Status", size="VIP", data=tmp,
                            hue_order=("down-regulated", "not significant", "up-regulated"),
                            palette=("#377EB8", "grey", "#E41A1C"), alpha=0.5, s=20)
            plt.axhline(y=y_intercept, color="grey", linestyle="--")
            plt.axvline(x=x_intercept[0], color="grey", linestyle="--")
            plt.axvline(x=x_intercept[1], color="grey", linestyle="--")
            plt.legend(loc='center left', bbox_to_anchor=(1.04, 0.5))
            plt.xlabel("log2 Fold Change")
            plt.ylabel("-log10 P-value")
            plt.tight_layout()
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            plt.savefig(self.path + g1 + "-" + g2 + '/volcano.png', dpi=300, bbox_inches="tight")
            path_list.append(g1 + "-" + g2 + '/volcano.png')
            plt.close()
        return path_list

    def f_test(self, df1, df2, length1=0, length2=0):
        """
        方差齐性检验
        返回双侧检验的p值
        """
        len1 = length1 - 1
        len2 = length2 - 1
        F = np.var(df1[:length1]) / np.var(df2[length1:length1 + length2])
        single_tailed_pvalue = f.cdf(F, len1, len2)
        double_tailed_pvalue = single_tailed_pvalue * 2
        return double_tailed_pvalue

    def get_vip(self):
        """
        根据mean表计用opls算每组对比的vip值
        返回vip值列表
        """
        filterData = self.mean_data
        filterData_name = filterData.columns[filterData.columns.str.startswith("Mean ")]
        filterData = filterData.drop(filterData_name, axis=1)
        path_list = []
        df_desc = filterData.loc[:, filterData.columns[0:self.descNum]]
        df_input = filterData.drop(list(filterData.columns[0:self.descNum]), axis=1)
        col = 0
        Gseq = -1
        # 修改列名,把样本名(即列名)前面加上所属组名
        for i in self.sample.split(" "):
            Gseq = Gseq + 1
            col = col + int(i)
            df_input_subGroup = df_input.loc[:, df_input.columns[(col - int(i)):col]]
            for n in range(len(df_input_subGroup.columns)):  # 把每组的样本名前加上所属的组的信息，方便下面对比时进行索引
                new_name = self.group.split(" ")[Gseq] + "_" + df_input_subGroup.columns[n]
                df_input = df_input.rename(columns={df_input_subGroup.columns[n]: new_name})
        vip_list = []
        for i in self.contrast.split(" "):
            g1 = i.split(":")[0]
            g2 = i.split(":")[1]
            df_compare1 = df_input.loc[:, df_input.columns.str.startswith(g1)]
            df_compare2 = df_input.loc[:, df_input.columns.str.startswith(g2)]
            df_compare1 = df_compare1.astype('float64').T
            df_compare2 = df_compare2.astype('float64').T
            df_com = pd.concat([df_compare1, df_compare2], axis=0)
            Y = np.array([0 for _ in range(df_compare1.shape[0])] + [1 for j in range(df_compare2.shape[0])])
            Y = check_array(Y, dtype=np.float64, copy=True, ensure_2d=False)
            X = df_com
            if self.opls_log == True:
                X.replace(0, np.nan)
                X = np.log10(X)  # log对数转换
            opls = OPLS(n_components=1, scale="UV")
            opls.fit(X, Y)
            vip_res = opls.vip
            vip_res.columns = ['vip']
            vip_list.append(vip_res)
        return vip_list

    def p_adjust(self, p, method):
        """
        Benjamini-Hochberg p-value correction for multiple hypothesis testing.
        """
        p = np.asfarray(p)
        if method == "fdr":
            by_descend = p.argsort()[::-1]
            by_orig = by_descend.argsort()
            steps = float(len(p)) / np.arange(len(p), 0, -1)
            q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
            return q[by_orig]
        elif method == "holm":
            by_inccend = p.argsort()
            by_orig = by_inccend.argsort()
            steps = float(len(p)) - np.arange(len(p))
            q = np.minimum(1, np.maximum.accumulate(steps * p[by_inccend]))
            return q[by_orig]

    def zscorePlot(self):
        df_zscore_list = []
        contrast_list = self.contrast.split(" ")
        sample_list = self.sample.split(" ")
        group_list = self.group.split(" ")

        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            g1_len = int(sample_list[group_list.index(g1)])
            g2_len = int(sample_list[group_list.index(g2)])

            df = self.difference[1][i]
            df_in = df.drop(df.columns[0:self.descNum], axis=1).drop(['FC', 'Pvalue', 'logFC', "vip"], axis=1)
            zscoredata = df_in.apply(lambda x: (x - x.mean()) / x.std(), axis=1)

            zscoredata_1 = zscoredata.iloc[:, 0:g1_len]
            zscoredata_2 = zscoredata.iloc[:, g1_len:]

            zscoredata_1.loc[:, "NAME_EN"] = df.loc[:, "NAME_EN"].tolist()
            zscoredata_2.loc[:, "NAME_EN"] = df.loc[:, "NAME_EN"].tolist()

            zscoredata_1_long = pd.melt(zscoredata_1, id_vars="NAME_EN")
            zscoredata_2_long = pd.melt(zscoredata_2, id_vars="NAME_EN")

            zscoredata_1_long.sort_values(by='NAME_EN', inplace=True)
            zscoredata_2_long.sort_values(by='NAME_EN', inplace=True)

            data = {
                g1: (zscoredata_1_long.loc[:, ["NAME_EN", "value"]]).values.tolist(),
                g2: (zscoredata_2_long.loc[:, ["NAME_EN", "value"]]).values.tolist()
            }
            df_zscore_list.append(data)
        return df_zscore_list

    def heatmap(self):
        '''
        绘制每组对比差异物质的层次聚类热图
        返回差异物质的层次聚类图
        '''
        contrast_list = self.contrast.split(" ")
        path_list = []
        sample_list = self.sample.split(" ")
        group_list = self.group.split(" ")
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            g1_len = int(sample_list[group_list.index(g1)])
            g2_len = int(sample_list[group_list.index(g2)])
            col_names = np.repeat([g1, g2], [g1_len, g2_len])
            data = self.difference[1][i]
            data = data.drop(data.columns[2:self.descNum], axis=1).drop(['FC', 'Pvalue', 'logFC', "vip"], axis=1)
            data1 = data.drop(["id","NAME_EN"], axis=1)
            data1.index = data["NAME_EN"]
            fig_data1 = sns.clustermap(data=data1.loc[:,data1.columns.str.startswith(g1)],
                           method='complete',  # 'average'算法
                           metric='euclidean',  # 欧式距离'euclidean'
                           figsize=(10, 15),
                           row_cluster=False,  # 行方向聚类
                           col_cluster=True,  # 列方向不聚类
                           cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors#,
                           #z_score=0
                           ).data2d
            plt.close()
            fig_data2 = sns.clustermap(data=data1.loc[:, data1.columns.str.startswith(g2)],
                                       method='complete',  # 'average'算法
                                       metric='euclidean',  # 欧式距离'euclidean'
                                       figsize=(10, 15),
                                       row_cluster=False,  # 行方向聚类
                                       col_cluster=True,  # 列方向不聚类
                                       cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors  # ,
                                       # z_score=0
                                       ).data2d
            plt.close()
            fig_data = pd.concat([fig_data1, fig_data2], axis=1)
            fig_data_tmp = data1[fig_data.columns]
            fig_data_res = sns.clustermap(data=fig_data,
                                       method='complete',  # 'average'算法
                                       metric='euclidean',  # 欧式距离'euclidean'
                                       figsize=(10, 15),
                                       row_cluster=True,  # 行方向聚类
                                       col_cluster=False,  # 列方向不聚类
                                       cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
                                       z_score=0
                                       ).data2d
            # plt.close()
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            fig_data_res.to_excel(self.path + g1 + "-" + g2 + '/Hierarchical_clustering_data.xlsx')
            plt.savefig(self.path + g1 + "-" + g2 + '/cluster.png', dpi=300, bbox_inches="tight")
            path_list.append(g1 + "-" + g2 + '/cluster.png')
        return path_list

    def heatmapAll(self):
        '''
        绘制每组对比差异物质的层次聚类热图，
        返回差异物质的层次聚类图
        '''
        heatmap_data = self.mean_data
        heatmap_data = heatmap_data.loc[heatmap_data['NAME_EN'].notna(), :]
        sample_name = list(heatmap_data.columns[13:])
        group = self.group
        group_list = group.split(" ")
        sample_num = self.sample
        sample_list = list(map(int, sample_num.split(" ")))
        col_names = np.repeat(group_list, sample_list)
        heatmpa_name = heatmap_data.columns[heatmap_data.columns.str.startswith("Mean ")]
        heatmap_data1 = heatmap_data.drop(heatmpa_name, axis=1)
        heatmap_data1 = heatmap_data1.iloc[:, 13:]
        QC_name = heatmap_data1.columns[heatmap_data1.columns.str.startswith("QC")]
        heatmap_data1 = heatmap_data1.drop(QC_name, axis=1)
        heatmap_data1.index = heatmap_data["NAME_EN"]
        heatmap_data1 = heatmap_data1.apply(pd.to_numeric, errors='coerce')
        sample_name1 = list(heatmap_data1.columns)
        fig_data1 = pd.DataFrame()
        for i in sample_list:
            tmp_name = sample_name1[:i]
            [sample_name1.remove(j) for j in sample_name1[:i]]
            tmp_data = heatmap_data1[tmp_name]
            fig_data_tmp = sns.clustermap(data=tmp_data,
                                       method='complete',  # 'average'算法
                                       metric='euclidean',  # 欧式距离'euclidean'
                                       figsize=(10, 15),
                                       row_cluster=False,  # 行方向不聚类
                                       col_cluster=True,  # 列方向聚类
                                       cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors  # ,
                                       # z_score=0
                                       ).data2d
            plt.close()
            fig_data1 = pd.concat([fig_data1, fig_data_tmp], axis=1)
        fig_data2 = heatmap_data1[fig_data1.columns]
        fig_data_res = sns.clustermap(data=fig_data2,
                                      method='complete',  # 'average'算法
                                      metric='euclidean',  # 欧式距离'euclidean'
                                      figsize=(10, 15),
                                      row_cluster=True,  # 行方向聚类
                                      col_cluster=False,  # 列方向不聚类
                                      cmap=palettable.cartocolors.diverging.ArmyRose_7.mpl_colors,
                                      z_score=0
                                      ).data2d
        plt.close()
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        fig_data_res.to_excel(self.path + 'Hierarchical_clustering_data_TOTAL.xlsx')
        return fig_data_res

    def boxplot(self):

        def f(x):  # 用于计算箱线图的上边缘，Q3，median, Q1，下边缘
            Q1, median, Q3 = np.percentile(x, [25, 50, 75])
            IQR = Q3 - Q1
            up = np.max(x[x <= (Q3 + 1.5*IQR)])
            down = np.min(x[x >= (Q1 - 1.5*IQR)])
            return pd.Series([up, Q3, median, Q1, down])

        contrast_list = self.contrast.split(" ")
        # path_list = []
        df_box_list = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            df = self.difference[1][i].copy(deep=True)
            df_in = df.drop(df.columns[0:self.descNum], axis=1).drop(['FC', 'Pvalue', 'logFC', "vip"], axis=1)
            df1 = df_in.loc[:, df_in.columns.str.startswith(g1)]
            df2 = df_in.loc[:, df_in.columns.str.startswith(g2)]

            df1_box = df1.apply(f, axis=1)
            df1_box.columns = ['up', 'Q3', 'median', 'Q1', 'down']

            df2_box = df2.apply(f, axis=1)
            df2_box.columns = ['up', 'Q3', 'median', 'Q1', 'down']

            df_result = pd.DataFrame({"NAME_EN": df["NAME_EN"], "Pvalue": df["Pvalue"],
                                      g1: df1.values.tolist(),
                                      g2: df2.values.tolist(),
                                      g1 + "_boxdata": df1_box.values.tolist(),
                                      g2 + "_boxdata": df2_box.values.tolist()
                                      })
            df_result = df_result.reset_index()
            ##df_result.to_excel(r"E:\pycode\py_yun\lqq\A-B\box.xlsx")
            data = {}
            for j in range(len(df_result)):
                met = df_result["NAME_EN"][j]
                P = round(df_result["Pvalue"][j],4)
                data[met] = [
                    {
                        "name": g1,
                        "point_values": df_result[g1][j],
                        "box_values": df_result[g1 + "_boxdata"][j],  # 顺序：上边缘，Q3，median, Q1，下边缘
                        "P": P
                    },
                    {
                        "name": g2,
                        "point_values": df_result[g2][j],
                        "box_values": df_result[g2 + "_boxdata"][j],  # 顺序：上边缘，Q3，median, Q1，下边缘
                        "P": P
                    },
                ]
            df_box_list.append(data)
        return df_box_list

    def vennplot(self, venn_list_num=None):
        '''

        :param venn_list_num: 默认为None时候,做所有对比的venn图。可以填数组，如[0,2,3],代表做第0，2，3个对比的Venn图。注意：对比策略的顺序从0开始。
        :return:字典venn_data，venn_list_num长度小于5时返回韦恩图画图数据，大于等于6时返回upsetplot画图数据
        '''

        from functools import reduce

        contrast_list = self.contrast.split(" ")
        if len(contrast_list) > 1:
            ##venn_list_num为None的时候，venn_list_num，取所有对比的序号，做所有对比的venn
            if venn_list_num is None:
                venn_list_num = list(range(len(contrast_list)))

            group = [contrast_list[i] for i in venn_list_num]
            sets_data = [set(self.difference[1][i]["NAME_EN"]) for i in venn_list_num]
            s_all = reduce(lambda i, j: i | j, sets_data)  # 所有物质

            venn_matrix = pd.DataFrame(data=None, index=s_all, columns=group)
            for i in range(len(group)):
                venn_matrix.iloc[:, i] = venn_matrix.index.isin(sets_data[i]).astype(int)
            # venn_matrix.to_excel(self.path+re.sub(":","","_VS_".join(group))+"_venn_matrix.xlsx")

            # 把每一行的数据转换成字符串，如[1,1,0,0,0]-->'11000'
            venn_select = venn_matrix.apply(lambda x: "".join(x.astype(str).values.tolist()), 1)
            N = len(venn_list_num)
            all_type = [bin(n).split('0b')[-1].zfill(N) for n in range(1, 2 ** N)]

            # 统计venn图每个区域包含的物质和数量
            venn_dict = {}
            groups = pd.Series(group)
            for ii in all_type:
                dem = venn_select[venn_select == ii].index.tolist()
                dem_num = len(dem)
                num = pd.Series(list(ii))
                set1 = groups[num == '1'].str.cat(sep=' ∩ ')
                set2 = groups[num == '0'].str.cat(sep=' - ')
                if len(set2) == 0:
                    venn_des = set1
                else:
                    venn_des = set1 + " - " + set2
                venn_dict[ii] = {
                    'dem': dem,  # 物质
                    'dem_num': dem_num,  # 数量
                    'description': venn_des  # 描述
                }

            # 画图需要的数据顺序
            venn_pd = pd.DataFrame(venn_dict).T
            # venn_pd.to_excel(self.path+re.sub(":","","_VS_".join(group))+".xlsx")

            if len(venn_list_num) == 2:
                venn_data = {
                    'group': group,
                    'value': list(venn_pd.loc[["10", "01", "11"], "dem_num"])
                }

            elif len(venn_list_num) == 3:
                venn_data = {
                    'group': group,
                    'value': list(venn_pd.loc[["100", "010", "001", "110", "101", "011", "111"], "dem_num"])
                }

            elif len(venn_list_num) == 4:
                venn_data = {
                    'group': group,
                    'value': list(venn_pd.loc[["1000", "0100", "0010", "0001",
                                               "1100", "1010", "1001", "0110", "0101", "0011",
                                               "1110", "1101", "1011", "0111",
                                               "1111"], "dem_num"])
                }

            elif len(venn_list_num) == 5:
                venn_data = {
                    'group': group,
                    'value': list(venn_pd.loc[["10000", "01000", "00100", "00010", "00001",
                                               "11000", "10100", "10010", "10001", "01100", "01010", "01001", "00110",
                                               "00101",
                                               "00011",
                                               "11100", "11010", "11001", "10110", "10101", "10011", "01110", "01101",
                                               "01011",
                                               "00111",
                                               "11110", "11101", "11011", "10111", "01111",
                                               "11111"], "dem_num"])
                }

            elif len(venn_list_num) >= 6:
                pass

            return venn_data
        else:
            return '一组对比无法做韦恩图分析'

    def get_kmeans(self):

        from functools import reduce
        from sklearn.cluster import KMeans

        diff_temp = reduce(lambda i, j: i + j, [x.index.tolist() for x in self.difference[1]])  # 所有对比的差异代谢物并集
        diff = list(set(diff_temp))  # 差异代谢物去重

        # 获取差异代谢物每一组mean数据
        mean_data1 = self.mean_data.copy(deep=True)
        mean_data1 = mean_data1.loc[diff,:] #mean_data1的MS2.name后面会用到
        mean_data2 = mean_data1.loc[:, mean_data1.columns.str.startswith("Mean ")]
        mean_data2 = mean_data2.drop(["Mean QC"], axis=1, errors='ignore')

        mean_scale = mean_data2.apply(lambda x: (x - x.mean()) / x.std(), axis=1)  # 标准化
        mean_scale.columns = mean_scale.columns.str[5:]  # 列名去掉Mean

        #根据差异代谢物总数量确认center
        if len(mean_scale.index) >= 150:
            center = 9
        elif len(mean_scale.index) >= 7:
            center = 7
        elif len(mean_scale.index) >= 2:
            center = 2
        elif len(mean_scale.index) == 1:
            center = 1

        #kmeans
        k_means = KMeans(n_init=25, max_iter=200, random_state=1234, n_clusters=center)
        cluster = k_means.fit(mean_scale)
        #获取cluster,KMeans的结果从0开始，这边统一加1变成从1开始
        kmeans_result = mean_scale.copy(deep=True)
        kmeans_result["cluster"] = (cluster.labels_)+1
        kmeans_result["Metabolites"] = mean_data1["NAME_EN"]
        #获取cluster_centers_，index统一加1
        centers = pd.DataFrame(cluster.cluster_centers_)
        centers.index=[ int(i)+1 for i in centers.index.tolist()]
        centers.columns=mean_scale.columns
        # kmeans_result.to_excel(B.path+'kmeans_result.xlsx')
        # centers.to_excel(B.path + 'centers.xlsx')

        #整理数据
        kmeanspalette=["#F8766D", "#D39200", "#93AA00", "#00BA38", "#00C19F", "#00B9E3", "#619CFF", "#DB72FB", "#FF61C3"]
        y_max = mean_scale.max().max()
        y_min = mean_scale.min().min()
        kmeans_data={}
        for i in range(1,center+1):
            data_temp = kmeans_result.loc[kmeans_result["cluster"]==i,mean_scale.columns]##取对应cluster的数据
            data_list_temp = np.array(data_temp).tolist()
            data_columns = data_temp.columns.tolist()
            kmeans_data["cluster"+str(i)]={
                "max": y_max,
                "min": y_min,
                "metabolites":len(data_temp),
                "color":kmeanspalette[i-1],
                "data":[list(map(list,zip(data_columns, ii))) for ii in data_list_temp],
                "specialData":list(map(list,zip(centers.columns.tolist(), centers.loc[i,:].tolist()) ) )
            }
        return kmeans_data

    def matchstick(self,shownum=15):
        contrast_list = self.contrast.split(" ")
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            file_name = g1 + "-" + g2
            dec = self.difference[1][i]

            dec_i = dec.loc[:,["NAME_EN","logFC","Pvalue","vip"]]
            dec_i = dec_i[dec_i["NAME_EN"].notna()]
            dec_i["logFC"] = dec_i["logFC"].astype('float64') ###rocdata2 = rocdata2.astype({'classID': 'float'})
            if (dec_i.shape[0] != 0):
                ###下调
                if (dec_i.loc[dec_i['logFC'] < 0,:].shape[0] != 0):
                    dec_i_down = dec_i.loc[dec_i['logFC'] < 0, :]
                    dec_i_down = dec_i_down.sort_values(by='logFC', axis=0)  # 升序
                    dec_i_down.loc[:,"col"] = "blue"
                    dec_i_down15 = dec_i_down.head(min(dec_i_down.shape[0], shownum))
                    dec_i_down15.reset_index(inplace=True)
                    data_down=[]
                    for i in range(len(dec_i_down15)):
                        # print(i)
                        if(dec_i_down15["Pvalue"][i]>=0 and dec_i_down15["Pvalue"][i]<0.001):
                            star = 3
                        elif(dec_i_down15["Pvalue"][i]>=0.001 and dec_i_down15["Pvalue"][i]<0.01):
                            star = 2
                        elif(dec_i_down15["Pvalue"][i]>=0.01 and dec_i_down15["Pvalue"][i]<0.05):
                            star = 1
                        data = {
                            "name": dec_i_down15["NAME_EN"][i],
                            "value": round(dec_i_down15["logFC"][i], 3),
                            "color": dec_i_down15["col"][i],
                            "alpha": round(dec_i_down15["vip"][i], 3),
                            "star": star
                        }
                        data_down.append(data)
                    min_value=round(min(dec_i_down15["logFC"]), 3)

                ###上调
                if (dec_i.loc[dec_i['logFC'] >= 0,:].shape[0] != 0):
                    dec_i_up = dec_i.loc[dec_i['logFC'] >= 0, :]
                    dec_i_up = dec_i_up.sort_values(by='logFC', ascending=False)  # 降序
                    dec_i_up.loc[:, "col"] = "red"
                    dec_i_up15 = dec_i_up.head(min(dec_i_up.shape[0], shownum))
                    dec_i_up15.reset_index(inplace=True)
                    data_up = []
                    for i in range(len(dec_i_up15)):
                        # print(i)
                        if (dec_i_up15["Pvalue"][i] >= 0 and dec_i_up15["Pvalue"][i] < 0.001):
                            star = 3
                        elif (dec_i_up15["Pvalue"][i] >= 0.001 and dec_i_up15["Pvalue"][i] < 0.01):
                            star = 2
                        elif (dec_i_up15["Pvalue"][i] >= 0.01 and dec_i_up15["Pvalue"][i] < 0.05):
                            star = 1
                        data = {
                            "name": dec_i_up15["NAME_EN"][i],
                            "value": round(dec_i_up15["logFC"][i], 3),
                            "color": dec_i_up15["col"][i],
                            "alpha": round(dec_i_up15["vip"][i], 3),
                            "star": star
                        }
                        data_up.append(data)
                    max_value = round(max(dec_i_up15["logFC"]), 3)

            # data_down_df = pd.DataFrame(data_down,columns=["name","value","color","alpha","star"])
            # data_up_df = pd.DataFrame(data_up, columns=["name", "value", "color", "alpha", "star"])

            data_out={
                "down":data_down,
                "up":data_up,
                "max":max_value,
                "min":min_value
            }
            return (data_out)

    def corrplot(self):
        """
        绘制每组差异物质的相关性热力图
        返回每组对比差异物质的相关性热力图
        """
        contrast_list = self.contrast.split(" ")
        path_list = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            data = self.difference[1][i]
            data = data.drop(data.columns[2:self.descNum], axis=1).drop(['FC', 'Pvalue', 'logFC', "vip"], axis=1)
            data = data.drop(["id"], axis=1)
            data1 = pd.DataFrame(data[list(data.columns)[1:]].values.T, columns=data.iloc[:, 0:1])
            corr_data = data1.corr()
            annot_label = pd.DataFrame(np.empty_like(corr_data))
            annot_label = np.where(corr_data > 0.5, "*", "")
            length = len(corr_data)
            plt.figure(figsize=(10, 10))
            plt.gca()
            plt.axis()
            sns.heatmap(corr_data, annot=annot_label, fmt="", annot_kws={"color": "black", "size": 1 / length * 30})
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            plt.savefig(self.path + g1 + "-" + g2 + "/corr.png", dpi=500, bbox_inches="tight")
            plt.close()
            path_list.append(g1 + "-" + g2 + '/corr.png')
            corr_data.to_excel(self.path + g1 + "-" + g2 + "/correlational_matrix.xlsx")
        return path_list

    def radarplot(self):
        """
        绘制每组对比差异物质的雷达图
        返回每组对比差异物质的雷达图
        """
        contrast_list = self.contrast.split(" ")
        path_list = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            data = self.difference[1][i]
            data_colname = data.loc[:, ["NAME_EN"]].values.flatten().tolist()
            data = data.loc[:, ["logFC"]].values.flatten().tolist()
            if len(data) > 3:
                if len(data) > 10:
                    data1 = [round(i, 2) for i in data[0:10]]
                    minv = round(min(data1), 2)
                    maxv = round(max(data1), 2)
                    data_colname = data_colname[0:10]
                elif len(data) <= 10:
                    data1 = [round(i, 2) for i in data]
                    minv = round(min(data1), 2)
                    maxv = round(max(data1), 2)
                    data_colname = data_colname
            N = len(data_colname)
            data1 += data1[:1]
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            ax = plt.subplot(111, polar=True)
            plt.xticks(angles[:-1], data_colname, color="grey", size=8)
            ax.set_rlabel_position(0)  # 将y轴ticks标签偏转角度，如-22.5度
            # ax.set_rmax(maxv)
            min_lim = minv - (maxv - minv) / 5
            max_lim = maxv
            plt.ylim(min_lim, max_lim)
            plt.yticks([round(minv + (maxv - minv) / 5 * i, 2) for i in range(5)],
                       [str(round(minv + (maxv - minv) / 5 * i, 2)) for i in range(5)], color="red", size=5)
            plt.grid(True)
            ax.plot(angles, data1, linewidth=1, linestyle="solid")
            ax.fill(angles, data1, "b", alpha=0.1)
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            plt.savefig(self.path + g1 + "-" + g2 + "/radar.png", dpi=500, bbox_inches="tight")
            plt.close()
            path_list.append(g1 + "-" + g2 + '/radar.png')
        return path_list

    def mapping(self):
        """
        根据差异表中的KEGGID和FC值标注红蓝
        返回KEGG和pathway分析所需要列表
        """
        mean_data = self.mean_data
        contrast_list = self.contrast.split(" ")
        kegg_data_list = []
        pathway_data_list = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            file_name = g1 + "-" + g2
            dec = self.difference[1][i]
            dec1 = dec[["NAME_EN", "FC"]]
            dec1.loc[dec1["FC"] > 1, "FC"] = "red"
            dec1.loc[dec1["FC"] != "red", "FC"] = "blue"
            kegg_data = pd.merge(mean_data, dec1, on="NAME_EN", how="inner")
            kegg_data = kegg_data[kegg_data["KEGG_ID"].notna()]
            kegg_data = kegg_data[["KEGG_ID", "FC"]]
            # 有多个KEGG的id 拆分
            kegg_data = kegg_data['KEGG_ID'].str.split(';', expand=True).stack().reset_index(level=0).set_index('level_0').rename(columns={0:'KEGG_ID'}).join(kegg_data.drop('KEGG_ID', axis=1))
            kegg_data = kegg_data.drop_duplicates(["KEGG_ID"])
            kegg_data_list.append(kegg_data)
            pathway_data = kegg_data[["KEGG_ID"]]
            pathway_data_list.append(pathway_data)
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            kegg_data.to_csv(self.path + file_name + "/" + file_name + ".txt", sep="\t", index=False, header=None)
        return kegg_data_list, pathway_data_list

    def read_pickle(self, file):
        with open("C:/Users/Administrator/Desktop/GB_cloud/keggda/"+ self.species + "/" + file, "rb") as f:
            data = pickle.load(f)
        return data

    def kegg_analysis(self):
        """
        根据mapping的生成的kegglist和keggda的数据，统计通路和计算物质个数，抓取keggda相关物种的通路图
        返回每组对比差异物质的kegg结果表
        """
        contrast_list = self.contrast.split(" ")
        kegg_data_list = self.kegg_pathway_mapping_list[0]
        with open("C:/Users/Administrator/Desktop/GB_cloud/keggda/"+ self.species +"/pw_kegg_id.plk", "rb") as f:
            kegg_database = pickle.load(f)
        with open("C:/Users/Administrator/Desktop/GB_cloud/keggda/"+ self.species +"/pw_description.plk", "rb") as f:
            pw_description_database = pickle.load(f)
        kegg_results_list = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            file_name = g1 + "-" + g2
            input_data = ""
            kegg_data = kegg_data_list[i]
            pw_id = []
            hit_keggid_num = []
            hit_keggid = []
            total_keggid_num = []
            total_keggid = []
            pw_description = []
            for i in kegg_database:
                len_j = 0
                pw_hit_keggid = ""
                for j in kegg_data["KEGG_ID"]:
                    if j in kegg_database[i]:
                        pw_hit_keggid = pw_hit_keggid + j + ";"
                        len_j = len_j + 1
                if len_j > 0:
                    hit_keggid_num.append(len_j)
                    hit_keggid.append(pw_hit_keggid[:-1])
                    pw_id.append(i)
                    total_keggid_num_tmp = len(kegg_database[i])
                    total_keggid_tmp = ""
                    for k in kegg_database[i]:
                        total_keggid_tmp = total_keggid_tmp + k + ";"
                    total_keggid_tmp = total_keggid_tmp[:-1]
                    pw_description_tmp = pw_description_database[i]
                    total_keggid_num.append(total_keggid_num_tmp)
                    total_keggid.append(total_keggid_tmp)
                    pw_description.append(pw_description_tmp)
            kegg_res = list(zip(pw_id, pw_description, hit_keggid_num, hit_keggid, total_keggid_num, total_keggid))
            kegg_results = pd.DataFrame(kegg_res,
                                        columns=['Pathway', 'Description', '# compounds_num(dem)', "Compounds(dem)",
                                                 "# compounds(all)", "Compounds(all)"])
            kegg_results = kegg_results.loc[kegg_results["# compounds_num(dem)"]>=self.kegghitmin, :]
            kegg_results = kegg_results.sort_values(by="# compounds_num(dem)", ascending=False)
            kegg_results_list.append(kegg_results)
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            if not os.path.exists(self.path + g1 + "-" + g2 + "/maps"):
                os.makedirs(self.path + g1 + "-" + g2 + "/maps")
            kegg_results.to_excel(self.path+file_name+"/"+"kegg.xlsx", index=False)
            for i in pw_id:
                fig_input_file = "C:/Users/Administrator/Desktop/GB_cloud/keggda/"+ self.species +"/maps/" + i + ".png"
                picture = Image.open(fig_input_file)
                fig_output_file = self.path + g1 + "-" + g2 + "/maps/" + i + ".png"
                picture.save(fig_output_file)
        self.kegg_results_list = kegg_results_list
        return kegg_results_list

    def pathway_analysis(self):
        """
        根据mapping的生成的pathwaylist和keggda的数据，统计通路，计算物质个数和p值等
        生成每组对比差异物质的pathway结果表、气泡图和矩形树图
        """
        contrast_list = self.contrast.split(" ")
        pathway_data_list = self.kegg_pathway_mapping_list[1]
        kegg_results_list = self.kegg_results_list
        cmp_total_count = self.read_pickle("species_total_num.plk")
        uniq_count = cmp_total_count[self.species]
        # pw_cmp_count = self.read_pickle("pw_cmp_count.plk")
        pw_description = self.read_pickle("pw_description.plk")
        self.pathway_results_list = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            file_name = g1 + "-" + g2
            pw_data = pathway_data_list[i]
            kegg_data = kegg_results_list[i]
            kegg_data = kegg_data.reset_index(drop=True)
            q_size = pw_data.shape[0]
            p_list = []
            pw_cmp_num_list = []
            hit_num_list = []
            pathway_id_list = []
            pw_description_list = []
            for j in range(kegg_data.shape[0]):
                pathway_id = kegg_data["Pathway"][j]
                hit_num = int(kegg_data["# compounds_num(dem)"][j])
                hit_num_1 = hit_num - 1
                pw_cmp_num = kegg_data["# compounds(all)"][j]
                p_value = hypergeom.sf(hit_num_1, uniq_count, pw_cmp_num, q_size)
                pw_description_list.append(re.split(" - ", pw_description[pathway_id])[0])  # pw_description[pathway_id]
                pathway_id_list.append(pathway_id)
                hit_num_list.append(hit_num)
                pw_cmp_num_list.append(pw_cmp_num)
                p_list.append(p_value)
            p_ln = -np.log(p_list)
            p_fdr = self.p_adjust(p_list, method="fdr")
            p_holm = self.p_adjust(p_list, method="holm")
            pathway_results = list(zip(pw_description_list, pw_cmp_num_list, hit_num_list, p_list,
                                       p_ln, p_holm, p_fdr))
            pathway_results = pd.DataFrame(pathway_results, columns=["Pathway", "Total", "Hits", "Raw p",
                                                                     "-ln(p)", "Holm adjust", "FDR"])
            pathway_results = pd.concat([pathway_results, kegg_data[["Compounds(dem)", "Compounds(all)"]]], axis=1)
            self.pathway_results_list.append(pathway_results)
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            pathway_results.to_excel(self.path + g1 + "-" + g2 + "/Pathway_Analysis.xlsx", index=False)

            x = pathway_results["Hits"]
            y = pathway_results["-ln(p)"]
            fig, ax = plt.subplots()
            scatter = ax.scatter(x, y, c=y, s=10 * x, cmap="PiYG") #"Blues"
            axins = inset_axes(ax,
                               width="20%",  # width = 5% of parent_bbox width
                               height="5%",  # height : 50%
                               loc="lower left",
                               bbox_to_anchor=(1.02, 0.35, 1, 1),
                               bbox_transform=ax.transAxes,
                               borderpad=0,
                               )
            fig.colorbar(scatter, cax=axins, orientation="horizontal")
            handles, labels = scatter.legend_elements(prop="sizes", alpha=0.6)
            legend2 = ax.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc="lower left", title="Sizes")
            # plt.tight_layout()
            plt.savefig(self.path + g1 + "-" + g2 + '/bubble.png', dpi=300, bbox_inches="tight")
            plt.close()

            plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
            plt.rcParams['axes.unicode_minus'] = False
            data = pathway_results
            data1 = data.sort_values(by="-ln(p)", ascending=False)
            x = data1["Hits"]
            y = data1["-ln(p)"]
            labels = data1["Pathway"]
            labels[5:] = ""
            min_value = min(y)
            max_value = max(y)
            label_color = (y - min_value) / (max_value - min_value)
            # plot
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            plot = squarify.plot(sizes=x,  # 方块面积大小
                                 label=labels,  # 指定标签
                                 color=plt.cm.summer(label_color),  # 指定自定义颜色
                                 alpha=0.8,  # 指定透明度
                                 # value=x,  # 添加数值标签
                                 edgecolor='white',  # 设置边界框
                                 linewidth=0.1,  # 设置边框宽度
                                 text_kwargs={"fontsize": 10}
                                 )
            # 设置标签大小
            plt.rc('font', size=4)  # 无效！
            # 去除坐标轴
            ax.axis('off')
            # 去除上边框和右边框刻度
            ax.tick_params(top='off', right='off')
            plt.savefig(self.path + g1 + "-" + g2 + '/treemap.png', dpi=300, bbox_inches="tight")
            plt.close()

    def kegg_Enrichment_bubble(self, shownum=15):
        contrast_list = self.contrast.split(" ")
        kegg_results_list = self.kegg_results_list
        pathway_results_list = self.pathway_results_list
        result = []

        ## 需要kegg的表格和pathway中的p
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            ## 读取所分析kegg分析的结果
            keggresult = kegg_results_list[i]
            keggresult['Description2'] = [i[:i.rfind(' - ')] for i in keggresult.loc[:, 'Description'].values.tolist()]

            ## 读取所分析pathway分析的结果
            pathwayresult = pd.DataFrame(pathway_results_list[i])
            pathwayresult2 = pathwayresult.loc[:, ['Pathway', 'Raw p']]

            result_merge = pd.merge(keggresult, pathwayresult2, left_on='Description2', right_on='Pathway',
                                    how="inner")
            result_merge['RichFactor']=result_merge.loc[:,'# compounds_num(dem)']/result_merge.loc[:,'# compounds(all)']
            result_merge_orderp = result_merge.sort_values(by='Raw p', axis=0)  ##按照p值进行排序
            orderp15 = result_merge_orderp.head(min(result_merge_orderp.shape[0], shownum))

            orderp15.reset_index(inplace=True)
            orderp15 = orderp15.loc[:,
                       ["Pathway_y", "RichFactor", '# compounds_num(dem)','Raw p']]
            res = list(zip(orderp15["Pathway_y"], orderp15["RichFactor"], orderp15["# compounds_num(dem)"], orderp15["Raw p"]))
            result.append(res)
            print(result)
        return result

    def kegg_dascore(self, shownum=15 , classdatabase="C:/Users/Administrator/Desktop/GB_cloud/KEGG_Class.xlsx"):

        contrast_list = self.contrast.split(" ")
        kegg_results_list = self.kegg_results_list
        pathway_results_list = self.pathway_results_list

        # biotree_database = self.biotree_database
        # biotree_database = biotree_database.rename(columns={"MS2 name": "MS2.name"})
        # biotree_database1 = biotree_database[["MS2.name", "KEGG ID"]]

        ## 需要kegg的表格和pathway中的p
        ## 读取class数据库
        kegg_class = pd.read_excel(classdatabase, converters={'kegg ID': str})
        kegg_class.rename(columns={'kegg ID': 'kegg_ID',
                                   'ms2 description': 'ms2_description'
                                   }, inplace=True)  # 名字不要有空格
        kegg_class2 = kegg_class.loc[:, ['kegg_ID', 'ms2_description']]
        kegg_class2['kegg_ID'] = kegg_class2['kegg_ID'].astype("str")

        res_das_list = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            file_name = g1 + "-" + g2
            ## 读取所分析kegg分析的结果
            keggresult = kegg_results_list[i]
            keggresult['Description2'] = [i[:i.rfind(' - ')] for i in keggresult.loc[:, 'Description'].values.tolist()]
            keggresult['kegg_ID'] = [i.replace(self.species, '') for i in keggresult.loc[:, 'Pathway'].values.tolist()]
            # keggresult['kegg_ID'] = [i.replace('mmu', '') for i in keggresult.loc[:, 'Pathway'].values.tolist()]
            keggresult['kegg_ID'] = keggresult['kegg_ID'].astype("str")
            keggresult_merge = pd.merge(keggresult, kegg_class2, on='kegg_ID', how="inner")

            ## 读取所分析pathway分析的
            pathwayresult = pd.DataFrame(pathway_results_list[i])
            pathwayresult2 = pathwayresult.loc[:, ['Pathway', 'Raw p']]
            result_merge = pd.merge(keggresult_merge, pathwayresult2,
                                    left_on='Description2', right_on='Pathway',
                                    how="inner")
            ###DA SCORE PART,导入颜色信息文件
            dec = self.difference[1][i]
            dec1 = dec[["NAME_EN", "FC", "KEGG_ID"]]
            dec1.loc[dec1["FC"] > 1, "FC"] = "red"
            dec1.loc[dec1["FC"] != "red", "FC"] = "blue"

            kegg_data = dec1[dec1["KEGG_ID"].notna()]
            kegg_data = kegg_data[["KEGG_ID", "FC"]]
            kegg_data = kegg_data.drop_duplicates(["KEGG_ID"])
            kegg_data2 = \
            kegg_data.loc[:, ['KEGG_ID', 'FC']].set_index("KEGG_ID").to_dict(orient='dict')["FC"]
            # print(kegg_data2)
            # kegg_data.to_csv(self.path + file_name + "/" + file_name + "2.txt", sep="\t", index=False, header=None)

            for z in range(result_merge.shape[0]):
                # print(result_merge['Compounds(dem)'][z])
                tempz = result_merge['Compounds(dem)'][z]
                tempdict = {}
                for x in tempz.split(";"):
                    if x in kegg_data2:
                        tempdict[x] = kegg_data2[x]
                result_merge.loc[z, 'up'] = list(tempdict.values()).count("red")
                result_merge.loc[z, 'down'] = list(tempdict.values()).count("blue")
            result_merge.loc[:, 'DAscore'] = (result_merge.loc[:, 'up'] - result_merge.loc[:,'down']) / result_merge.loc[:,'# compounds_num(dem)']
            result_merge_orderp = result_merge.sort_values(by='Raw p', axis=0)  ##按照p值进行升序
            orderp15 = result_merge_orderp.head(min(result_merge_orderp.shape[0], shownum))
            orderp15 = orderp15.reset_index()

            das_list = []
            for d in range(orderp15.shape[0]):
                if (orderp15.loc[d, "Raw p"] >= 0 and orderp15.loc[d, "Raw p"] < 0.001):
                    star = int(3)
                elif (orderp15.loc[d, "Raw p"] >= 0.001 and orderp15.loc[d, "Raw p"] < 0.01):
                    star = int(2)
                elif (orderp15.loc[d, "Raw p"] >= 0.01 and orderp15.loc[d, "Raw p"] < 0.05):
                    star = int(1)
                else:
                    star = int(0)
                das_dict_temp = {"Pathway": orderp15.loc[d, "Pathway_y"],
                                 "score": orderp15.loc[d, "DAscore"],
                                 "count": orderp15.loc[d, "# compounds_num(dem)"],
                                 "Raw p": orderp15.loc[d, "Raw p"],
                                 "ms2_description": orderp15.loc[d, "ms2_description"],  ### colour class
                                 "star": star
                                 }
                das_list.append(das_dict_temp)
            res_das_list.append(das_list)
            print(res_das_list)
        return res_das_list

    def chordplot(self):
        """
        根据每组对比的差异表注释进行物质分类，计算物质之间的相关性和P值
        返回node和linklist
        """
        contrast_list = self.contrast.split(" ")
        path_list = []
        data_node_list = []
        data_link_list = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            data = self.difference[1][i]
            # mapping_tmp = data[["NAME_EN", "CLASS_EN"]]
            data_res = data.sort_values(by="Pvalue", ascending=True)
            data_res1 = data_res.iloc[:min(50, data_res.shape[0]), :]
            # data_res1 = pd.merge(data_res1, mapping_tmp, on="NAME_EN", how="left")
            data_res1_name = data_res1["NAME_EN"]
            data_res1_name = data_res1_name.reset_index(drop=True)
            data_tmp = data_res1.drop(['FC', 'Pvalue', 'logFC', "vip", "CLASS_EN"], axis=1)
            data_tmp1 = data_tmp.iloc[:, 12:]
            data_tmp2 = data_tmp1
            data_tmp2.index = data_tmp["NAME_EN"]
            n1 = n2 = data_tmp2.shape[0]
            id = np.array(range(n1))
            data_node_tmp1 = pd.DataFrame([np.array(data_res1["NAME_EN"]), np.array(data_res1["logFC"]), id, np.array(data_res1["CLASS_EN"])]).T
            data_node_tmp1.columns = ["NAME_EN", "logFC", "id", "CLASS_EN"]
            data_node_tmp1.loc[data_node_tmp1["CLASS_EN"].isna(), "CLASS_EN"] = "others"
            data_node_tmp1 = data_node_tmp1.sort_values(by="CLASS_EN", ascending=True)
            category_tmp = data_node_tmp1["CLASS_EN"]
            category_tmp1 = list(set(category_tmp))
            category_tmp1.sort()
            category_tmp2 = []
            for i in category_tmp:
                for j in range(len(category_tmp1)):
                    if category_tmp1[j] == i:
                        category_tmp2.append(j)
            data_node_tmp1["category"] = np.array(category_tmp2)
            data_node = data_node_tmp1
            data_node["NAME_EN"] = data_node["NAME_EN"].str.lower()
            data_node_list.append(data_node)
            data_p_corr = pd.DataFrame(columns=range(n1), index=range(n2))
            data_p_corr.columns = data_p_corr.index = data_tmp["NAME_EN"]
            corr_res = data_p_corr
            corr_res1 = []
            for i in range(n1):
                for j in range(n2):
                    x = data_tmp2.iloc[i, :]
                    y = data_tmp2.iloc[j, :]
                    corr_res.iloc[i, j] = corr_tmp = spearmanr(x, y)[0]
                    a = data_res1_name[i]
                    b = data_res1_name[j]
                    corr_res1.append([a, b, corr_tmp])
            l1 = np.tile(data_res1_name, len(data_res1_name))
            id1 = np.tile(id, len(id))
            l2 = np.repeat(data_res1_name, len(data_res1_name))
            id2 = np.repeat(id, len(id))
            corr_res_array = np.array(corr_res).flatten()
            data_link = pd.DataFrame([l1, id1, l2, id2, corr_res_array]).T
            data_link.columns = ["from", "id1", "to", "id2", "value"]
            data_link1 = data_link.loc[data_link["from"]!=data_link["to"], :]
            if data_tmp.shape[0] >= 50:
                data_link2 = data_link1.loc[(data_link1["value"]<-0.9) | (data_link1["value"]>0.9), :]
                if data_link2.shape[0] < 50:
                    data_link2 = data_link1.loc[(data_link1["value"]<-0.8) | (data_link1["value"]>0.8), :]
                else:
                    data_link2 = data_link2
            elif data_tmp.shape[0] >= 10 and data_tmp.shape[0] < 50:
                data_link2 = data_link1.loc[(data_link1["value"]<-0.8) | (data_link1["value"]>0.8), :]
                if data_link2.shape[0] < 50:
                    data_link2 = data_link1.loc[(data_link1["value"] < -0.6) | (data_link1["value"] > 0.6), :]
                else:
                    data_link2 = data_link2
            else:
                data_link2 = data_link1
            data_link2["from"] = data_link2["from"].str.lower()
            data_link2["to"] = data_link2["to"].str.lower()
            data_link3 = data_link2[["from", "to", "value"]]
            if not os.path.exists(self.path + g1 + "-" + g2):
                os.makedirs(self.path + g1 + "-" + g2)
            data_link3.to_excel(self.path + g1 + "-" + g2 + "/Chordplot_Analysis.xlsx", index=False)
            data_link_list.append(data_link1)
        return data_node_list, data_link_list

    def piechart(self):
        # 分类信息对应初次级代谢表格
        class_mapping = pd.read_excel("C:/Users/Administrator/Desktop/GB_cloud/class_mapping.xlsx")
        meandata = self.mean_data
        # 统计频次和百分比
        subclass_count = pd.DataFrame(meandata["CLASS_CH"].value_counts())
        subclass_count['subclass'] = subclass_count.index
        subclass_count.columns = ['count', 'subclass']
        class_data = pd.merge(subclass_count, class_mapping, how='inner')
        class_data.sort_values(by=['class'], axis=0, inplace=True)
        class_data = class_data.reset_index(drop=True)
        class_data['percent(%)'] = (class_data['count'] / class_data['count'].sum()) * 100
        order = ['class', 'subclass', 'count', 'percent(%)']
        class_data = class_data[order]
        class_data.to_excel(self.path + "/Classification.xlsx", index=False)

        primary = [int(class_data.loc[0, 'count']), int(class_data.loc[1:4, 'count'].sum()),
                   int(class_data.loc[5:10, 'count'].sum())]
        secondary = list(class_data['count'])
        secondary = [int(x) for x in secondary]
        third = [1]
        fig, ax = plt.subplots()
        label1 = list(class_data['class'].drop_duplicates())
        label2 = list(class_data['subclass'].drop_duplicates())
        labels = label1 + label2
        color1 = plt.get_cmap("tab20")([0, 12, 16])
        color2 = plt.get_cmap("tab20")(range(11))

        # 外层
        ax.pie(primary,
               radius=1.2, autopct='%1.1f%%', pctdistance=0.9,
               wedgeprops={'width': 0.3, 'edgecolor': 'w'},
               colors=color1
               )
        # 第二层
        ax.pie(secondary,
               radius=1, autopct='%1.1f%%', pctdistance=0.75,
               wedgeprops={'width': 0.3, 'edgecolor': 'w'},
               colors=color2
               )
        # 内层
        ax.pie(third, radius=0.4, colors='w')

        ax.set(aspect="equal", title='Classification for all metabolites')
        plt.legend(labels,
                   fontsize=9,
                   title='Class',
                   bbox_to_anchor=(1, 1), loc='best', borderaxespad=0.)
        plt.savefig( self.path + "/piechart.png", dpi=500, bbox_inches="tight")
        plt.close()
        return class_data

    def rocplot(self):
        contrast_list = self.contrast.split(" ")
        print(self)
        path_list = []
        df_roc = []
        for i in range(len(contrast_list)):
            g1 = contrast_list[i].split(":")[0]
            g2 = contrast_list[i].split(":")[1]
            path_list = []
            df = self.difference[0][i]
            df = df.dropna(axis=0, how='any', thresh=None, subset=['NAME_EN'], inplace=False)
            df_deg = pd.DataFrame()
            df_deg = df[['NAME_EN', 'logFC']]
            df_deg = df_deg.astype({'logFC': 'float'})
            df_deg_up = df_deg[(df_deg['logFC'] >= 0)]
            df_deg_down = df_deg[(df_deg['logFC'] < 0)]
            df_input = df.drop(list(df.columns[0:self.descNum]), axis=1)
            mean_names_bool = df_input.columns.str.contains("Mean")
            mean_names = df_input.columns[mean_names_bool]
            df_input2 = df_input[mean_names]
            df_sub = df_input.drop(list(df_input2.columns[0:2]), axis=1)
            df_sub = df_sub.drop(columns=['logFC', 'vip', 'FC', 'Pvalue'], axis=1)
            col = 0
            Gseq = -1
            for j in self.sample.split(" "):
                Gseq = Gseq + 1
                col = col + int(j)
                df_sub_subGroup = df_sub.loc[:, df_sub.columns[(col - int(j)):col]]
                for n in range(len(df_sub_subGroup.columns)):  # 把每组的样本名前加上所属的组的信息，方便下面对比时进行索引
                    new_name = self.group.split(" ")[Gseq]
                    df_sub = df_sub.rename(columns={df_sub_subGroup.columns[n]: new_name})
            df_sub.index = df["NAME_EN"]
            df_sub.loc[len(df_sub)] = 1
            df_sub.loc[len(df_sub) - 1, g1] = 0
            df_sub = df_sub.T
            df_sub = df_sub.rename(columns={df_sub.columns[len(df_sub.columns) - 1]: 'classID'})
            rocdata2 = df_sub.melt(id_vars=['classID'])
            rocdata2 = rocdata2.astype({'value': 'float'})
            rocdata2 = rocdata2.astype({'classID': 'float'})
            rocdata2 = rocdata2.astype({'classID': 'int'})
            # y_test = rocdata2.loc[0:0.5 * len(rocdata2)]
            # y_score = rocdata2.loc[0.5 * len(rocdata2) + 1:len(rocdata2)]
            # y_label = y_label.tolist
            # y_pre= y_pre.tolist
            ms2name = list(set(rocdata2['NAME_EN']))
            df_roc_list = []
            for m in range(len(ms2name)):
                # print(ms2name[m])
                rocdata2_m = rocdata2.loc[rocdata2['NAME_EN'] == ms2name[m]]
                y_label = rocdata2_m['classID'].values
                y_pre = rocdata2_m['value'].values
                fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=1)
                # for i, value in enumerate(thersholds):
                #     # print(("%f %f %f" % (fpr[i], tpr[i], value)))
                #     # print("%f %f" % (fpr[i], tpr[i]))
                # print(len(list(zip(fpr,tpr))))

                roc_auc = auc(fpr, tpr)
                data2 = {
                    "name": ms2name[m],
                    "auc": roc_auc,
                    "value": list(zip(fpr, tpr))
                }
                df_roc_list.append(data2)
            df_roc.append(df_roc_list)
        return df_roc


if __name__ == '__main__':
    A = Public(path="C:/Users/Administrator/Desktop/GB_cloud/", file_name="AQ-BYHN20220111-QTRAP-ZP-data.xlsx", descNum=13,
               sample="3 3 3 3 3", fillMethod="min", filterMethod="rsd", normMethod="neibiao", group="Ⅰ Ⅱ Ⅲ Ⅳ Ⅴ")
    A.normalize()
    B = ModelAnalysis(path="C:/Users/Administrator/Desktop/GB_cloud/", species="osa", descNum=13, sample="3 3 3 3 3",
                      group="Ⅰ Ⅱ Ⅲ Ⅳ Ⅴ", contrast="Ⅰ:Ⅱ", pvalue=0.05, FC_up=1, FC_down=1, # Ⅰ:Ⅲ Ⅰ:Ⅳ Ⅰ:Ⅴ
                      mean_data=pd.read_excel("C:/Users/Administrator/Desktop/GB_cloud/norm.xlsx"), vip_cut=1, pca_scaling="Par", opls_scaling="UV",
                      pca_log=True, opls_log=True, ci=0.95, kegghitmin=2)
    B.compare_filter_dec()
    B.pca_model()
    B.total_pca()
    B.opls()
    B.volcanoplot()
    # B.radarplot()
    # B.heatmap()
    # B.corrplot()
    # B.mapping()
    # B.kegg_analysis()
    # B.pathway_analysis()
    # B.chordplot()
    # B.heatmapAll()
    # B.kegg_dascore()
    # B.kegg_Enrichment_bubble()
    B.rocplot()
    # B.boxplot()
    # B.zscorePlot()
    # B.get_kmeans()
    # B.matchstick(shownum=15)
    # B.piechart()

