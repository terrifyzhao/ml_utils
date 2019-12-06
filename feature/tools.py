#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/12/4
# @Author : 胡茂海
# @Site   : 
# @File   : tools.py

"""
贝叶斯块分箱算法
=================
这是一个自动分箱算法,基于贝叶斯块算法.来自于博客
https://jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/
"""
import numpy as np
from sklearn.utils.multiclass import type_of_target

def bayesian_blocks(t):
    # copy and sort the array
    t = [x[0] for x in t]
    print(t)
    t = np.sort(t)
    N = t.size

    # create length-(N + 1) array of cell edges
    edges = np.concatenate([t[:1],
                            0.5 * (t[1:] + t[:-1]),
                            t[-1:]])
    block_length = t[-1] - edges

    # arrays needed for the iteration
    nn_vec = np.ones(N)
    best = np.zeros(N, dtype=float)
    last = np.zeros(N, dtype=int)

    for K in range(N):
        width = block_length[:K + 1] - block_length[K + 1]
        count_vec = np.cumsum(nn_vec[:K + 1][::-1])[::-1]

        fit_vec = count_vec * (np.log(count_vec) - np.log(width))
        fit_vec -= 4
        fit_vec[1:] += best[:K]

        i_max = np.argmax(fit_vec)
        last[K] = i_max
        best[K] = fit_vec[i_max]

    change_points = np.zeros(N, dtype=int)
    i_cp = N
    ind = N
    while True:
        i_cp -= 1
        change_points[i_cp] = ind
        if ind == 0:
            break
        ind = last[ind - 1]
    change_points = change_points[i_cp:]

    return edges[change_points]


class WeightOfEvidence:
    """计算某一离散特征的woe值
    Attributes:
        woe (Dict): - 训练好的证据权重
        iv (Float): - 训练的离散特征的信息量
    """

    def __init__(self):
        self.woe = None
        self.iv = None

    def _posibility(self, x, tag, event=1):
        """计算触发概率
        Parameters:
            x (Sequence): - 离散特征序列
            tag (Sequence): - 用于训练的标签序列
            event (any): - True指代的触发事件
        Returns:
            Dict[str,Tuple[rate_T, rate_F]]: - 训练好后的好坏触发概率
        """
        if type_of_target(tag) not in ['binary']:
            raise AttributeError("tag must be a binary array")
        if type_of_target(x) in ['continuous']:
            raise AttributeError("input array must not continuous")
        tag = np.array(tag)
        x = np.array(x)
        event_total = (tag == event).sum()
        non_event_total = tag.shape[-1] - event_total
        x_labels = np.unique(x)
        pos_dic = {}
        for x1 in x_labels:
            y1 = tag[np.where(x == x1)[0]]
            event_count = (y1 == event).sum()
            non_event_count = y1.shape[-1] - event_count
            rate_event = 1.0 * event_count / event_total
            rate_non_event = 1.0 * non_event_count / non_event_total
            pos_dic[x1] = (rate_event, rate_non_event)
        return pos_dic

    def fit(self, x, y, *, event=1, woe_min=-20, woe_max=20):
        """训练对单独一项自变量(列,特征)的woe值.
        Parameters:
            x (Sequence): - 离散特征序列
            y (Sequence): - 用于训练的标签序列
            event (any): - True指代的触发事件
            woe_min (munber): - woe的最小值,默认值为-20
            woe_max (munber): - woe的最大值,默认值为20
        """
        woe_dict = {}
        iv = 0
        pos_dic = self._posibility(x=x, tag=y, event=event)
        for l, (rate_event, rate_non_event) in pos_dic.items():
            if rate_event == 0:
                woe1 = woe_min
            elif rate_non_event == 0:
                woe1 = woe_max
            else:
                woe1 = np.log(rate_event / rate_non_event)  # np.log就是ln
            iv += (rate_event - rate_non_event) * woe1
            woe_dict[str(l)] = woe1
        self.woe = woe_dict
        self.iv = iv

    def transform(self, X):
        """将离散特征序列转换为woe值组成的序列
        Parameters:
            X (Sequence): - 离散特征序列
        Returns:
            numpy.array: - 替换特征序列枚举值为woe对应数值后的序列
        """
        return np.array([self.woe.get(i) for i in X])


if __name__ == '__main__':
    x = np.array([1,2,3,4,5,30,15,20,36,6,7,8,9,2,3,4,5,6,71,8,9,2,3,4,5,6,7,8,9,2,3,4,5,6,7,8,9,30])
    print(bayesian_blocks(x))