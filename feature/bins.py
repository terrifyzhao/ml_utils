#!/usr/bin/env python
# encoding:utf-8
# @Time   : 2019/12/9
# @Author : 胡茂海
# @Site   : 
# @File   : bins.py
import  pandas as pd
import numpy as np

def get_bins(y,x,drop_ratio=1.0,n=3):
    df1 = pd.DataFrame({'x':x,'y':y})
    justmiss = df1[['x','y']][df1.x.isnull()]
    notmiss = df1[['x','y']][df1.x.notnull()]
    bin_values=[]
    if n is None:
        d1 = pd.DataFrame({'x':notmiss.x,'y':notmiss.y,'Bucket':notmiss.x})
    else:
        x_uniq = notmiss.x.drop_duplicates().get_values()
        if len(x_uniq)<=n:
            bin_values=list(x_uniq)
            bin_values.sort()
        else:
            x_series = sorted(notmiss.x.get_values())
            x_cnt = len(x_series)
            bin_ration = np.linspace(1.0/n,1,n)
            bin_values = list(set([x_series[int(ratio*x_cnt)-1] for ratio in bin_ration]))
            bin_values.sort()
            if x_series[0] < bin_values[0]:
                bin_values.insert(0,x_series[0])
        if len(bin_values) ==1:
            bin_values.insert(0,bin_values[0] -1)
        d1 = pd.DataFrame({'x':notmiss.x,'y':notmiss.y,'Bucket':pd.cut(notmiss.x,bin_values,precision=8,include_lowest=True)})
    
    d2 = d1.groupby('Bucket',as_index=True)
    d3 = pd.DataFrame({},index=[])
    d3['MIN_VALUE']=d2.min().x
    d3['MAX_VALUE']=d2.max().x
    d3['COUNT'] = d2.count().y
    d3['EVENT'] = d2.sum().y
    d3['NONEVENT'] = d2.count().y - d2.sum().y
    # d3 = d2.reset_index(drop = True)
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUES':np.nan},index =[0])
        d4['MAX_VALUE'] =np.nan
        d4['EVENT'] = justmiss.sum().y
        d4['NONEVENT'] = justmiss.count().y - justmiss.sum().y
        d3 = d3.append(d4,ignore_index = True)

    total_event = d3.sum().EVENT
    total_no_event = d3.sum().NONEVENT
    d3['EVENT_RATE'] = d3.EVENT /d3.COUNT
    d3['NON_EVENT_RATE'] = d3.NONEVENT / d3.COUNT
    d3['DIST_EVENT'] = d3.EVENT / total_event
    d3['DIST_NON_EVENT'] = d3.NONEVENT / total_no_event

    if drop_ratio < 1.0:
        d3['DIST_EVENT1'] =d3['EVENT'].apply(lambda x:1.0/ total_event if x==0 else x / total_event)
        d3['DIST_NON_EVENT'] = d3['NONEVENT'].apply(lambda x:1.0/ total_no_event if x==0 else x/ total_no_event)
        d3['WOE'] = np.log(d3.DIST_EVENT1 / d3.DIST_NON_EVENT1)
        d3['IV'] = (d3.DIST_EVENT1 - d3.DIST_NON_EVENT1) * d3.WOE
    else:
        d3['WOE'] = np.log(d3.DIST_EVENT / d3.DIST_NON_EVENT)
        d3['IV'] = (d3.DIST_EVENT - d3.DIST_NON_EVENT) * np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)

    d3['VAR_NAME'] ='VAR'
    d3 = d3.replace([np.inf,-np.inf],0)
    d3.IV = d3.IV.sum()
    return bin_values

if __name__ == '__main__':
    data = pd.read_csv('titanic.csv')
    y = data['Survived'].values
    x = data['Age'].values
    print(get_bins(y, x, drop_ratio=1.0, n=3))