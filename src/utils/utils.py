# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:14:51 2017

@author: nberliner
"""


def get_ts_steps(df):
    ts_steps = [ item for item in df.columns if len(item)==2 and item[0] == 't' ]
    return(ts_steps)