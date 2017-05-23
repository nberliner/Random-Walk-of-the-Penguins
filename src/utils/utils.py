# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:14:51 2017

@author: nberliner
"""


def get_ts_steps(df):
    ts_steps = [ item for item in df.columns if len(item)==2 and item[0] == 't' ]
    return(ts_steps)
    
    
def antarcticSeason(time):
    year, month = time.year, time.month
    if month >= 11: # i.e. November
        year += 1
    return(year)