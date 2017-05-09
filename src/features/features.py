# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:17:30 2017

@author: nberliner
"""
import numpy as np

from utils.NestDistance import NestDistance



def add_features(df_features, radius):
    """
    Top-level function to add all desined features to the feature DataFrame
    containing the time-series information.
    """
    
    # A bit inefficient to re-compute but convenient
    nest_distance = NestDistance()
    
    df_features = add_proximity_nestCount(df_features, radius, nest_distance)
    
    return(df_features)


def add_proximity_nestCount(df_features, radius, nest_distance):
    """
    This will add the median change of all nests found within radius of each
    location per species. Note that only nests of the same species are considered.
    """
    # Need to make sure the DataFrame is sorted
    df_features.sort_index(inplace=True)
        
    # Extract only the time stop column names
    ts_step = [ item for item in df_features.columns if len(item)==2 and item[0] == 't' ]
    ts_step = ts_step[-1] # only take the values of the last year
    
    values = list()
    siteCount = list()
    
    # Iterate over every site_id, species and year
    for site_id, species, year in df_features.index:
        
        neighbour_sites = nest_distance.query(site_id, radius)
        val = list()
        count = 0 # it may happen that at a given site
        for nn in neighbour_sites:
            try:
                val.append(df_features.loc[(nn, species, year),ts_step])
                count += 1
            except KeyError:
                pass
        
        # Compute the val to store
        val = np.array(val)
        val = val[np.isfinite(val)]
        if val.shape == (0,):
            val = [0, ]
        val = np.median(val)
        
        values.append(val)
        siteCount.append(count)
    
    df_features = df_features.assign(proximityNestCountChange = values)
    df_features = df_features.assign(siteCount = siteCount)
    
    return(df_features)