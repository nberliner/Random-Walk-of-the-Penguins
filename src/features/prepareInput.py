# -*- coding: utf-8 -*-
"""
Created on Wed May  3 15:05:49 2017

@author: nberliner
"""

import numpy as np
import pandas as pd


def assemble_timeseries_input(df, nan_mask, size):
    """ 
    Convert the input data into chunks of timeseries
    """
    # Extract the values from the DataFrame
    arr = df.values
    
    # Assemble the indices for a single row. This will allow to split a one 
    # dimensional array (i.e. one row) into the desired time series chunks.
    idx_x = [ np.arange(i,i+size) for i in range(arr.shape[1]-size) ]
    idx_y = [ i+size for i in range(arr.shape[1]-size) ]
    indices = [ i+size-1 for i in range(arr.shape[1]-size) ]
    
    # Split the whole array (column wise). The indices created above can be
    # used to split the whole DataFrame (the individual rows will be split in
    # the next step)
    x = [ arr[:,idx] for idx in idx_x ]
    y = [ arr[:,idx] for idx in idx_y ]
    columns = [ list(df.columns)[idx] for idx in indices ]
    
    # Split the rows. Above, we only split the columns
    x = np.array([ i.flatten() for item in x for i in np.split(item, item.shape[0], axis=0) ])
    y = np.array([ i for item in y for i in np.split(item, item.shape[0], axis=0) ])
    
    # The time series is split, and here the "ordering" each subseries is recorded.
    # These indices can be used to assemble auxiliary data matching the same order.
    ordering = [ (index, col) for col in columns for index in list(df.index) ]

    # Extract the site_id, the penguin species and the year
    site_id = [ index[0] for index,col in ordering ]
    species = [ index[1] for index,col in ordering ]
    year = [ col for index,col in ordering ]
    
    # Assemble the missing flag
    nan = [ nan_mask.loc[index,str(int(col)+1)] for index,col in ordering ]
    
    # Put everything into a new DataFrame
    values = {'site_id': site_id, 'species': species, 'year': year,
              'y_true': y.flatten(), 'inferred_y_true': nan}
    columns = ['site_id', 'species', 'year', 'y_true', 'inferred_y_true']
    for i in range(size):
        key = "t%s" %i
        values[key] = x[:,i]
        columns.append(key)
    
    df_features = pd.DataFrame(values, columns=columns)
    df_features.set_index(['site_id', 'species', 'year'], inplace=True)
    
    return(df_features)
    


def add_feature(df_features, df_addition, columnName):
    """
    Add a new feature to the df_feature DataFrame (output of assemble_timeseries_input()).
    df_addition must be a dataframe with the same shape as the nest counts. The index
    of df_features will be used as indexer into the df_addition DataFrame.
    """
    values = list()
    for (site_id, species, year) in df_features.index:
        values.append(df_addition.loc[(site_id, species),year])
        
    df_features = df_features.assign(**{columnName: values}) # http://stackoverflow.com/a/41759638/1922650
        
    return(df_features)