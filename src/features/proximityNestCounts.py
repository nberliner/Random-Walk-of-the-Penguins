# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:59:29 2017

@author: nberliner
"""
import numpy as np
import pandas as pd


def proximity_nest_counts(df_nestCount, radius, nest_distance, site_count_full=True):
    """
    Assemble a dataframe that has the same format as the nestCount dataframe,
    but holds the number of nest found in its proximity. If site_count_full is
    True, this dataframe will be expanded to have the same shape as nestCount as
    well. This information can then be used enrich the time series input.
    """
    # Create a copy to store the values in
    df_count_in_radius = df_nestCount.copy()
    df_site_count = pd.DataFrame(np.zeros(df_nestCount.shape[0]), index=df_nestCount.index, columns=['site_count'])
    
    for site_id, penguin_species in list(df_count_in_radius.index):
        # Query the sites within radius
        neighbor_sites = nest_distance.query(site_id, radius)
        
        # Assemble the indices for the rows that we need to sum
        allowedIndexes = list(df_nestCount.index)
        idx = [ (site, penguin_species) for site in neighbor_sites if (site, penguin_species) in allowedIndexes ]
        
        # Select the relevant rows, sum along the first axis and set the values in the new DataFrame
        # if idx is an empty list, the result of sum() will be all zeros which is what we want here.
        df_count_in_radius.loc[(site_id, penguin_species),:] = df_nestCount.loc[idx,:].sum(axis=0)
        
        # Set the nest count in the proximity
        df_site_count.loc[(site_id, penguin_species),:] = len(idx)
    
    
    # Expand the info to the full DataFrame (useful for later alignment)
    if site_count_full:
        df_site_count_ = df_nestCount.copy()
        df_site_count_.iloc[:,:] = 0
        
        for i, (site_id, penguin_species) in enumerate(df_site_count_.index):
            df_site_count_.loc[site_id] = df_site_count.loc[(site_id, penguin_species)]['site_count']
            
                
        df_site_count = df_site_count_
        
    return(df_count_in_radius, df_site_count)
