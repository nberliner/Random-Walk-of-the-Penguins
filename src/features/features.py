# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:17:30 2017

@author: nberliner
"""
import numpy as np
import pandas as pd

from features.seaIce import get_seaIce
from features.krillbase import KrillBase

from utils.NestDistance import NestDistance
from utils.utils import get_ts_steps


class Features():
    
    def __init__(self, krill_radius, nestCount_radius, padding):
        
        self.krill_radius = krill_radius
        self.nestCount_radius = nestCount_radius
        self.padding = padding
        
        self.nest_distance = NestDistance()
        
        self.seaIce = get_seaIce(padding)
        
        self.krillbase = KrillBase()
        self.krillbase.create(krill_radius)

    def add_features(self, df_features):
        """
        Top-level function to add all desined features to the feature DataFrame
        containing the time-series information.
        """
        # Add the species information
        df_features = add_species(df_features)
        
        # Add the proximity nest count data
        df_features = add_proximity_nestCount(df_features, self.nestCount_radius, self.nest_distance)
        
        # Add the sea ice data
        df_features = add_seaIce(df_features, self.seaIce)
        
        # Add the krill data
        df_features = add_krill(df_features, self.krillbase)
        
        return(df_features)




def add_species(df_features):
    # These are the species of each row
    species = df_features.reset_index()['species']

    # Create the categories for the species
    categories = np.zeros((species.shape[0],3))
    categories[:,0] = species == 'adelie penguin'
    categories[:,1] = species == 'chinstrap penguin'
    categories[:,2] = species == 'gentoo penguin'

    # Assemble the DataFrame and add it to the features
    df = pd.DataFrame(categories, index=df_features.index, columns=['adelie penguin', 'chinstrap penguin', 'gentoo penguin'])
    df_features = pd.concat([df_features, df], axis=1)
    
    return(df_features)


def add_proximity_nestCount(df_features, radius, nest_distance):
    """
    This will add the median change of all nests found within radius of each
    location per species. Note that only nests of the same species are considered.
    """
    # Need to make sure the DataFrame is sorted
    df_features.sort_index(inplace=True)
        
    # Extract only the time stop column names
    #ts_step = [ item for item in df_features.columns if len(item)==2 and item[0] == 't' ]
    ts_step = get_ts_steps(df_features)[-1] # only take the values of the last year
    #ts_step = ts_step[-1] # only take the values of the last year
    
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

    

#def add_seaIce(df_features, agg_type, padding=1):
#    # Obtain the sea ice values
#    seaIce = get_seaIce(agg_type, padding=padding)
#    
#    # Assemble a DataFrame with the sea ice
#    tmp = df_features.reset_index()
#    vals = np.array([ seaIce[key] for key in zip(tmp['site_id'], tmp['year']) ])
#    
#    seaIceCol = [ 'sea_ice_px_%i'%i for i in range(vals.shape[1]) ]
#    df_seaIce = pd.DataFrame(vals, index=df_features.index, columns=seaIceCol)
#    
#    df_features = pd.concat([df_features, df_seaIce], axis=1)
#    return(df_features)

def add_seaIce(df_features, seaIce):
    # Assemble a DataFrame with the sea ice
    tmp = df_features.reset_index()
    vals = np.array([ seaIce[key] for key in zip(tmp['site_id'], tmp['year']) ])
    
    seaIceCol = [ 'sea_ice_month_%i'%i for i in range(vals.shape[1]) ]
    df_seaIce = pd.DataFrame(vals, index=df_features.index, columns=seaIceCol)
    
    df_features = pd.concat([df_features, df_seaIce], axis=1)
    return(df_features)


def add_krill(df_features, krillbase):
    vals = [ krillbase.query(site_id, int(year)) for (site_id, _, year) in list(df_features.index) ]
    df_features = df_features.assign(krill=vals)
    return(df_features)
    