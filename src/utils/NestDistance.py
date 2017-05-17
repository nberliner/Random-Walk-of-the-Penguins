# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:38:14 2017

@author: nberliner
"""
import numpy as np
import pandas as pd

from geopy.distance import vincenty
from scipy.spatial.distance import pdist, squareform

from data.data import breeding_locations


class NestDistance():
    """
    Compute the vincenty distance between each site_id. The computed distance
    can be queried to retrieve all site_id's withing a given radius. Will return
    an empty list for site_id's for which no geographic location is found. 
    """
    
    def __init__(self):
        
        self.df = self._loadData()
        self.dist = self._distanceMatrix(self.df)
            
    def _loadData(self):
        df = breeding_locations()
        #df.reset_index(inplace=True) # make compatible with the earlier version for now
        return(df)
    
    def _distanceMatrix(self, df):
        fname = '../data/interim/nest_distMat.npy'
        try:
            distMat = np.load(fname)
            print("Found nest count pre-computed distance matrix in data/interim")
        except IOError:
            # Keep only the latitude and longitude information
            # Note that latitude comes first!
            data = df[['latitude_epsg_4326', 'longitude_epsg_4326']].values
            
            # Define the distance function
            metric = lambda lat, lng: vincenty(lat, lng).meters / 1000. # in kilometers
            
            # Compute the full distance matrix
            dist = squareform(pdist(data, metric=metric))
            
            # Place the array in a DataFrame
            distMat = pd.DataFrame(dist, index=list(df.index), columns=list(df.index))
            np.save(fname, distMat)
        
        return(distMat)
    
    def query(self, site_id, radius):
        try:
            # Select the site_id
            candidates = self.dist.loc[site_id, :]
            
            # Extract all other sites that are closer than radius
            sites = list(candidates[candidates <= radius].index)
            
            # Drop the entry to itself
            sites.remove(site_id)
        except KeyError:
            # Return an empty list if the site was not found, e.g. STOK
            sites = list()
        
        return(sites)