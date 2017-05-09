# -*- coding: utf-8 -*-
"""
Created on Wed May  3 11:38:14 2017

@author: nberliner
"""
import pandas as pd

from geopy.distance import vincenty
from scipy.spatial.distance import pdist, squareform


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
        # Load the raw data
        fname = '../data/raw/training_set_observations.csv'
        try:
            df = pd.read_csv(fname, usecols=['site_id', 'longitude_epsg_4326', 'latitude_epsg_4326'])
        except IOError:
            raise IOError("You need to download and place the 'training_set_observations.csv' file into the 'data/raw' folder")
    
        df.drop_duplicates(inplace=True)
        return(df)
    
    def _distanceMatrix(self, df):
        # Keep only the latitude and longitude information
        # Note that latitude comes first!
        data = df[['latitude_epsg_4326', 'longitude_epsg_4326']].values
        
        # Define the distance function
        metric = lambda lat, lng: vincenty(lat, lng).meters / 1000. # in kilometers
        
        # Compute the full distance matrix
        dist = squareform(pdist(data, metric=metric))
        
        # Place the array in a DataFrame
        dist = pd.DataFrame(dist, index=list(df['site_id']), columns=list(df['site_id']))
        
        return(dist)
    
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