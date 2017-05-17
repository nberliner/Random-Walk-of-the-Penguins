# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:17:45 2017

@author: nberliner
"""
import numpy as np

from geopy.distance import vincenty
from scipy.spatial.distance import cdist


from data.data import load_krill_data, breeding_locations



class KrillBase():
    
    def __init__(self):
        
        self.df_krill = load_krill_data()
        self.df_breeding = breeding_locations()
        self.distMat = self._compute_distMat(self.df_krill, self.df_breeding)
        
        self.krillbase = None
        
    def _compute_distMat(self, df_krill, df_breeding):
        fname = '../data/interim/krill_distMat.npy'
        try:
            distMat = np.load(fname)
            print("Found krill pre-computed distance matrix in data/interim")
        except IOError:
            print("Computing krill distMat and caching result in data/interim/")
            print("This can take a while.. (apologies for computing this via brute force)")
            # Extract the latitude and longitude values
            data_krill = df_krill[['LATITUDE', 'LONGITUDE']].values
            data_breeding = df_breeding[['latitude_epsg_4326', 'longitude_epsg_4326']].values

            # Define the distance function
            metric = lambda lat, lng: vincenty(lat, lng).meters / 1000. # in kilometers

            # Compute the full distance matrix
            distMat = cdist(data_breeding, data_krill, metric=metric)
            np.save(fname, distMat)

        return(distMat)
    
    def create(self, radius):
        """
        Assemble the features that computes the average number of observed krill per location for the
        specified radius.
        """
        self.krillbase = dict()
        for idx, site_id in enumerate(list(self.df_breeding.index)):
            
            krill_stations = np.where(self.distMat[idx,:] <= radius)[0]
            for year in range(1980,2017):
                if len(krill_stations) == 0:
                    krill = np.nan
                else:
                    # Select only those observations that are within the range and the year
                    krill = self.df_krill.iloc[krill_stations,:].copy()
                    krill = krill[(krill['SEASON'] == year)]['STANDARDISED_KRILL_UNDER_1M2']
                    krill = krill.sum() / krill_stations.shape[0]
                
                self.krillbase[(site_id, year)] = krill
    
    def query(self, site_id, year, nan_value=0):
        """
        Get the krill concentration for a given site and year. If no krill was observed, set the value
        to nan_value.
        """
        val = self.krillbase[(site_id, year)]
        if np.isnan(val):
            val = nan_value
            
        return(val)