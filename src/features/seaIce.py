# -*- coding: utf-8 -*-
"""
Created on Mon May 15 12:35:29 2017

@author: nberliner
"""
import pickle

import numpy as np
import pandas as pd

from geopy.distance import vincenty
from scipy.spatial.distance import cdist

from data.data import breeding_locations
from data.seaIceData import sea_ice_filenames, load_sea_ice_tiffs

    

def compute_distMat(df_locations, lats, longs):
    # Extract the coordinates from the site_id locations
    data = df_locations[['latitude_epsg_4326', 'longitude_epsg_4326']].values

    # Put the latitude and longitude into an array
    latlng = np.column_stack((lats.flatten(), longs.flatten()))
        
    # Define the distance function
    metric = lambda lat, lng: vincenty(lat, lng).meters / 1000. # in kilometers
        
    # Compute the full distance matrix
    distMat = cdist(data, latlng, metric=metric)
    
    return(distMat)



def compute_closestPixel(lats, longs):
    # Load the site_id locations
    df_locations = breeding_locations()
    
    # Compute the distance matrix
    fname = '../data/interim/sea_ice_distMat.npy'
    try:
        print("Loading sea ice distMat from data/interim/")
        distMat = np.load(fname)
    except IOError:
        print("Computing sea ice distMat and caching result in data/interim/")
        print("This can take a while.. (apologies for computing this via brute force)")
        distMat = compute_distMat(df_locations, lats, longs)
        np.save(fname, distMat)
        print("Done.")
    
    # Get the closest location.
    idx = distMat.argmin(axis=1)
    
    rows = [ np.unravel_index(item, lats.shape)[0] for item in idx ]
    cols = [ np.unravel_index(item, lats.shape)[1] for item in idx ]
    
    df_closestPixel = pd.DataFrame({'row': rows, 'col': cols}, columns=['row', 'col'], index=df_locations.index)
    
    return(df_closestPixel)

    


#def compute_aggregate_sea_ice(tiffValues, agg='average'):
#    # Store the result in a new array. Use the first axis for the years, the rest are the dimensions of the image
#    sea_ice = np.zeros((37, 332, 316))
#    
#    for i, year in enumerate(range(1980,2017)):
#        # Check if the agg is defined
#        if not agg in ['average', 'median', 'max', 'min']:
#            raise ValueError('agg %s not defined' %agg)
#        
#        
#        # Put all the values in an array and use the first axis to compute the agg
#        vals = np.array([ tiffValues[tiff] for tiff in sea_ice_filenames(year) ])
#        
#        if agg == 'average':
#            arr = np.average(vals, axis=0)
#        elif agg == 'median':
#            arr = np.median(vals, axis=0)
#        elif agg == 'max':
#            arr = np.max(vals, axis=0)
#        elif agg == 'min':
#            arr = np.min(vals, axis=0)
#        else:
#            raise ValueError('This is bad. We should never arrive here!')
#        
#        sea_ice[i,:,:] = arr
#    
#    return(sea_ice)




def select_subarray(array, row, col, pad=1):
    """
    Select a small square area around a center pixel. If the array is 3D, the
    first axis is kept and the square is taken from the second and third axis.
    """
    if len(array.shape) == 2:
        val = array[(row-pad):(row+pad+1),(col-pad):(col+pad+1)]
    elif len(array.shape) == 3:
        val = array[:,(row-pad):(row+pad+1),(col-pad):(col+pad+1)]
    else:
        raise ValueError("select_subarray only works with 2D or 3D input arrays.")
    return(val)


#def assemble_sea_ice(df_closestPixel, values, padding=1, flatten=True):
#    """
#    Compute the sea ice feature for each site id and year. The result will be
#    a dictionary that can be used as fast lookup for adding the feature to the
#    data. 
#    """
#    # Create the new DataFrame holding the data
#    years = [ year for year in range(1980,2017) ]
#    site_ids = list(df_closestPixel.index)
#    
#    seaIce = dict()
#    for site_id in site_ids:
#        for idx, year in enumerate(years):
#            # Extract the sea ice data
#            row, col = df_closestPixel.loc[site_id]
#            feature = select_subarray(values[idx,:,:], row, col, pad=padding)
#            
#            # Flatten the array if requested
#            if flatten:
#                feature = feature.flatten()
#            
#            # Keep the result in a dictionary
#            seaIce[(site_id, str(year))] = feature
#    
#    return(seaIce)
    

#def sea_ice_agg_window(df_closestPixel, tiffValues, agg_type, padding, flatten):
#    """
#    Compute an aggregate for each pixel over the year. The location values
#    will then be returned as flatten array.
#    """
#    # Compute the aggregate of the tiff values (final result will be a dict)
#    seaIce = compute_aggregate_sea_ice(tiffValues, agg=agg_type)
#    seaIce = assemble_sea_ice(df_closestPixel, seaIce, padding=padding, flatten=flatten)
#    return(seaIce)
    

def sea_ice_agg_time(df_closestPixel, tiffValues, padding=1):
    """
    Aggregate the spatial information and return the yearly time progression
    """
    years = [ year for year in range(1980,2017) ]
    site_ids = list(df_closestPixel.index)
    
    seaIce = dict()
    for site_id in site_ids:
        for idx, year in enumerate(years):
            # Assemble the values of the respective years                    
            arr = np.array([ tiffValues[tiff] for tiff in sea_ice_filenames(year) ])
            
            # Fix two missing tiff. This will simply duplicate the adjecent values
            if year == 1988:
                arr = np.insert(arr, 0, arr[0], axis=0)
            elif year == 1987:
                arr = np.insert(arr, 10, arr[10], axis=0)
            
            # Select the subarray
            row, col = df_closestPixel.loc[site_id]
            vals = select_subarray(arr, row, col, pad=padding)
            
            weight = vals.shape[0] * vals.shape[1]
            
            # Compute the weighted average of sea ice
            vals[vals == -1] = np.nan
            vals = np.nansum(vals, axis=1) # first axis, i.e. rows
            vals = np.nansum(vals, axis=1) # second axis, i.e. columns
            
            # Normalise by the number of pixels
            vals = vals / weight
            
            seaIce[(site_id, str(year))] = vals # will allow to make an array
    
    return(seaIce)
    


def get_seaIce(padding):
    
    fname = "../data/interim/seaIce_data.p"
    
    try:
        seaIce = pickle.load(open(fname, "rb" ))
        print("Found cached sea ice data. Loaded from data/interim")
    except IOError:
        # Load the data
        lats, longs, tiffValues = load_sea_ice_tiffs()
        
        # Compte the closest pixel for each location
        df_closestPixel = compute_closestPixel(lats, longs)
        
        # Compute the sea ice data (final result will be a dict)
        seaIce = sea_ice_agg_time(df_closestPixel, tiffValues, padding=padding)
        
        # Cache the result
        pickle.dump(seaIce, open(fname, 'wb'))
    
    return(seaIce)