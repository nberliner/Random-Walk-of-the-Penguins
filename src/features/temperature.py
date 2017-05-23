# -*- coding: utf-8 -*-
"""
Created on Fri May 19 16:02:52 2017

@author: nberliner
"""
import numpy as np

from netCDF4 import Dataset

import datetime
from dateutil.relativedelta import relativedelta

from data.data import breeding_locations


class Temperature():
    
    def __init__(self):
        
        self.lat, self.lon, self.time, self.temp = self._load_nc()
        
        self.siteLocations = breeding_locations()
    
    def _load_nc(self):
        fname = '../data/external/temperature/gistemp250.nc'
        try:
            with Dataset(fname, mode='r') as fh:
                lat = fh.variables['lat'][:]
                lon = fh.variables['lon'][:]
                time = self._convertTime(fh.variables['time'][:]) # will be datetime object
                temp = fh.variables['tempanomaly'][:] # axis are time, lat, lon
        except IOError:
            raise IOError("You must obtain the air.mon.mean.nc and place it into data/external/temperature/")
            
        # Remove years that are not needed
        idx = np.array([ t.year >= 1979 for t in time ])
        time = time[idx]
        temp = temp[idx,:,:].data # it's a masked array otherwise
        
        # Remove latitudes that are out of range
        idx = np.array([ (l < -58) & (l > -80) for l in lat ])
        lat = lat[idx]
        temp = temp[:,idx,:]
        
        # Create new entries for the 2017 season which was not yet observed
        newTemp = np.zeros((6,temp.shape[1],temp.shape[2]))
        newTime = np.zeros(6, dtype='object')
        for i, month in enumerate([5, 6, 7, 8, 9, 10]):
            idx = np.array([ (t.month == month) & (t.year >=  2010) for t in time ])
            newTemp[i,:,:] = np.mean(temp[idx,:,:], axis=0)
            newTime[i] = datetime.date(2017, month, 15)
        
        temp = np.concatenate([temp, newTemp], axis=0)
        time = np.concatenate([time, newTime], axis=0)
        
        # Replace the missing values with the "global" mean
        temp[np.isclose(temp, 32767)] = np.nan
        mean_temperature = np.nanmean(temp, axis=(1,2))
        for idx, mT in enumerate(mean_temperature):
            temp[idx,:,:][np.isnan(temp[idx,:,:])] = mT

        return(lat, lon, time, temp)
    
    def _convertTime(self, time):
        # The reference is specified as 1800/1/1 and the time are the hours passed since then
        time_conversion = lambda t: (datetime.date(1800,1,1) + relativedelta(days=int(t)))
        time = np.array([ time_conversion(t) for t in time ])
        return(time)
    
    def _antarcticSeason(self, time):
        year, month = time.year, time.month
        if month >= 11: # i.e. November
            year += 1
        return(year)

    def _closestPoint(self, lat, lon):
        idx_lat = np.argmin(np.abs(self.lat - lat))
        idx_lon = np.argmin(np.abs(self.lon - lon))
        return(idx_lat, idx_lon)
    
    def _closeArea(self, lat, lon, area=2):
        idx_lat = ((self.lat >= lat-area) & (self.lat <= lat-area))
        idx_lon = ((self.lon >= lon-area) & (self.lon <= lon-area))
        return(idx_lat, idx_lon)
    
    def _siteid2LatLon(self, site_id):
        return(self.siteLocations.loc[site_id].values)
        
    def query(self, site_id, year):
        lat, lon = self._siteid2LatLon(site_id)
        idx_lat, idx_lon = self._closestPoint(lat, lon)
        #idx_time = np.array([ t.year == year for t in self.time ])
        idx_time = np.array([ self._antarcticSeason(t) == year for t in self.time ])
        return(self.temp[idx_time,idx_lat,idx_lon])




#class Temperature():
#    
#    def __init__(self):
#        
#        self.lat, self.lon, self.time, self.temp = self._load_nc()
#        
#        self.siteLocations = breeding_locations()
#    
#    def _load_nc(self):
#        fname = '../data/external/temperature/air.mon.mean.nc'
#        try:
#            with Dataset(fname, mode='r') as fh:
#                lat = fh.variables['lat'][:]
#                lon = fh.variables['lon'][:]
#                time = self._convertTime(fh.variables['time'][:]) # will be datetime object
#                temp = fh.variables['air'][:] # axis are time, lat, lon
#        except IOError:
#            raise IOError("You must obtain the air.mon.mean.nc and place it into data/external/temperature/")
#            
#        # Remove years that are not needed
#        idx = np.array([ t.year >= 1980 for t in time ])
#        time = time[idx]
#        temp = temp[idx,:,:]
#
#        return(lat, lon, time, temp)
#    
#    def _convertTime(self, time):
#        # The reference is specified as 1800/1/1 and the time are the hours passed since then
#        time_conversion = lambda t: (datetime.date(1800,1,1) + relativedelta(hours=t))
#        time = np.array([ time_conversion(t) for t in time ])
#        return(time)
#
#    def _closestPoint(self, lat, lon):
#        idx_lat = np.argmin(np.abs(self.lat - lat))
#        idx_lon = np.argmin(np.abs(self.lon - lon))
#        return(idx_lat, idx_lon)
#    
#    def _siteid2LatLon(self, site_id):
#        return(self.siteLocations.loc[site_id].values)
#        
#    def query(self, site_id, year):
#        lat, lon = self._siteid2LatLon(site_id)
#        idx_lat, idx_lon = self._closestPoint(lat, lon)
#        idx_time = np.array([ t.year == year for t in self.time ])
#        return(self.temp[idx_time,idx_lat,idx_lon].data)