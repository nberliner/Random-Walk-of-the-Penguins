# -*- coding: utf-8 -*-
"""
Created on Wed May  3 16:16:17 2017

@author: nberliner
"""
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler

# Use sklearn scaler
class Scaler():
    """
    Scale the input data so that each timeseries will have roughly the same scale.
    """
    
    def __init__(self, df_nestCount, scalerType='robust'):
        
        self.df_nestCount = df_nestCount.copy()
        
        if scalerType == 'robust':
            self.data_scaler = RobustScaler()
        elif scalerType == 'standard':
            self.data_scaler = StandardScaler()
        else:
            raise ValueError("scalerType must be either 'robust' or 'standard'.")
        
        
        self.df_nestCountTransformed = df_nestCount.copy()
        self.df_nestCountTransformed[:] = self._fit_transform()
        
    def _fit_transform(self):
        val = self.data_scaler.fit_transform(self.df_nestCount.values.T)
        return(val.T)
    
    def transform(self, df):
        assert(all(df.index == self.df_nestCount.index))
        val = self.data_scaler.transform(df.values.T)
        df[:] = val.T
        return(df)
    
    def inverse_transform(self, df):
        assert(all(df.index == self.df_nestCount.index))
        val = self.data_scaler.inverse_transform(df.values.T)
        df[:] = val.T
        return(df)


## Use another scaler


#class percentChange():
#    
#    def __init__(self):
#        """
#        Compute the percentage change relative to the previous period, i.e. the current
#        value divided by the previous one.
#        
#        Use 100 as general value
#        """
#        self.df_nestCount = None
#        self.df_scaled = None
#        self.zero_value = None
#    
#    def _getDenominator(self, year):
#        # Get the deonminator for year
#        previousYear = str(int(year)-1)
#        denominator = self.df_nestCount.loc[:,previousYear]
#        denominator[denominator == 0] = self.zero_value
#        return(denominator)
#    
##    def fit(self, df_nestCount, df_nestCountError, zero_value=1.):
##        # Score the current df_nestCount
##        self.df_nestCount = df_nestCount
##        self.df_nestCountError = df_nestCountError
##        self.zero_value = zero_value
##        
##        # Divide the dataframe and compute the average percentage change per location and year
##        denominator = df_nestCount.shift(axis=1)
##        
##        # Keep a record of where the DataFrame didn't change
##        no_change = self.df_nestCount == denominator
##        
##        denominator[denominator == 0] = zero_value
##        
##        denominatorError = np.sqrt(df_nestCountError**2 + df_nestCountError.shift(axis=1)**2)
##        
##        self.df_scaled = self.df_nestCount / denominator
##        self.df_scaled.iloc[:,1:][zero_mask.iloc[:,1:]] = 1 # set all
##        
##        self.df_scaledError = self.df_nestCountError / denominatorError
##        
##        return(self.df_scaled, self.df_scaledError)
#        
#    def fit(self, df_nestCount, df_nestCountError, zero_value=1.):
#        # Score the current df_nestCount
#        self.df_nestCount = df_nestCount
#        self.df_nestCountError = df_nestCountError
#        self.zero_value = zero_value
#        
#        # Divide the dataframe and compute the average percentage change per location and year
#        self.delta = df_nestCount - df_nestCount.shift(axis=1)
#        
#        denominatorError = np.sqrt(df_nestCountError**2 + df_nestCountError.shift(axis=1)**2)
#        
#        self.df_scaled = self.delta / (df_nestCount + 1) # avoid dividing by zero
#        
#        self.df_scaledError = self.df_nestCountError / denominatorError
#        
#        return(self.df_scaled, self.df_scaledError)
#    
#    def inverse_transform(self, df, keep=True):
#        #assert(str(int(df_predict.name)-1) == self.denominator_predict.name)
##        df_predict = df * self._getDenominator(df.name)
#        df_predict = df * self.delta[df.name] + self.df_nestCount[str(int(df.name)-1)]
#        
#        # If the data is not already in the DataFrame, add them
#        if keep and df.name not in self.df_nestCount.columns:
#            self.df_nestCount = self.df_nestCount.assign(**{df.name: df_predict})
#        
#        df_predict -= 1 # remove the added count
#        return(df_predict)
        
        
class percentChange():
    
    def __init__(self):
        """
        Compute the percentage change relative to the previous period, i.e. the current
        value divided by the previous one.
        
        Use 100 as general value
        """
        self.df_nestCount = None
        self.df_scaled = None
        self.zero_value = None
        
        self.pseudoCount = 1
    
    def _getDenominator(self, year):
        # Get the deonminator for year
        previousYear = str(int(year)-1)
        denominator = self.df_nestCount.loc[:,previousYear]
        denominator[denominator == 0] = self.zero_value
        return(denominator)
    
        
    def fit(self, df_nestCount, df_nestCountError, zero_value=1.):
        # Score the current df_nestCount
        self.df_nestCount = df_nestCount + self.pseudoCount
        self.df_nestCountError = df_nestCountError
        self.zero_value = zero_value
        
        # Add a pseudo count
        #self.nestCountPresent = df_nestCount + self.pseudoCount # Add pseudo count
        #self.nestCountPast = self.nestCountPresent.shift(axis=1)
        
        # Divide the dataframe and compute the average percentage change per location and year
        #self.delta = self.nestCountPresent - self.nestCountPast
        
        denominatorError = np.sqrt(df_nestCountError**2 + df_nestCountError.shift(axis=1)**2)
        
        #self.denominator = np.maximum(self.nestCountPresent, self.nestCountPast)
        self.df_scaled = (self.df_nestCount - self.df_nestCount.shift(axis=1)) / self.df_nestCount.shift(axis=1)
        
        self.df_scaledError = self.df_nestCountError / denominatorError
        
        return(self.df_scaled, self.df_scaledError)
    
    def inverse_transform(self, df, keep=True):
        df_predict = self.df_nestCount[str(int(df.name)-1)] * (df + 1)
        
        # If the data is not already in the DataFrame, add them
        if keep and df.name not in self.df_nestCount.columns:
            self.df_nestCount = self.df_nestCount.assign(**{df.name: df_predict})
            
        df_predict -= self.pseudoCount
        
        return(df_predict)