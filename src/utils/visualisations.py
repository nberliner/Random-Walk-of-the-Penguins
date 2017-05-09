# -*- coding: utf-8 -*-
"""
Created on Tue May  2 16:56:44 2017

@author: nberliner
"""
import numpy as np
import matplotlib.pyplot as plt

from random import sample

from data.data import load_nest_counts


class PenguinVisualisation():
    
    def __init__(self, predictions=None):
        
        # Load the data
        df_nestCount, df_nestCountError, nan_mask = load_nest_counts()
        
        self.nestCount = df_nestCount
        self.nestCountError = df_nestCountError
        self.predictions = predictions
    
    def _createFigure(self, rowIndex):
        fig = plt.figure(figsize=(15,5))
        ax = fig.add_subplot(111)
        
        ax.set_title('Penguin count for %s at %s' %(rowIndex[1], rowIndex[0]), size=16)
        ax.set_xlabel('Time', size=14)
        ax.set_ylabel('Count', size=14)
        return(ax)
    
    def plot_random(self, returnFigure=False):
        indices = sample(list(self.nestCount.index), 10)
        
        fig = plt.figure(figsize=(14,20))
        for i, rowIndex in enumerate(indices, start=1):
            ax = fig.add_subplot(10,1,i)
            ax.set_title(rowIndex)
            ax = self.plot_penguins(rowIndex, ax)
        
        fig.tight_layout()
        
        if returnFigure:
            return(fig)

    def plot_penguins(self, rowIndex, ax=None):
        
        if ax is None:
            ax = self._createFigure(rowIndex)
        
        # Get the count data and the associated error
        s = self.nestCount.loc[rowIndex]
        yerr = s.values * self.nestCountError.loc[rowIndex].values
    
        idx = np.array(s.isnull())

        X_line = np.array(list(s.index))
        X_line[idx] = np.nan

        X_scatter = np.array(list(s.index[~idx]), dtype='float64') # convert the type for errorbar function
        Y_scatter = np.array(s[~idx])
        yerr      = yerr[~idx]
        
        #ax.scatter(X_scatter, Y_scatter)
        ax.errorbar(X_scatter, Y_scatter, yerr=yerr, fmt='o')
        ax.plot(X_line, s.values)
        
        if self.predictions is not None:
            X_pred = np.array([ int(x) for x in self.predictions.columns ])
            Y_pred = self.predictions.loc[rowIndex].values
            
            ax.scatter(X_pred, Y_pred, color='red', s=60)

        #ax.set_xlim([1979, 2014])
        
        return(ax)