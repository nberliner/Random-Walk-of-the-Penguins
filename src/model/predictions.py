# -*- coding: utf-8 -*-
"""
Created on Thu May  4 16:33:46 2017

@author: nberliner
"""

import numpy as np
import pandas as pd

from data.data import load_submissions
from features.features import add_features


### Fit the model

def fit_model(df_features, model, epochs=5, batch_size=128):
    """
    Fit the model to all entries in df_features.
    """
    dat_x, aux_input, acc_input, dat_y = assemble_model_input(df_features)
    
    model.fit({'ts_input': dat_x, 'aux_input': aux_input, 'acc_input': acc_input},
          {'main_output': dat_y},
          epochs=epochs, batch_size=batch_size)
    
    return(model)
    

def assemble_model_input(df_features):
    """
    The model requires the input to be divided into different "input groups".
    This function will take the feature DataFrame and compile the groups.
    """
    # Extract the time series columns
    ts_steps = [ item for item in df_features.columns if len(item)==2 and item[0] == 't' ]
    dat_x = df_features.loc[:,ts_steps].values[:,:,np.newaxis]
    
    # Extract the auxiliary input columns
    ignore = ['y_true', 'inferred_y_true', 'countError', 'y_pred', 'site_id', 'species', 'year']
    aux_cols = [ item for item in df_features.columns if item not in ts_steps and item not in ignore ]
    aux_input = df_features.loc[:,aux_cols].values
    
    # Extract the e_n error column
    acc_input = df_features.loc[:,'countError'].values
    
    # Extract the true nest count
    dat_y = df_features.loc[:,'y_true'].values
    
    return(dat_x, aux_input, acc_input, dat_y)



### Predict 

def select_last_year(df_features):
    """
    In order to predict into the future, we always need the input from the last
    year. 
    """
    df_lastYear = df_features.reset_index()
    
    lastYear = str(df_features.reset_index()['year'].apply(int).max())
    df_lastYear = df_lastYear[df_lastYear['year'] == lastYear]

    df_lastYear.set_index(['site_id', 'species', 'year'], inplace=True)
    return(df_lastYear)
    
def model_predict(df_features, model):
    # Assemble the data
    dat_x, aux_input, acc_input, dat_y = assemble_model_input(df_features)
    
    # Run the prediction
    y_pred = model.predict({'ts_input': dat_x, 'aux_input': aux_input, 'acc_input': acc_input})
    
    return(y_pred)
    

def predict(df_features, steps, model, radius):
    """
    Predicts for all entries and df_feature and steps into the future. The
    predicted values will be taken as input to predict the subsequent years.
    As long as the additional features can be assembled, the model can predict
    into the future.
    """
    
    # First, predict the whole thing
    df_features = df_features.assign(y_pred=model_predict(df_features, model))
    
    # Now, predict the upcoming years
    for i in range(steps+1):
        # Select the last year
        df_lastYear = select_last_year(df_features)

        # We already have the y_true values for the last available year, i.e. first shift the data
        df_pred = shift(df_lastYear, radius)

        # Predict the counts, they will be added to y_true for next years shift
        y_pred = model_predict(df_pred, model)
        df_pred.loc[:,'y_true'] = y_pred
        df_pred.loc[:,'y_pred'] = y_pred
        
        # Add the new predictions to the DataFrame
        df_pred.set_index(['site_id', 'species', 'year'], inplace=True)
        df_features = pd.concat([df_features, df_pred], axis=0)
    
    return(df_features)
    
    
def shift(df_features, radius):
    """
    Shift the feature DataFrame so that the predicted value will become the
    value of the past year etc. This will need to recompute the added features
    and teh radius parameter thus needs to be specified.
    """
    ts_steps = [ item for item in df_features.columns if len(item)==2 and item[0] == 't' ]
    
    # Shift the count data
    for i in range(len(ts_steps)-1):
        df_features.loc[:,ts_steps[i]] = df_features.loc[:,ts_steps[i+1]]

    df_features.loc[:,ts_steps[-1]] = df_features.loc[:,'y_true']

    # Set some entries to nan
    df_features.loc[:,'y_true'] = np.nan
    df_features.loc[:,'inferred_y_true'] = np.nan
    
    # Recomute the features
    #radius = 2
    df_features = add_features(df_features, radius)
    
    # Advance the year index
    df_features.reset_index(inplace=True)
    year = str(int(df_features.loc[:,'year'][0])+1)
    df_features.loc[:,'year'] = year
    df_features.set_index(['site_id', 'species', 'year'])
    
    return(df_features)



## Assemble predictions

def convert_predictions(df_predictions, scaler):
    """
    Rescale the predictions to compute the final score and prepare the data
    for submission.
    """
    # Select the relevant part
    df_predictions = df_predictions.loc[:,'y_pred'].reset_index()
    
    # Pivot the table
    df_predictions = df_predictions.pivot_table(index=['site_id', 'species'], columns='year', values='y_pred')
    
    # Rescale
    for year in df_predictions.columns:
        df_predictions.loc[:,year] = scaler.inverse_transform(df_predictions.loc[:,year])
    
    return(df_predictions)
    

def assemble_submission(df_predictions):
    assert(all([ year in df_predictions.columns for year in ['2014', '2015', '2016', '2017'] ]))
    
    df_submissions_format = load_submissions()
    df_submissions = df_predictions.loc[:,['2014', '2015', '2016', '2017']]
    
    _, df_submissions = df_submissions_format.align(df_submissions, join='left', axis=0)
    
    return(df_submissions)
    

## Compute the AMAPE of the last two years, i.e. 2012, 2013
class AMAPE():
    """
    Compute the AMAPE score for all predictions.
    """
    def __init__(self):
        
        self.df_nestCount, self.df_nestCountError = self._loadData()
        
    def _loadData(self):
        fname_count = '../data/raw/training_set_nest_counts.csv'
        fname_error = '../data/raw/training_set_e_n.csv'

        # Load the data
        try:
            df_nestCount = pd.read_csv(fname_count, index_col=[0,1])
        except IOError:
            raise IOError("You need to download and place the 'training_set_nest_counts.csv' file into the 'data/raw' folder")
        try:
            df_nestCountError = pd.read_csv(fname_error, index_col=[0,1])
        except IOError:
            raise IOError("You need to download and place the 'training_set_e_n.csv' file into the 'data/raw' folder")
        
        # Sort the index to make sure that we're comparing the right entries in the end
        df_nestCount.sort_index(inplace=True)
        df_nestCountError.sort_index(inplace=True)
        
        return(df_nestCount, df_nestCountError)
    
    def _amape(self, y_true, y_pred, accuracies):
        """ Adjusted MAPE
        """
        not_nan_mask = ~np.isnan(y_true)

        # calculate absolute error
        abs_error = (np.abs(y_true[not_nan_mask] - y_pred[not_nan_mask]))

        # calculate the percent error (replacing 0 with 1
        # in order to avoid divide-by-zero errors).
        pct_error = abs_error / np.maximum(1, y_true[not_nan_mask])

        # adjust error by count accuracies
        adj_error = pct_error / accuracies[not_nan_mask]

        # return the mean as a percentage
        return np.mean(adj_error)

    def _compute(self, predictions):
        try:
            y_true = self.df_nestCount.loc[:,predictions.name]
            accuracies = self.df_nestCountError.loc[:,predictions.name]
        except KeyError:
            return(np.nan)
        
        predictions.sort_index(inplace=True)
        assert(all(predictions.index == self.df_nestCount.index))
        
        score = self._amape(predictions, y_true, accuracies)
        return(score)
    
    def __call__(self, df_pred):
        scores = list()
        for year in df_pred.columns:
            scores.append(self._compute(df_pred.loc[:,year]))
        
        scores = pd.DataFrame({'AMAPE': scores}, index=df_pred.columns)
            
        return(scores)