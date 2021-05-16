# -*- coding: utf-8 -*-
"""
Created on Sat May 15 20:36:05 2021

@author: kevin
"""

import numpy as np
import pandas as pd
import os
import pickle

def split(df):
    past = 24*7
    future = 24*1
    x = pd.DataFrame(index=df.index)
    for p in reversed(range(1, past+1)):
        x = x.merge(df.shift(p), how='inner', left_index=True, right_index=True)
    y = pd.DataFrame(index=df.index)
    for f in range(future):
        y = y.merge(df.shift(-f), how='inner', left_index=True, right_index=True)
        
    index = x.dropna().index.intersection(y.index.dropna())
    return x.loc[index].values, y.loc[index].values

def run(args):
    # Split train test
    X, Y = [], []
    path = args.training_data_folder
    for file in os.listdir(path):
        if file.endswith('.csv') and file.startswith('target'):
            df = pd.read_csv(path+os.sep+file, index_col=0)
            x, y = split(df)
            X.extend(x)
            Y.extend(y)
    X, Y = np.array(X), np.array(Y)
    
    # Train
    from model import Model
    model = Model()
    model.train(X, Y)
    with open(args.model, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump(model, f)
    
def act(args):
    # Stretch
    consumption = pd.read_csv(args.consumption, index_col=0)
    generation = pd.read_csv(args.generation, index_col=0)
    x_df = consumption.merge(generation, how='inner', left_index=True, right_index=True)
    date = pd.date_range(pd.to_datetime(x_df.index[-1]), periods=25, freq='h').astype(str).tolist()[1: ]

    with open(args.model, 'rb') as f:  # Python 3: open(..., 'rb')
        model = pickle.load(f)
    x_df = x_df.values.flatten()
    y_pred = model.predict(x_df)
    
    actions = []

    data = np.append(date, np.array(actions), axis=1)
    return data
    