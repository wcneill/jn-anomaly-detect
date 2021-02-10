from in_place import InPlace

import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from pathlib import Path
import os
from typing import List, Union

class SimpleDataset(Dataset):
    """
    A simple custom dataset class that returns event data to a 
    PyTorch ``DataLoader`` object.
    """
    def __init__(self, data, transform=None):
        """
        :param data: Your data.
        :param transform: Optional, default ``None``. Transformation for 
            preprocessing or data augmentation. 
        """
        super().__init__()
        self.data = data
        self.transform = transform
        
        if isinstance(data, (pd.DataFrame, pd.Series)):
            self.data = torch.tensor(data.to_numpy())
        elif not isinstance(data, torch.Tensor):
            self.data = torch.tensor(data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        item = self.data[idx]
            
        if self.transform:
            return self.transform(item)
        return item

    def __len__(self):
        return len(self.data)    
    
class TimeSeriesDataSet(Dataset):
    """
    A custom dataset that accepts a single time series and chunks it into 
    sub-sequences of a desired size. 
    
    This class truncates the input sequence so that its length is evenly 
    divisible by the desired window size. This truncation happens on the 
    front end of the (older data is truncated if necessary).
    
    It is expected that there are no empty dimensions in the input data. 
    """
    def __init__(self, data, window, transform=None):
        """
        :param data: The input time series for windowing.
        :param window: The window size or sub-sequence length desired.
        :param difference: If True, creates a dataset of the difference
            between consecutive time steps.
        """
        super().__init__()
        
        if data.shape[0] < window:
            raise RuntimeError(f"Timeseries of length {data.shape[0]} is too short for window size of {window}.")
        if isinstance(data, (pd.Series, pd.DataFrame)):
            dates = data.index
            data = torch.tensor(data.copy().to_numpy()) 
        else: 
            raise TypeError(f"Data should be Pandas dataframe. Found {type(data)}.")
        
        try:
            self.n_features = data.shape[1]
        except IndexError:
            self.n_features = 1
        
        rem = len(data) % window
        self.transform = transform
        self.window = window
        self.data = data[rem:]
        self.seqs = torch.stack(torch.split(self.data, window))        
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        if isinstance(idx, int):
            sub_seq = torch.clone(self.seqs[idx]).view(-1, self.n_features)
            if self.transform:
                sub_seq = self.transform(sub_seq)
            return sub_seq
        else:
            raise TypeError("Dataset only accepts a single integer index.")
            
        
    def add_dataset(self, other):
        """
        In place concatenation of another dataset to this one. 
        """
        # Check for matching sequence length and number of variables
        if (self.n_features == other.n_features) and (self.window == other.window):
            self.seqs = torch.unbind(self.seqs) + torch.unbind(other.seqs)
            self.seqs = torch.stack(self.seqs)
        else:
            raise ValueError("sequence length and/or num features mismatch:\n" \
                            + f" This data set shape: {self.seqs.shape}\n" \
                            + f" Other data set shape: {other.seqs.shape}")
        
        self.data = torch.cat((self.data, other.data))
