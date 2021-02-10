# data and ml
import torch
import numpy as np
import pandas as pd

def sequence(ts, window):
    """
    Divide a Pandas DataFrame or Series into sub-sequences and convert to 
    torch tensors. 
    
    :param ts: The time series to divide.
    :param window: The size of the resulting sub-sequences. 
    :return: A tensor of sub-sequences.
    """
    if isinstance(ts, (pd.Series, pd.DataFrame)):
        ts = torch.tensor(ts.copy().to_numpy()) 
    elif not isinstance(ts, torch.Tensor): 
        ts = torch.tensor(ts)
    
    rem = len(ts) % window
    ts = ts[rem:]
    return torch.stack(torch.split(ts, window))                   

class Difference(object):
    """
    A callable class. Applies a difference transform to a sequence. Useful for detrending.
    """
    
    def __call__(self, sequence):
        """
        :param sequence: The input sequence to transform.
        :return: Returns a sequence representing the difference between each point and 
            the next in the input sequence.
        """
        initial_values = torch.clone(sequence[0])
        diff = sequence - torch.roll(sequence, 1, 0)
        diff[0] = initial_values
        return diff
        
    def inverse(self, diffs : torch.Tensor) -> torch.Tensor:
        """
        Reconstructs a time series from its difference transform. The first element
        of the sequence should be the first value of the time-series you wish to 
        reconstruct. 

        :param diffs: The tensor containing an initial value and following time-step
            differences.
        :return: The reconstructed time series data. 
        """
        seq = torch.empty_like(diffs)
        seq[0] = diffs[0]
        for i in range(1, len(diffs)):
            seq[i] = seq[i-1] + diffs[i]
        return seq