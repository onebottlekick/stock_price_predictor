import numpy as np
import pandas as pd
from typing import Tuple
import pandas_datareader as pdr
from datetime import date, datetime



def load_data(company: str, start:Tuple[int, int, int]=None, end:Tuple[int, int, int]=None) -> pd.DataFrame:
    '''
    Load stock price data from yahoo finance api
    
    Args:
        company: company name
        start: from
        end: to
    Return:
        stock price data: pandas DataFrame
    Example:
        >>> data = load_data('AAPL', (2012, 1, 1), (2021, 1, 1))
    '''
    start = datetime(*start)
    end = datetime(*end)
    data = pdr.DataReader(company, 'yahoo', start, end)
    return data


def windowing(data:pd.DataFrame, window_size:int) -> np.ndarray:
    '''
    makes time series data, label
    
    Args:
        data: stock price data
        window_size: interval date between data and label
    Return:
        X, y: data and label
    Example:
        >>> X, y = windowing(data, 14)
    '''
    X = []
    y = []
    
    for i in range(len(data)-window_size*2):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i + window_size*2])
        
    return np.array(X), np.array(y)