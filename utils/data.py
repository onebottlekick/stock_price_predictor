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

data = load_data('AAPL', (2012, 1, 1), (2021, 1, 1))