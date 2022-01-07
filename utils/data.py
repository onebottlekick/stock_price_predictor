import torch

import numpy as np
import pandas as pd
import pandas_datareader as pdr

from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from typing import Tuple

class StockDataset:
    def __init__(self, company: str, start:Tuple[int, int, int], end:Tuple[int, int, int], window_size:int):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.scaler = MinMaxScaler()
        data = self.load_data(company, start, end)
        data = self.scaler.fit_transform(data)
        X, y = self.windowing(data, window_size)
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2)
            
        self.X_train = torch.from_numpy(self.X_train).type(torch.Tensor).to(device)
        self.X_test = torch.from_numpy(self.X_test).type(torch.Tensor).to(device)
        self.y_train = torch.from_numpy(self.y_train).type(torch.Tensor).to(device)
        self.y_test = torch.from_numpy(self.y_test).type(torch.Tensor).to(device)


    @staticmethod
    def load_data(company: str, start:Tuple[int, int, int], end:Tuple[int, int, int]) -> pd.DataFrame:
        '''
        Load stock price data from yahoo finance api

        Args:
            company: company name
            start: from
            end: to
        Return:
            stock price data: pandas DataFrame (columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
        Example:
            >>> data = load_data('AAPL', (2012, 1, 1), (2021, 1, 1))
        '''
        start = datetime(*start)
        end = datetime(*end)
        data = pdr.DataReader(company, 'yahoo', start, end)
        return data

    @staticmethod
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
    
    def inverse_t(self, pred):
        '''
        inverse transform from MinMaxScaler
        
        Args:
            pred: model predictions
        Return:
            inverse transformed predictions
        Example:
            >>> stockdataset.inverse_t(model(X_test)[0].detach().numpy())
        '''
        return self.scaler.inverse_transform(pred)