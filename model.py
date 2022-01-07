import torch
from torch import nn

from typing import Tuple


def train(model, train_data: Tuple[torch.Tensor, torch.Tensor], val_data: Tuple[torch.Tensor, torch.Tensor], **kwargs):
    '''
    model train function
    
    Args:
        model: pytorch model
        train_data: X_train, y_train
        val_data: X_test, y_test
        kwargs:
            epochs: train epochs
            learning_rate: optimizer learning rate
        Example:
            >>> train(model, (X_train, y_train), (X_test, y_test), epochs=200, learning_rate=0.01)
    '''
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=kwargs['learning_rate'])
    
    for epoch in range(1, kwargs['epochs']+1):
        model.train()
        data, target = train_data
        outputs = model(data)
        optimizer.zero_grad()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        print(f'Epoch: {epoch:3d}, train_loss={loss.item():.4f},', end=' ')
        
        
        model.eval()
        with torch.no_grad():
            data, target = val_data
            outputs = model(data)
            val_loss = criterion(outputs, target).item()
        print(f'val_loss={val_loss:.4f}')



class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad=True)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad=True)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out)
        return out
    
    
class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, requires_grad=True)
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out) 
        return out
    

