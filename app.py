import model
import os

from PyQt5.QtWidgets import *
from PyQt5 import uic

from utils.data import StockDataset


# FIXME get company lists
companies = ['AAPL', 'a', 'b', 'c', 'd', 'e']

# FIXME set default model lists
model_list = os.listdir('models')
class Predictor(QMainWindow):
    def __init__(self):
        super(Predictor, self).__init__()
        uic.loadUi('mygui.ui', self)
        self.show()
        
        # for company in companies:
        #     self.company.addItem(company)
        self.company.addItems(companies)
        self.models.addItems(model_list)
            
        self.train_button.clicked.connect(self.train)
        
        self.browse_model.clicked.connect(self.browse_file)
        
    def train(self):
        data = StockDataset(self.company.currentText(), tuple(map(int, (self.start_date_train.text().split('-')))), tuple(map(int, (self.end_date_train.text().split('-')))), int(self.window_size.text()))
        net = model.LSTM(self.input_dim.value(), self.hidden_dim.value(), self.num_layers.value(), self.output_dim.value())
        model.train(net, (data.X_train, data.y_train), (data.X_test, data.y_test), epochs=int(self.epochs.text()), learning_rate=float(self.learning_rate.text()))
    
    def browse_file(self):
        fname = QFileDialog.getOpenFileName(self, 'open file', './models')
        self.models.addItem(fname[0].split('models/')[-1])

def main():
    app = QApplication([])
    window = Predictor()
    app.exec_()
    

if __name__ == '__main__':
    main()