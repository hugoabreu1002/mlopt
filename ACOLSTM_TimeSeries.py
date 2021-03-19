from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from mlopt.ACO import ACO

class ACOLSTM_TimeSeries:
    def __init__(self, X, y, train_test_split=[80,20], verbose=False):
        self._X = X
        self._y = y
        self._train_test_split = train_test_split
        self._verbose = verbose
        
    
        
    