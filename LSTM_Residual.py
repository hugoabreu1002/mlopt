from sklearn.metrics import mean_absolute_error as mae
import numpy as np
import random
from tqdm import tqdm
from mlopt.ACOLSTM import ACOLSTM

class LSTM_Residual:
    """ A Residual correction aproach for time series"""
    def __init__(self, data, y_arima, tr_ts_percents=[80,20]):
        """
            data - original data
            y_arima - forecasted data
            num_epochs - number of epochs
            size_pop - size of population
            prob_mut - probability of mutation
            tr_ts_percents - list of train and test percentages. E.G: [80,20]
            alpha_stop - early stop criteria.
        """
        self._data = data
        self._data_train = data[:int(tr_ts_percents[0]/100*len(data))]
        self._data_test = data[int(tr_ts_percents[0]/100*len(data)):]
        self._y_arima = y_arima
        self._erro = data-y_arima
        self._data_train_arima = y_arima[:int(tr_ts_percents[0]/100*len(y_arima))]
        self._data_test_arima = y_arima[int(tr_ts_percents[0]/100*len(y_arima)):]
        self._tr_ts_percents = tr_ts_percents
        self._best_of_all = None
        
    def train_test_split(self, serie, num_lags, print_shapes = False):
        """
            Slipts a time series to train and test Data.
            X data are data num_lags behind y data.
        """
        len_serie = len(serie)
        X = np.zeros((len_serie, num_lags))
        y = np.zeros((len_serie,1))
        for i in np.arange(0, len_serie):
            if i-num_lags>0:
                X[i,:] = serie[i-num_lags:i]
                y[i] = serie[i]

        len_train = np.floor(len_serie*self._tr_ts_percents[0]/100).astype('int')
        len_test = np.ceil(len_serie*self._tr_ts_percents[1]/100).astype('int')

        X_train = X[0:len_train]
        y_train = y[0:len_train]
        X_test = X[len_train:len_train+len_test]
        y_test = y[len_train:len_train+len_test]

        return X_train, y_train, X_test, y_test
    
    def gen_population(self):
        """
            Generates the population. 
            The population is a list of lists where every element in the inner list corresponds to:
            [lag_residue_regression, lag_original_sarimax_association, lag_estimated_residue, forecast_estimated_residue
            , 'object_resiue_regression', 'object_association', fitness]
            
            The lags and forecast variables are token from a uniform distribution from 1 to 20.
        """
        population = [[1,1,1,1,'objeto_erro','objeto_ass',np.inf]]*self._size_pop
        for i in range(0, self._size_pop):
            population[i] = [random.randint(1, 20), random.randint(1, 20),  random.randint(1, 20), random.randint(1, 20), 'objeto_erro', 'objeto_ass', 10]
        
        return population

    def fit(self, lag_error, searchSpace, options_ACO, saturate=True, saturation=[0,1]): 
        
        erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida = self.train_test_split(
            self._erro, lag_error)
        
        #LSTM_erro
        lstmOptimizer = ACOLSTM(erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida, 1,
                        options_ACO=options_ACO)

        final_model, error_hat_test = lstmOptimizer.optimize(searchSpace)

        print("shape data test arima: ")
        print(self._data_test_arima.shape)
        print("shape error_hat_test")
        print(error_hat_test.shape)
        
        y_hat_test = self._data_test_arima[:] + error_hat_test[:,0]
        if saturate:
            y_hat_test[y_hat_test < saturation[0]] = saturation[0]
            y_hat_test[y_hat_test > saturation[1]] = saturation[1]
            
        print("shape y_hat_test")
        print(y_hat_test.shape)
        
        fitness = mae(y_hat_test, self._data_test)
        print("Final ftiness MAE {0}".format((fitness)))

        return final_model, y_hat_test