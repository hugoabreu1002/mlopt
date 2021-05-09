from keras.models import Sequential
from keras.layers import LSTM, Dense, LayerNormalization
from keras import backend as K
from mlopt.ACO import ACO
from sklearn.metrics import mean_absolute_error as MAE
import tensorflow as tf
import warnings
import numpy as np
import warnings
import sys

class ACOLSTM:
    """
        X: X for lstm.
        
        y: y for lstm.
        
        train_test_split: division in train and test for X and y in lstm training and test.
    
        options_ACO: parametrization for ACO algorithm. EG:
            {'antNumber':2, 'antTours':1, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2}
    """
    def __init__(self, X_train, y_train, X_test, y_test, n_variables, options_ACO, verbose=False):
        self._X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_variables))
        self._y_train = y_train
        self._X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_variables))
        self._y_test = y_test
        self._verbose = verbose
        self._options_ACO = options_ACO
        self._y_hat = None
        self._best_result = None
        self._best_result_fitness = None
        self._n_variables = n_variables
    
    def setACO(self):
        alpha = self._options_ACO['alpha']
        beta = self._options_ACO['beta']
        rho = self._options_ACO['rho']
        Q = self._options_ACO['Q']
        ACOsearch = ACO(alpha, beta, rho, Q)    
        
        return ACOsearch
            
    def setModel(self, parameters):
        model = None
        K.clear_session()
        model = Sequential()
        if self._verbose:
            print(parameters)

        #model.add(LayerNormalization())

        model.add(LSTM(units=parameters['fl_qtn'], activation=parameters['fl_func'],
                    recurrent_activation=parameters['fl_func'],
                    return_sequences=True, input_shape=(self._X_train.shape[1], self._n_variables)))
        
        model.add(LSTM(units=parameters['sl_qtn'], activation=parameters['sl_func'],
                    recurrent_activation=parameters['fl_func']))
        
        model.add(Dense(units=parameters['tl_qtn'], activation=parameters['tl_func']))

        model.add(Dense(self._y_train.shape[1]))
        
        model.compile(optimizer=parameters['optimizer'], loss='mae', metrics=['mse'])
            
        return model
    
    def fitModel(self, X):
        search_parameters={'fl_qtn':X[0],'fl_func':self._activations[X[1]],
                           'sl_qtn':X[2],'sl_func':self._activations[X[3]],
                           'tl_qtn':X[4],'tl_func':self._activations[X[5]],
                           'optimizer':self._optimizers[X[6]]}
        
        setedModel = self.setModel(search_parameters)

        for i in range(self._X_train.shape[1]):
            X_train_col = self._X_train[i]
            if np.isnan(np.sum(X_train_col)):
                raise("X train has nan in column {0}".format(i))

        if np.isnan(np.sum(self._y_train)):
            raise("y train has nan")

        setedModel.fit(self._X_train, self._y_train, epochs=self._epochs[X[7]], verbose=self._verbose, shuffle=False,
                   use_multiprocessing=True)
        
        return setedModel
    
    def optimize(self, Layers_Qtd=[[10, 14, 18, 22], [6, 8, 10], [1, 2, 4]],
                 activations=['elu', 'selu', 'tanh', 'relu', 'linear', 'sigmoid'],
                 optimizers=['SGD', 'adam', 'rmsprop','Adagrad'],
                 epochs = [100,200,300]):
        """
            searchSpace: is the space of search for the ants.
            Ants 'X' will move for the graph in this problem based on the following parameters
            
            search_parameters={'fl_qtn':X[0],'fl_func':activation[X[1]],'sl_qtn':X[2],'sl_func':activation[X[3]],
                               'optimizer':optimizer[X[4]]}
                               
            activation and optimizer choices:\n
            activation = ['elu', 'selu', 'tanh', 'relu', 'linear', 'sigmoid']\n
            optimizer = ['SGD', 'adam', 'rmsprop','Adagrad']\n
            epochs = [100,200,400]\n            

            searchSpace. E.G:\n
                firstLayer_qtn = [1,10, 30, 50]\n
                firstLayer_func = list(range(6))\n              
                secondLayer_qtn = [1,10, 30, 50]\n
                secondLayer_func = list(range(6))\n
                thirdLayer_qtn = [1,5, 10, 15]\n
                thirdLayer_func = list(range(6))\n
                optimizer = list(range(4))\n
                epochs = list(range(3))\n
                searchSpace = [firstLayer_qtn, firstLayer_func, secondLayer_qtn, secondLayer_func, thirdLayer_qtn, thirdLayer_func, optimizer, epochs]
        """

        self._activations = activations
        self._optimizers = optimizers
        self._epochs = epochs
        
        def fitnessFunction(X, *args):
            fitedmodel = self.fitModel(X)
            if self._verbose:
                print(fitedmodel.summary())
                
            y_hat = fitedmodel.predict(self._X_test)
            try:
                fitness = MAE(y_hat, self._y_test)
            except:
                fitness = sys.maxsize
                pass
            return fitness

        searchSpace = [Layers_Qtd[0], list(range(len(self._activations))),
                       Layers_Qtd[1], list(range(len(self._activations))),
                       Layers_Qtd[2], list(range(len(self._activations))),
                       list(range(len(self._optimizers))), list(range(len(self._epochs)))]
        
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        ACOsearch = self.setACO()
        self._best_result, self._best_result_fitness = ACOsearch.optimize(self._options_ACO['antNumber'],
                                                                          self._options_ACO['antTours'],
                                                                          dimentionsRanges=searchSpace,
                                                                          function=fitnessFunction,
                                                                          verbose=self._verbose)
        
        finalFitedModel = self.fitModel(self._best_result)
        y_hat = finalFitedModel.predict(self._X_test)
        
        return finalFitedModel, y_hat
        