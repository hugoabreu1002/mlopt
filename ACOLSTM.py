from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from mlopt.TimeSeriesUtils import MAPE
from mlopt.ACO import ACO
from numpy.random import seed
from tensorflow.random import set_seed
import warnings

class ACOLSTM:
    """
        X: X for lstm.
        
        y: y for lstm.
        
        train_test_split: division in train and test for X and y in lstm training and test.
    
        options_ACO: parametrization for ACO algorithm. EG:
            {'antNumber':2, 'antTours':1, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2}
    """
    def __init__(self, X_train, y_train, X_test, y_test, options_ACO, verbose=False):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._verbose = verbose
        self._options_ACO = options_ACO
        self._y_hat = None
        self._best_result = None
        self._best_result_fitness = None
        seed(1)
        set_seed(2)
    
    def setACO(self):
        alpha = self._options_ACO['alpha']
        beta = self._options_ACO['beta']
        rho = self._options_ACO['rho']
        Q = self._options_ACO['Q']
        ACOsearch = ACO(alpha, beta, rho, Q)    
        
        return ACOsearch
            
    def setModel(self, parameters):
        model = Sequential()
        model.add(LSTM(parameters['fl_qtn'], activation=parameters['fl_func'],
                       recurrent_activation=parameters['fl_refunc'],
                       return_sequences=True, input_shape=self._X.shape[0]))
        
        model.add(LSTM(parameters['sl_qtn'], activation=parameters['sl_func'],
                       recurrent_activation=parameters['sl_refunc'],))
        model.add(Dense(self._y.shape[1]))
        model.compile(optimizer=parameters['optimizer'], loss='mse')
        
        return model
    
    def fitModel(self, X):
        activation = ['elu', 'selu', 'tanh', 'relu', 'linear', 'sigmoid']
        optimizer = ['SGD', 'adam', 'rmsprop','Adagrad']
        epochs = [100,200,400]
        
        search_parameters={'fl_qtn':X[0],'fl_func':activation[X[1]],'fl_refunc':activation[X[2]],
                            'sl_qtn':X[3],'sl_func':activation[X[4]],'sl_refunc':activation[X[5]],
                            'optimizer':optimizer[X[6]]}
        
        model = self.setModel(search_parameters)
        model.fit(self._X_train, self._y_train, epochs=epochs[X[7]], verbose=0, shuffle=False,
                    use_multiprocessing=True)
        y_hat = model.predict(self._X_test)
        fitness = MAPE(y_hat, self._y_test)
        
        return model
    
    def optimize(self, searchSpace, verbose=True):
        """
            searchSpace: is the space of search for the ants.
            Ants 'X' will move for the graph in this problem based on the following parameters
            
            search_parameters={'fl_qtn':X[0],'fl_func':activation[X[1]],'fl_refunc':activation[X[2]],
                               'sl_qtn':X[3],'sl_func':activation[X[4]],'sl_refunc':activation[X[5]],
                               'optimizer':optimizer[X[6]]}
                               
            activation and optimizer choices:\n
            activation = ['elu', 'selu', 'tanh', 'relu', 'linear', 'sigmoid']\n
            optimizer = ['SGD', 'adam', 'rmsprop','Adagrad']\n
            epochs = [100,200,400]\n            
            searchSpace. E.G:\n
                fl_qtn = [10, 30, 50]\n
                fl_func = list(range(6))\n
                fl_refunc = list(range(6))\n                
                sl_qtn = [10, 30, 50]\n
                sl_func = list(range(6))\n
                sl_refunc = list(range(6))\n
                optimizer = list(range(4))\n
                epochs = list(range(3))\n
                searchSpace = [fl_qtn, fl_func, fl_refunc, sl_qtn, sl_func, sl_refunc, optimizer]
        """
        def fitnessFunction(X, *args):
            model = self.fitModel(X)
            y_hat = model.predict(self._X_test)
            fitness = MAPE(y_hat, self._y_test)
            return fitness
        
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        ACOsearch = self.setACO()
        self._best_result, self._best_result_fitness = ACOsearch.optimize(self._options_ACO['antNumber'],
                                                                          self._options_ACO['antTours'],
                                                                          dimentionsRanges=searchSpace,
                                                                          function=fitnessFunction,
                                                                          verbose=verbose)
        
        finalFitedModel = self.fitModel(self._best_result)
        y_hat = finalFitedModel.predict(self._X_test)
        
        return finalFitedModel, y_hat
        