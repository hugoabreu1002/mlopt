from keras.models import Sequential
from keras.layers import LSTM, Dense, LayerNormalization, Conv1D, BatchNormalization, Dropout, Flatten
from keras import backend as K
from mlopt.ACO import ACO
from sklearn.metrics import mean_absolute_error as MAE
import warnings
import numpy as np
import warnings

class ACOLSTM:
    """
        X_train: X_train for lstm.
        y_train: y_train for lstm.
        X_test: X_test for lstm.
        y_test: y_test for lstm.

        n_variables: equal to y_test.shape[1]
        
        train_test_split: division in train and test for X and y in lstm training and test.
    
        options_ACO: parametrization for ACO algorithm. EG:
            {'antNumber':2, 'antTours':1, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2}
    """
    def __init__(self, X_train, y_train, X_test, y_test, n_variables,
                 options_ACO={'antNumber':30, 'antTours':10, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2},
                 verbose=False):
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

        model.add(LSTM(units=parameters['fl_qtn'], activation=parameters['fl_func'],
                    recurrent_activation=parameters['fl_func'],
                    return_sequences=True, input_shape=(self._X_train.shape[1], self._n_variables)))
        model.add(LayerNormalization())
        
        model.add(LSTM(units=parameters['sl_qtn'], activation=parameters['sl_func'], return_sequences=True,
                    recurrent_activation=parameters['fl_func']))

        model.add(Flatten())
        model.add(Dense(units=parameters['tl_qtn'], activation=parameters['tl_func']))
        model.add(Dense(self._y_train.shape[1]))
        
        model.compile(optimizer=parameters['optimizer'], loss='mae', metrics=['mse'])
            
        return model
    
    def fitModel(self, X):
        search_parameters={'fl_qtn':X[0],'fl_func':self._activations[X[1]],'fl_refunc':self._activations[X[2]],
                           'sl_qtn':X[3],'sl_func':self._activations[X[3]],'sl_refunc':self._activations[X[4]],
                           'tl_qtn':X[5],'tl_func':self._activations[X[6]],
                           'optimizer':self._optimizers[X[7]]}
        
        setedModel = self.setModel(search_parameters)

        for i in range(self._X_train.shape[1]):
            X_train_col = self._X_train[i]
            if np.isnan(np.sum(X_train_col)):
                raise("X train has nan in column {0}".format(i))

        if np.isnan(np.sum(self._y_train)):
            raise("y train has nan")

        setedModel.fit(self._X_train, self._y_train, epochs=self._epochs[X[8]], verbose=self._verbose, shuffle=False,
                   use_multiprocessing=True)
        
        return setedModel

    def _fitnessFunction(self, X, *args):
        fitedmodel = self.fitModel(X)
        if self._verbose:
            print(fitedmodel.summary())
            
        y_hat = fitedmodel.predict(self._X_test)[:,0]
        if np.isnan(np.sum(y_hat)):
            fitness = 1000
        else:
            print("SHAPES output hat: {0} and test: {1}".format(y_hat.shape, self._y_test[:,0].shape))
            fitness = MAE(y_hat, self._y_test[:,0])
        return fitness
    
    def optimize(self, Layers_Qtd=[[10, 14, 18, 22], [6, 8, 10], [1, 2, 4]],
                 activations=['elu', 'selu', 'tanh', 'relu', 'linear', 'sigmoid'],
                 optimizers=['SGD', 'adam', 'rmsprop','Adagrad'],
                 epochs = [100,200,300]):
        """
            Layers_Qtd are the layers number of elements in each of the three layers to be search.
            As a list of list. E.G: [[10, 14, 18, 22], [6, 8, 10], [1, 2, 4]].

            First layer will be searched in ACO with number of elements as Layers_Qtd[0]... and so on
                               
            activation and optimizer choices:\n
            activation = ['elu', 'selu', 'tanh', 'relu', 'linear', 'sigmoid']\n
            optimizer = ['SGD', 'adam', 'rmsprop','Adagrad']\n
            epochs = [100,200,400]\n
        """

        self._activations = activations
        self._optimizers = optimizers
        self._epochs = epochs
        searchSpace = [Layers_Qtd[0], list(range(len(self._activations))), list(range(len(self._activations))), 
                       Layers_Qtd[1], list(range(len(self._activations))), list(range(len(self._activations))),
                       Layers_Qtd[2], list(range(len(self._activations))),
                       list(range(len(self._optimizers))), list(range(len(self._epochs)))]
        
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        ACOsearch = self.setACO()
        self._best_result, self._best_result_fitness = ACOsearch.optimize(self._options_ACO['antNumber'],
                                                                          self._options_ACO['antTours'],
                                                                          dimentionsRanges=searchSpace,
                                                                          function=self._fitnessFunction,
                                                                          verbose=self._verbose)
        
        finalFitedModel = self.fitModel(self._best_result)
        y_hat = finalFitedModel.predict(self._X_test)[:,0]
        
        return finalFitedModel, y_hat

class ACOCLSTM(ACOLSTM):
    """
        X: X for lstm.
        
        y: y for lstm.
        
        train_test_split: division in train and test for X and y in lstm training and test.
    
        options_ACO: parametrization for ACO algorithm. EG:
            {'antNumber':2, 'antTours':1, 'alpha':2, 'beta':2, 'rho':0.5, 'Q':2}
    """
    def setModel(self, parameters):
        model = None
        K.clear_session()
        model = Sequential()
        if self._verbose:
            print(parameters)

        #model.add(Input(shape=(self._X_train.shape[1], self._n_variables)))
        model.add(Conv1D(filters=parameters['conv_fl_filters_qtn'], kernel_size=(int(parameters['conv_fl_kernel_sz'])),
                         padding='same'))
        model.add(Conv1D(filters=parameters['conv_sl_filters_qtn'], kernel_size=(int(parameters['conv_sl_kernel_sz'])),
                         padding='same'))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))

        model.add(LSTM(units=parameters['fl_qtn'], activation=parameters['fl_func'],
                       recurrent_activation=parameters['fl_refunc'],
                       return_sequences=True))
        model.add(LayerNormalization())
        
        model.add(LSTM(units=parameters['sl_qtn'], activation=parameters['sl_func'],
                    recurrent_activation=parameters['fl_refunc'],
                    return_sequences=True))

        model.add(Flatten())
        model.add(Dense(units=parameters['tl_qtn'], activation=parameters['tl_func']))
        model.add(Dense(self._y_train.shape[1]))
        
        model.compile(optimizer=parameters['optimizer'], loss='mae', metrics=['mse'])
            
        return model
    
    def fitModel(self, X):
        search_parameters={'conv_fl_filters_qtn':X[0], 'conv_fl_kernel_sz':X[1],
                           'conv_sl_filters_qtn':X[2], 'conv_sl_kernel_sz':X[3],
                           'fl_qtn':X[4],'fl_func':self._activations[X[5]], 'fl_refunc':self._activations[X[6]],
                           'sl_qtn':X[7],'sl_func':self._activations[X[8]], 'sl_refunc':self._activations[X[9]],
                           'tl_qtn':X[10],'tl_func':self._activations[X[11]],
                           'optimizer':self._optimizers[X[12]]}
        
        setedModel = self.setModel(search_parameters)

        for i in range(self._X_train.shape[1]):
            X_train_col = self._X_train[i]
            if np.isnan(np.sum(X_train_col)):
                raise("X train has nan in column {0}".format(i))

        if np.isnan(np.sum(self._y_train)):
            raise("y train has nan")

        setedModel.fit(self._X_train, self._y_train, epochs=self._epochs[X[13]], verbose=self._verbose, shuffle=False,
                   use_multiprocessing=True)
        
        return setedModel
    
    def optimize(self, Layers_Qtd=[[10, 14, 18, 22], [6, 8, 10], 
                                   [10, 14, 18, 22], [6, 8, 10], [1, 2, 4]], 
                 ConvKernels=[[8,12],[4,6]],
                 activations=['elu', 'selu', 'tanh', 'relu', 'linear', 'sigmoid'],
                 optimizers=['SGD', 'adam', 'rmsprop','Adagrad'],
                 epochs = [100,200,300]):
        """
            Layers_Qtd are the layers number of elements in each layers to be search.
            As a list of list. E.G: [[10, 14, 18, 22], [6, 8, 10], [10, 14, 18, 22], [6, 8, 10], [1, 2, 4]]. The first two layers corresponds
            to number of filters in Conv1d layers.

            ConvKernels are the size of Conv1D kernels (first two layers). E.G: [[8,12],[4,6]]

            First layer will be searched in ACO with number of elements as Layers_Qtd[0]... and so on
                               
            activation and optimizer choices:\n
            activation = ['elu', 'selu', 'tanh', 'relu', 'linear', 'sigmoid']\n
            optimizer = ['SGD', 'adam', 'rmsprop','Adagrad']\n
            epochs = [100,200,400]\n
        """

        self._activations = activations
        self._optimizers = optimizers
        self._epochs = epochs
        searchSpace = [Layers_Qtd[0], ConvKernels[0],
                       Layers_Qtd[1], ConvKernels[1],
                       Layers_Qtd[2], list(range(len(self._activations))),list(range(len(self._activations))),
                       Layers_Qtd[3], list(range(len(self._activations))),list(range(len(self._activations))), 
                       Layers_Qtd[4], list(range(len(self._activations))),
                       list(range(len(self._optimizers))), list(range(len(self._epochs)))]
        
        warnings.filterwarnings("ignore") # specify to ignore warning messages
        ACOsearch = self.setACO()
        self._best_result, self._best_result_fitness = ACOsearch.optimize(self._options_ACO['antNumber'],
                                                                          self._options_ACO['antTours'],
                                                                          dimentionsRanges=searchSpace,
                                                                          function=self._fitnessFunction,
                                                                          verbose=self._verbose)
        
        finalFitedModel = self.fitModel(self._best_result)
        y_hat = finalFitedModel.predict(self._X_test)[:,0]
        
        return finalFitedModel, y_hat