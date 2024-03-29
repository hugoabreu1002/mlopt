from sklearn.metrics import mean_absolute_error as mae
from ..omodels.AgMlp import AgMlp as Ag_mlp
import numpy as np
import random
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class AGMLP_Residual:
    """ A Residual correction aproach for time series"""
    def __init__(self, data, mainForecast, num_epochs = 10, size_pop=10, prob_mut=0.8, tr_ts_percents=[80,20], alpha_stop=1e-4):
        """
            data - original data
            
            mainForecast - forecasted data
            
            num_epochs - number of epochs
            
            size_pop - size of population
            
            prob_mut - probability of mutation
            
            tr_ts_percents - list of train and test percentages. E.G: [80,20]
            
            alpha_stop - early stop criteria.
        """
        self._data = data
        self._data_train = data[:int(tr_ts_percents[0]/100*len(data))]
        self._data_test = data[int(tr_ts_percents[0]/100*len(data)):]
        self._mainForecast = mainForecast
        self._erro = data-mainForecast
        self._data_train_arima = mainForecast[:int(tr_ts_percents[0]/100*len(mainForecast))]
        self._data_test_arima = mainForecast[int(tr_ts_percents[0]/100*len(mainForecast)):]
        self._num_epochs = num_epochs
        self._size_pop = size_pop
        self._prob_mut = prob_mut
        self._tr_ts_percents = tr_ts_percents
        self._alpha_stop = alpha_stop
        self._fitness_array = np.array([])
        self._best_of_all = None
        
    def early_stop(self):
        array = self._fitness_array
        to_break=False
        if len(array) > 4:
            array_diff1_1 = array[1:] - array[:-1]
            array_diff2 = array_diff1_1[1:] - array_diff1_1[:-1]
            if (array_diff2[-4:].mean() > 0) and (abs(array_diff1_1[-4:].mean()) < self._alpha_stop):
                to_break = True

        return to_break
        
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

    def train_test_split_prev(self, serie, num_lags_pass, num_lags_fut, print_shapes = False):
        """
            Slipts a time series to train and test Data.
            X data are data num_lags_pass behind and num_lags_fut ahead y data.
        """
        len_serie = len(serie)
        X = np.zeros((len_serie, (num_lags_pass+num_lags_fut)))
        y = np.zeros((len_serie,1))
        for i in np.arange(0, len_serie):
            if (i-num_lags_pass > 0) and ((i+num_lags_fut) <= len_serie):
                X[i,:] = serie[i-num_lags_pass:i+num_lags_fut]
                y[i] = serie[i]
            elif (i-num_lags_pass > 0) and ((i+num_lags_fut) > len_serie):
                X[i,-num_lags_pass:] = serie[i-num_lags_pass:i]
                y[i] = serie[i]

        len_train = np.floor(len_serie*self._tr_ts_percents[0]/100).astype('int')
        len_test = np.ceil(len_serie*self._tr_ts_percents[1]/100).astype('int')

        X_train = X[0:len_train]
        y_train = serie[0:len_train]
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
            population[i] = [random.randint(1, 30), random.randint(1, 30),  random.randint(1, 30), random.randint(1, 30), 'objeto_erro', 'objeto_ass', 10]
        
        return population

    def set_fitness(self, population, start_set_fit): 
        for i in range(start_set_fit, len(population)):
            #obter o erro estimado
            erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida = self.train_test_split(
                self._erro, population[i][0])
            
            #AG_erro
            Ag_mlp_erro = Ag_mlp(erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida, self._num_epochs,
                self._size_pop, self._prob_mut).search_best_individual()
            best_erro = Ag_mlp_erro._best_of_all
            
            erro_estimado = np.concatenate([best_erro.predict(erro_train_entrada), best_erro.predict(erro_test_entrada)])

            #obter o y estimado
            X_ass_1_train_in, _, X_ass_1_test_in, _ = self.train_test_split(self._mainForecast, population[i][1])
            X_ass_2_train_in, _, X_ass_2_test_in, _ = self.train_test_split_prev(erro_estimado, population[i][2],
                                                                                 population[i][3])

            X_in_train = np.concatenate((X_ass_1_train_in, X_ass_2_train_in), axis=1)
            X_in_test = np.concatenate((X_ass_1_test_in, X_ass_2_test_in), axis=1) 
            
            #AG_ASS
            Ag_MLP_ass = Ag_mlp(X_in_train, self._data_train, X_in_test, self._data_test, self._num_epochs,
                                     self._size_pop, self._prob_mut).search_best_individual()
            best_ass = Ag_MLP_ass._best_of_all   
            
            
            population[i][-3] = best_erro
            population[i][-2] = best_ass
            population[i][-1] = mae(best_ass.predict(X_in_test), self._data_test)

        return population
    
    def cruzamento(self, population):
        """
            Crossover 
            the next population will receive the first 2 cromossoma
        """
        len_cross = len(population[0][:-3]) #gets the length of cromossomos, cutting the last three objetcts wich doest count as cromossoma.
        pop_ori = population
        len_pop = len(pop_ori)
        # do a loop for every individual keeping the first individual always.
        for p in range(1, len_pop):
            # if the proabiblity matches
            if np.random.rand() > (1 - self._prob_mut):
                # first half of cromossoma are taken from the simetric individual on the left (better)
                qt_to_cross = np.random.randint(0,len_cross)
                cromossoma_to_cross = np.random.choice(list(range(0,len_cross)), qt_to_cross)
                individual_to_take = int(p/2)
                for ctc in cromossoma_to_cross:
                    population[p][ctc] = pop_ori[individual_to_take][ctc]

        return population

    def mutation(self, population):
        for p in range(1, len(population)):
            if np.random.rand() > (1 - self._prob_mut):
                population[p][0] = population[p][0] + np.random.randint(-2, 2)
                if population[p][0] <= 0:
                    population[p][0] = 1
                
                population[p][1] = population[p][1] + np.random.randint(-2, 2)
                if population[p][1] <= 0:
                    population[p][1] = 1
                
                population[p][2] = population[p][2] + np.random.randint(-2, 2)
                if population[p][2] <= 0:
                    population[p][2] = 1
                
                population[p][3] = population[p][3] + np.random.randint(-2, 2)
                if population[p][3] <= 0:
                    population[p][3] = 1

        return population


    def new_gen(self, population, num_gen):
        """
            gets population already sorted, then do:
                crossover
                mutation
                set new fitness
                sort
        """
        population = self.cruzamento(population)
        population = self.mutation(population)
        population = self.set_fitness(population, int(self._size_pop*num_gen/(2*self._num_epochs)))
        population.sort(key = lambda x: x[:][-1]) 
        
        return population

    def search_best_model(self):
        ng = 0
        population = self.gen_population()
        population = self.set_fitness(population, ng)
        
        population.sort(key = lambda x: x[:][-1])
        self._fitness_array = np.append(self._fitness_array, population[0][-1])
        self._best_of_all = population[0]
        
        for ng in tqdm(range(0, self._num_epochs)):
            print('generation:', ng)
            population = self.new_gen(population, ng)
            if population[0][-1] < min(self._fitness_array):
                self._best_of_all = population[0]

            if self.early_stop():
                break
                
        return self

    def forecast_ahead(self, K, y_hat_main_aheadn, lag_residue_regression_index=0,
                      lag_original_association_index=1,
                      lag_estimated_residue_index=2,
                      forecast_estimated_residue_index=3,
                      error_object_index=4,
                      ass_object_index=5):
        """
            K - number of samples ahead

            y_hat_main_aheadn - comes from main model

            self._best_of_all : [lag_residue_regression, lag_original_association, lag_estimated_residue, forecast_estimated_residue
            , 'object_resiue_regression', 'object_association', fitness]
        """

        # TODO poder utilizar também com _best_of_all já exportados...
        
        gen_day_ahead = self._data.copy()
        
        lag_residue_regression = self._best_of_all[lag_residue_regression_index]
        forecast_estimated_residue = self._best_of_all[forecast_estimated_residue_index]
        lag_estimated_residue = self._best_of_all[lag_estimated_residue_index]
        error_model = self._best_of_all[error_object_index]
        ass_model = self._best_of_all[ass_object_index]
        
        for i in range(K):
            
            erro_day_ahead = gen_day_ahead - y_hat_main_aheadn[:len(gen_day_ahead)+i]

            erro_estimado_for_forecast = error_model.predict(
                erro_day_ahead[-lag_residue_regression:].reshape(1,-1))

            erro_fut = erro_day_ahead.copy()
            for _ in range(forecast_estimated_residue):
                erro_fut = np.append(erro_day_ahead, error_model.predict(
                    erro_fut[-lag_residue_regression:].reshape(1,-1)))

            X_ass_1_forecast_in = y_hat_main_aheadn[-self._best_of_all[lag_original_association_index]-1:]
            X_ass_2_forecast_in = np.concatenate((
                erro_estimado_for_forecast[-lag_estimated_residue-1:], erro_fut[forecast_estimated_residue-1:]))

            X_in_forecast = np.concatenate((X_ass_1_forecast_in, X_ass_2_forecast_in))

            y_forecast = ass_model.predict(X_in_forecast.reshape(1,-1))

            gen_day_ahead = np.append(gen_day_ahead, y_forecast)

        return gen_day_ahead[-K:]

class AGMLP_VR_Residual(AGMLP_Residual):
    def gen_population(self):
        """
            Generates the population. 
            The population is a list of lists where every element in the inner list corresponds to:
            [lag_residue_model, lag_original_main_model_association, lag_estimated_residue, forecast_estimated_residue,
            'percentage_of_mlps', 'object_resiue_model', 'object_association', fitness]
            
            The lags and forecast variables are token from a uniform distribution from 1 to 20.
        """
        population = [[1,1,1,1,50,'objeto_erro','objeto_ass',np.inf]]*self._size_pop
        for i in range(0, self._size_pop):
            population[i] = [random.randint(1, 20), random.randint(1, 20),  random.randint(1, 20), random.randint(1, 20), random.randint(1, 100), 'objeto_erro', 'objeto_ass', 10]
        
        return population

    def mutation(self, population):
        for p in range(1, len(population)):
            if np.random.rand() > self._prob_mut:
                population[p][0] = population[p][0] + np.random.randint(-2, 2)
                if population[p][0] <= 0:
                    population[p][0] = 1
                
                population[p][1] = population[p][1] + np.random.randint(-2, 2)
                if population[p][1] <= 0:
                    population[p][1] = 1
                
                population[p][2] = population[p][2] + np.random.randint(-2, 2)
                if population[p][2] <= 0:
                    population[p][2] = 1
                
                population[p][3] = population[p][3] + np.random.randint(-2, 2)
                if population[p][3] <= 0:
                    population[p][3] = 1
                    
                population[p][4] = population[p][4] + np.random.randint(-10, 10)
                if population[p][4] <= 0:
                    population[p][4] = 10
                if population[p][4] >= 100:
                    population[p][4] = 100

        return population
    
    def set_fitness(self, population, start_set_fit): 
        print('start_set_fit:', start_set_fit)
        
        for i in range(start_set_fit, len(population)):
            #obter o erro estimado
            erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida = self.train_test_split(self._erro, population[i][0])
            
            #AG_erro
            percent_VR_heuristic = population[i][4]
            
            VR_mlps_erro = Ag_mlp(erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida, self._num_epochs,
                                    self._size_pop, self._prob_mut).return_VotingRegressor(percent_VR_heuristic)

            erro_estimado = np.concatenate([VR_mlps_erro.VR_predict(erro_train_entrada), VR_mlps_erro.VR_predict(erro_test_entrada)])

            #obtain o y_hat. In thtat case only X data is needed from train_test_split and train_test_split_prev methods
            X_ass_1_train_in, _, X_ass_1_test_in, _ = self.train_test_split(self._mainForecast, population[i][1])
            X_ass_2_train_in, _, X_ass_2_test_in, _ = self.train_test_split_prev(erro_estimado, population[i][2],
                                                                                 population[i][3])
            #concatanates the X data for training
            X_in_train = np.concatenate((X_ass_1_train_in, X_ass_2_train_in), axis=1)
            X_in_test = np.concatenate((X_ass_1_test_in, X_ass_2_test_in), axis=1) 
            
            #AG_ASS
            VR_mlps_ass = Ag_mlp(X_in_train, self._data_train, X_in_test, self._data_test, self._num_epochs,
                                     self._size_pop, self._prob_mut).return_VotingRegressor(percent_VR_heuristic)
            
            #save the models and MAE fitness 
            population[i][-3] = VR_mlps_erro
            population[i][-2] = VR_mlps_ass
            population[i][-1] = mae(VR_mlps_ass.VR_predict(X_in_test), self._data_test)

        return population

    def forecastAhead(self, K, y_hat_main_aheadn,
                      y_true_data=None,
                      lag_residue_regression_index=0,
                      lag_original_association_index=1,
                      lag_estimated_residue_index=2,
                      forecast_estimated_residue_index=3,
                      error_object_index=5,
                      ass_object_index=6,
                      bestObject=None):
        """
            K - number of samples ahead

            y_hat_main_aheadn - comes from main model

            self._best_of_all : [lag_residue_model, lag_original_main_model_association, lag_estimated_residue, forecast_estimated_residue,
            'percentage_of_mlps', 'object_resiue_model', 'object_association', fitness]
        """

        # TODO poder utilizar também com _best_of_all já exportados...

        if y_true_data == None:
            y_true_data = self._data.copy()

        if bestObject == None:
            bestObject = self._best_of_all
        
        lag_residue_regression = bestObject[lag_residue_regression_index]
        forecast_estimated_residue = bestObject[forecast_estimated_residue_index]
        lag_estimated_residue = bestObject[lag_estimated_residue_index]
        error_model = bestObject[error_object_index]
        ass_model = bestObject[ass_object_index]
        
        for i in range(K):
            
            erro_samples_ahead = y_true_data - y_hat_main_aheadn[:len(y_true_data)]

            erro_estimado_for_forecast = error_model.VR_predict(
                erro_samples_ahead[-lag_residue_regression:].reshape(1,-1))

            erro_fut = erro_samples_ahead.copy()
            for _ in range(forecast_estimated_residue):
                erro_fut = np.append(erro_samples_ahead, error_model.VR_predict(
                    erro_fut[-lag_residue_regression:].reshape(1,-1)))

            X_ass_1_forecast_in = y_hat_main_aheadn[-bestObject[lag_original_association_index]-1:]
            X_ass_2_forecast_in = np.concatenate((
                erro_estimado_for_forecast[-lag_estimated_residue-1:], erro_fut[:forecast_estimated_residue]))

            X_in_forecast = np.concatenate((X_ass_1_forecast_in, X_ass_2_forecast_in))

            # print(X_ass_1_forecast_in.shape)
            # print(X_ass_2_forecast_in.shape)
            # print(X_in_forecast.shape)
            
            y_forecast = ass_model.VR_predict(X_in_forecast.reshape(1,-1))

            y_true_data = np.append(y_true_data, y_forecast)
            y_hat_main_aheadn = np.append(y_hat_main_aheadn, y_hat_main_aheadn[i])

        return y_true_data[-K:]
