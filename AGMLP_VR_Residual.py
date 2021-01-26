from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor
from sklearn.datasets import make_regression
from sklearn import preprocessing
from mlopt.AgMlp import AgMlp as Ag_mlp
from mlopt.AGMLP_Residual import AGMLP_Residual
import numpy as np
import random
from tqdm import tqdm

class AGMLP_VR_Residual(AGMLP_Residual):
    # TODO Documentar
    def gen_population(self):
        """
            Generates the population. 
            The population is a list of lists where every element in the inner list corresponds to:
            [lag_residue_regression, lag_original_sarimax_association, lag_estimated_residue, forecast_estimated_residue,
            'percentage_of_mlps', 'object_resiue_regression', 'object_association', fitness]
            
            The lags and forecast variables are token from a uniform distribution from 1 to 20.
        """
        population = [[1,1,1,1,1,'objeto_erro','objeto_ass',np.inf]]*self._size_pop
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
                    
                population[p][4] = population[p][3] + np.random.randint(-10, 10)
                if population[p][4] <= 0:
                    population[p][4] = 10

        return population
    
    def set_fitness(self, population, start_set_fit): 
        print('start_set_fit:', start_set_fit)
        
        for i in range(start_set_fit, len(population)):
            #obter o erro estimado
            erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida = self.train_test_split(
                self._erro, population[i][0])
            
            #AG_erro
            percent_VR_heuristic = population[i][4]
            
            VR_mlps_erro = Ag_mlp(erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida, self._num_epochs,
                                    self._size_pop, self._prob_mut).return_VotingRegressor(percent_VR_heuristic)

            erro_estimado = np.concatenate([VR_mlps_erro.VR_predict(erro_train_entrada), VR_mlps_erro.VR_predict(erro_test_entrada)])

            #obtain o y_hat. In thtat case only X data is needed from train_test_split and train_test_split_prev methods
            X_ass_1_train_in, _, X_ass_1_test_in, _ = self.train_test_split(self._y_sarimax, population[i][1])
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
