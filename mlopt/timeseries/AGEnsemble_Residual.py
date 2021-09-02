from sklearn.metrics import mean_absolute_error as mae
import numpy as np
from ..omodels.EnsembleSearch import EnsembleSearch
from .AGMLP_Residual import AGMLP_Residual

class AGEnsemble_Residual(AGMLP_Residual):

    def set_fitness(self, population, start_set_fit): 
        print('start_set_fit:', start_set_fit)
        for i in range(start_set_fit, len(population)):
            #obter o erro estimado
            erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida = self.train_test_split(self._erro, population[i][0])
            
            #AG_erro
            ensemble_residual = EnsembleSearch(erro_train_entrada, erro_train_saida, erro_test_entrada, erro_test_saida,
                                                self._num_epochs, self._size_pop, self._prob_mut, verbose=False).search_best()

            best_erro = ensemble_residual._best_of_all
            
            erro_estimado = np.concatenate([best_erro.predict(erro_train_entrada), best_erro.predict(erro_test_entrada)])

            #obter o y estimado
            X_ass_1_train_in, _, X_ass_1_test_in, _ = self.train_test_split(self._y_sarimax, population[i][1])
            X_ass_2_train_in, _, X_ass_2_test_in, _ = self.train_test_split_prev(erro_estimado, population[i][2], population[i][3])
                   
            X_in_train = np.concatenate((X_ass_1_train_in, X_ass_2_train_in), axis=1)
            X_in_test = np.concatenate((X_ass_1_test_in, X_ass_2_test_in), axis=1) 
            
            #AG_ASS
            ensemble_ass = EnsembleSearch(X_in_train, self._data_train, X_in_test, self._data_test,
                                            self._num_epochs, self._size_pop, self._prob_mut, verbose=False).search_best()

            best_ass = ensemble_ass._best_of_all
            
            
            population[i][4] = best_erro
            population[i][5] = best_ass
            population[i][-1] = mae(best_ass.predict(X_in_test), self._data_test)

        return population
