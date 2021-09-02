from sklearn.metrics import mean_absolute_error as mae
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import numpy as np
import random
from tqdm import tqdm
from sklearn.ensemble import VotingRegressor

class AgMlp:
    """
        Possible Upgrade in the future is to use number of hidden layer variable  
        and to use specifics parameters that depends on the solver and learning_rate:
        https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    """
    def __init__(self,X_train, y_train, X_test, y_test, num_generations, size_population, prob_mut, alpha_stop=1e-4):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._num_generations = num_generations
        self._size_population = size_population
        self._prob_mut = prob_mut
        self._fitness_array = np.array([])
        self._alpha_stop = alpha_stop
        self._best_of_all = None
        self._final_trained_mlps = None
        self._n_voting_mlps = None
    
    def gen_population(self):
        """
            Generates the population, which is a list of lists.
            Every individual (list) is composed by [solver, first hidden layer, second hidden layer,
            third hidden layer, activation function, learning rate]
        """
        sizepop=self._size_population
        population = [['']]*sizepop
        solver = ['lbfgs', 'adam', 'sgd']
        activation = ['identity', 'logistic', 'tanh', 'relu']
        learning_rate = ['constant', 'invscaling', 'adaptive']
        for i in range(0, sizepop):
            population[i] = [random.choice(solver), random.randint(1, 100), random.randint(1, 50), random.randint(1, 10), random.choice(activation),
                             random.choice(learning_rate),'MLP-object', 10]

        return population

    def set_fitness_and_sort(self, population, start_set_fit):
        for i in range(start_set_fit, len(population)):
            mlp_volatil = MLPRegressor(hidden_layer_sizes=(population[i][1], population[i][2], population[i][3]),
                                    activation = population[i][4], solver = population[i][0],
                                    learning_rate = population[i][5], max_iter = 500, early_stopping=True)
            #qt_fits=0
            mlp_volatil.fit(self._X_train, self._y_train)
            mae_fits= mae(self._y_test, mlp_volatil.predict(self._X_test))

            population[i][-1] = mae_fits
            population[i][-2] = mlp_volatil
        
        population.sort(key = lambda x: x[:][-1])
        
        return population

    def cruzamento(self, population):
        qt_cross = len(population[0])
        pop_ori = population
        for p in range(1, len(pop_ori)):
            if np.random.rand() > (1 - self._prob_mut):
                population[p][0:int(qt_cross/2)] = pop_ori[int(p/2)][0:int(qt_cross/2)]
                population[p][int(qt_cross/2):qt_cross] = pop_ori[int(p/2)][int(qt_cross/2):qt_cross]

        return population

    def mutation(self, population):
        for p in range(1, len(population)):
            if np.random.rand() > (1 - self._prob_mut):
                population[p][1] = population[p][1] + np.random.randint(1,10)
                population[p][2] = population[p][2] + np.random.randint(1,5)
                population[p][3] = population[p][3] + np.random.randint(1,2)

        return population
    
    def new_gen(self, population, num_gen):
        population = self.cruzamento(population)
        population = self.mutation(population)
        start_set_fit = int(self._size_population*num_gen/(2*self._num_generations))
        population = self.set_fitness_and_sort(population, start_set_fit)
        return population
    
    def early_stop(self):
        array = self._fitness_array
        to_break=False
        if len(array) > 4:
            array_diff1 = array[1:] - array[:-1]
            array_diff2 = array_diff1[1:] - array_diff1[:-1]
            if (array_diff2[-4:].mean() > 0) and (abs(array_diff1[-4:].mean()) < self._alpha_stop):
                to_break = True

        return to_break
    
    def search_best_individual(self):
        population = self.gen_population()
        population = self.set_fitness_and_sort(population, 0)

        self._fitness_array= np.append(self._fitness_array, population[0][-1])
        self._best_of_all = population[0][-2]

        for ng in tqdm(range(0, self._num_generations)):
            population = self.new_gen(population, ng)
            
            if population[0][-1] < min(self._fitness_array):
                self._best_of_all = population[0][-2]
                
            if self.early_stop():
                break

        self._final_trained_mlps = [p[-2] for p in population]
        
        return self

    def return_VotingRegressor(self, percent):
        """
            returns fited voting regressor objetc percent of bests mlps trained
        """
        self.search_best_individual()
        Number = int(len(self._final_trained_mlps)*percent/100)
        if Number < 1:
            Number = 1
        self._n_voting_mlps = self._final_trained_mlps[:Number]
        
        return self

    def VR_predict(self, Xin):
        filtered = list(filter(lambda x: type(x) is not str, self._n_voting_mlps))
        averaged_output = np.average(list(map(lambda x: x.predict(Xin), filtered)), axis=0)
        
        return averaged_output