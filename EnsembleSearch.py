from sklearn.metrics import mean_absolute_error as mae
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.svm import SVR 
from sklearn.ensemble import AdaBoostRegressor as ADA
from sklearn.ensemble import BaggingRegressor as BAG
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.linear_model import RANSACRegressor as RAN
from sklearn.linear_model import PassiveAggressiveRegressor as PAR
from sklearn.linear_model import SGDRegressor as SGD
from sklearn.ensemble import VotingRegressor
import numpy as np
import random
from tqdm import tqdm
import copy

class EnsembleSearch:
            
    def __init__(self, X_train, y_train, X_test, y_test, epochs=3, size_pop=40, prob_mut=0.8, alpha_stop=1e-4, verbose=True):
        
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test
        self._size_pop = size_pop
        self._epochs = epochs
        self._fitness_array = np.array([])
        self._best_of_all = None
        self._verbose = verbose
        self._alpha_stop = alpha_stop
        self._prob_mut = prob_mut

    def gen_population(self):

        population = [[]]*self._size_pop
        
        for i in range(self._size_pop):
            content_RFR = ['RFR',RFR(), 
                         {'n_estimators':np.random.randint(1,100),
                          'max_depth':np.random.randint(1,20),
                          'min_samples_split':np.random.randint(2,5),      
                          'min_samples_leaf':np.random.randint(2,10),   
                          'min_weight_fraction_leaf':np.random.rand(1)[0]/2}]
            
            content_SVR = ['SVR',SVR(),
                         {'kernel':random.choice(['linear','rbf','poly','sigmoid']),     
                          'epsilon':np.random.rand(1)[0]/4,
                          'C':random.choice([1,10,100,1000]),'gamma':'auto'}]
            
            content_ADA = ['ADA',ADA(), 
                         {'n_estimators':np.random.randint(1,50)}]
            
            content_BAG = ['BAG',BAG(), 
                         {'n_estimators':np.random.randint(1,50),'max_samples':np.random.randint(1,20)}]
            
            content_GBR = ['GBR',GBR(), 
                         {'n_estimators':np.random.randint(1,100),'max_depth':np.random.randint(1,20),        
                          'min_samples_split':np.random.randint(2,5),      
                          'min_samples_leaf':np.random.randint(2,10),     
                          'min_weight_fraction_leaf':np.random.rand(1)[0]/2}]
            
            content_RAN = ['RAN',RAN(), {}, np.inf]
            
            content_PAR = ['PAR',PAR(), 
                         {'C': np.random.randint(1,10), 'early_stopping':True,        
                          'n_iter_no_change':np.random.randint(1,10)}]
        
            content_SGD = ['SGD',SGD(), {'penalty':random.choice(['l2', 'l1', 'elasticnet']),
                                       'n_iter_no_change':np.random.randint(1,10)}]
            
            list_regressors_content = [content_RFR,content_SVR,content_ADA,content_BAG,content_GBR,content_RAN,content_PAR,content_SGD]
            
            weights = np.random.random(size=len(list_regressors_content))
            
            for j in range(len(list_regressors_content)):
                list_regressors_content[j][1] = list_regressors_content[j][1].set_params(**list_regressors_content[j][2])

            population[i] = [weights, list_regressors_content, 'voting_regressor', np.inf]
            
        return population

    def set_fitness(self, population):
        # must evaluated each regressor individually also and sort inside the individual
        # then make a better crossover.
        for i in range(len(population)):
            lista_tuplas_VR_indv = []
            nomes = []
            individual = population[i]
            for regressor_content in individual[1]:
                # adds X if name already used
                regressor_name = regressor_content[0]
                while regressor_name in nomes:
                    regressor_name = regressor_name+'X'
                nomes.append(regressor_name)

                #make the tuples for VR: (name, regressor)
                regressor_object = regressor_content[1]
                lista_tuplas_VR_indv.append((regressor_name,regressor_object))
            
            Voting_regressor = VotingRegressor(lista_tuplas_VR_indv, weights=individual[0])
            Voting_regressor.fit(self._X_train, self._y_train)
            
            mae_vr = mae(Voting_regressor.predict(self._X_test), self._y_test)
            # sets fitness 
            individual[-1] = mae_vr
            # sets the object
            individual[-2] = Voting_regressor

            population[i] = copy.copy(individual)
            
        return population

    def crossover(self, population):
        qtRegressors = len(population[0][1])
        qtParents = int(len(population))
        for badParent in range(int(qtParents/2), qtParents-1):
            #cruzamento
            goodParent = int(badParent/2)
            if np.random.rand() > (1 - self._prob_mut):
                randomRegsQt = np.random.randint(low=0,high=qtRegressors)
                for _ in range(randomRegsQt):
                    # crossonly the same regressors
                    # worse index i receives from better index i-1
                    randomSample = np.random.randint(low=0,high=qtRegressors)
                    population[badParent][1][randomSample] = population[goodParent][1][randomSample]

                population[badParent][0] = population[goodParent][0]
                
        return population

    def mutation(self, population):
        """
            receives a population and mutates the VR weight by a random from 0 to 1
        """
        for i in range(1, len(population)-1):
            #mutation
            if np.random.rand() > (1 - self._prob_mut):
                population[i][0] += np.random.randn()/10
                
        return population
    
    def next_population(self, population):
        population = self.crossover(copy.copy(population))
        population = self.mutation(copy.copy(population))
        return population
    
    def early_stop(self):
        array = self._fitness_array
        to_break=False
        if len(array) > 4:
            array_diff1_1 = array[1:] - array[:-1]
            array_diff2 = array_diff1_1[1:] - array_diff1_1[:-1]
            
            if (self._verbose):
                print('second derivative: ', array_diff2[-2:].mean()) 
                print('first derivative: ', abs(array_diff1_1[-2:].mean()))
                print('fitness: ', array[-1])
                
            if (array_diff2[-4:].mean()) > 0 and (abs(array_diff1_1[-4:].mean()) < self._alpha_stop):
                to_break = True
        
        return to_break

    def search_best(self):
        population = self.gen_population()
        population = self.set_fitness(population)
        population.sort(key = lambda x: x[-1])  
        self._fitness_array = np.append(self._fitness_array, population[0][-1])
        self._best_of_all = population[0][-2]
        
        for i in tqdm(range(self._epochs)):
            population = self.next_population(population)
            population = self.set_fitness(population)
            population.sort(key = lambda x: x[-1])
            
            #pegar o melhor de todas as épocas
            if population[0][-1] < min(self._fitness_array):
                self._best_of_all = population[0][-2]
            
            #adicionar ao array de fitness o atual
            self._fitness_array = np.append(self._fitness_array, population[0][-1])

            if self.early_stop():
                break
            
        return self