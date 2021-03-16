from sklearn.metrics import mean_absolute_error as mae
from sklearn.linear_model import LinearRegression as LR
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
            
            qt_regressor = np.random.randint(2,9)

            lista_RFR = ['RFR',RFR(), 
                         {'n_estimators':np.random.randint(1,100),
                          'max_depth':np.random.randint(1,20),
                          'min_samples_split':np.random.randint(2,5),      
                          'min_samples_leaf':np.random.randint(2,10),   
                          'min_weight_fraction_leaf':np.random.rand(1)[0]/2},
                         np.inf]
            
            lista_SVR = ['SVR',SVR(),
                         {'kernel':random.choice(['linear','rbf','poly','sigmoid']),     
                          'epsilon':np.random.rand(1)[0]/4,
                          'C':random.choice([1,10,100,1000]),'gamma':'auto'},
                         np.inf]
            
            lista_ADA = ['ADA',ADA(), 
                         {'n_estimators':np.random.randint(1,50)}, np.inf]
            
            lista_BAG = ['BAG',BAG(), 
                         {'n_estimators':np.random.randint(1,50),'max_samples':np.random.randint(1,20)}, np.inf]
            
            lista_GBR = ['GBR',GBR(), 
                         {'n_estimators':np.random.randint(1,100),'max_depth':np.random.randint(1,20),        
                          'min_samples_split':np.random.randint(2,5),      
                          'min_samples_leaf':np.random.randint(2,10),     
                          'min_weight_fraction_leaf':np.random.rand(1)[0]/2}, np.inf]
            
            lista_RAN = ['RAN',RAN(), {}, np.inf]
            
            lista_PAR = ['PAR',PAR(), 
                         {'C': np.random.randint(1,10), 'early_stopping':True,        
                          'n_iter_no_change':np.random.randint(1,10)}, np.inf]

            
            lista_SGD = ['SGD',SGD(), {'penalty':random.choice(['l2', 'l1', 'elasticnet']),'n_iter_no_change':np.random.randint(1,10)}, np.inf]
            
            lista_regressors = [lista_RFR,lista_SVR,lista_ADA,lista_BAG,
                                lista_GBR,lista_RAN,lista_PAR,lista_SGD]
            
            random.shuffle(lista_regressors)
            
            lista_regressors = lista_regressors[0:qt_regressor]
            
            for j in range(len(lista_regressors)):
                lista_regressors[j][1] = lista_regressors[j][1].set_params(**lista_regressors[j][2])

            population[i] = [qt_regressor, lista_regressors, 'voting_regressor', np.inf]
            
        return population

    def set_fitness(self, population):
        # must evaluated each regressor individually also and sort inside the individual
        # then make a better crossover.
        for i in range(len(population)):
            
            lista_tuplas_VR = []
            nomes = []
            for indv in population[i][1]:
                # adds X if name already used
                while indv[0] in nomes:
                    indv[0] = indv[0]+'X'
                nomes.append(indv[0])
                #make the tuples for VR: (name, regressor)
                lista_tuplas_VR.append((indv[0],indv[1]))

            Voting_regressor = VotingRegressor(lista_tuplas_VR)
            Voting_regressor.fit(self._X_train, self._y_train)
            
            mae_vr = mae(Voting_regressor.predict(self._X_test), self._y_test)
            # sets fitness 
            population[i][-1] = mae_vr
            # sets the object
            population[i][-2] = Voting_regressor
            
        return population
    
    def next_population(self, population):
        # In this algorithm there is no mutation. 
        # no need of mutation, since the algorithm already have stochastics inside.

        # in this for loop keeps best.
        for i in range(1, len(population)-1):
            #cruzamento
            if np.random.rand() > (1 - self._prob_mut):
                randomRegsQt = np.random.randint(0,min(population[i][0], population[i-1][0]))
                for _ in range(randomRegsQt):
                    randomSample = np.random.randint(0,randomRegsQt)
                    population[i][1][randomSample] = population[i-1][1][randomSample]
                
            #modificar nomes dos regressores se houver repetido
            nomes = []
            for reg in population[i][1]:
                while reg[0] in nomes: #adionar X se o nome já estiver dentro
                    reg[0] = reg[0]+'X'
                nomes.append(reg[0])
        
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