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
    
    def __init__(self, X_train, y_train, X_test, y_test,
                 size_pop=20, epochs=5, alpha_stop=1e-4, verbose=True):
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.size_pop = size_pop
        self.epochs = epochs
        self.fitness_array_ = np.array([])
        self.best_of_all_ = None
        self.verbose_ = verbose
        self.alpha_stop = alpha_stop

    def gen_population(self):
        # TODO review this method
        # population could be an atribute, or a list of instance of some individual class.
        # where one individual would be a list o regressors.

        population = [[]]*self.size_pop
        
        for i in range(self.size_pop):
            
            qt_regressor = np.random.randint(2,9)

            # TODO add parameters to LR
            # take a look at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html?highlight=linearregression#sklearn.linear_model.LinearRegression
            lista_LR = ['LR',LR(), {}]
            
            lista_RFR = ['RFR',RFR(), 
                         {'n_estimators':np.random.randint(1,100),
                          'max_depth':np.random.randint(1,20),
                          'min_samples_split':np.random.randint(2,5),      
                          'min_samples_leaf':np.random.randint(2,10),   
                          'min_weight_fraction_leaf':np.random.rand(1)[0]/2}]
            
            # TODO Consider to remove ou replace SVR, becouse is too slow
            lista_SVR = ['SVR',SVR(),
                         {'kernel':random.choice(['linear','rbf','poly','sigmoid']),     
                          'epsilon':np.random.rand(1)[0]/4,
                          'C':random.choice([1,10,100,1000]),'gamma':'auto'}]
            
            lista_ADA = ['ADA',ADA(), 
                         {'n_estimators':np.random.randint(1,50)}]
            
            lista_BAG = ['BAG',BAG(), 
                         {'n_estimators':np.random.randint(1,50),'max_samples':np.random.randint(1,20)}]
            
            lista_GBR = ['GBR',GBR(), 
                         {'n_estimators':np.random.randint(1,100),'max_depth':np.random.randint(1,20),        
                          'min_samples_split':np.random.randint(2,5),      
                          'min_samples_leaf':np.random.randint(2,10),     
                          'min_weight_fraction_leaf':np.random.rand(1)[0]/2}]
            
            lista_RAN = ['RAN',RAN(), {}]
            
            lista_PAR = ['PAR',PAR(), 
                         {'C': np.random.randint(1,10), 'early_stopping':True,        
                          'n_iter_no_change':np.random.randint(1,10)}]
            
            # TODO add parameters to SGD 
            # take a look at https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDRegressor.html#sklearn.linear_model.SGDRegressor
            # consider to implement with https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html#sklearn.kernel_approximation.Nystroem transformer
            # it would become a pipiline then?
            lista_SGD = ['SGD',SGD(), {}]
            
            lista_regressors = [lista_LR,lista_RFR,lista_SVR,lista_ADA,lista_BAG,
                                lista_GBR,lista_RAN,lista_PAR,lista_SGD]
            
            random.shuffle(lista_regressors)
            
            lista_regressors = lista_regressors[0:qt_regressor]
            
            for j in range(len(lista_regressors)):
                lista_regressors[j][1] = lista_regressors[j][1].set_params(**lista_regressors[j][2])

            population[i] = [qt_regressor, lista_regressors, 'voting_regressor', np.inf]
            
        return population

    def set_fitness(self, population):
        for i in range(len(population)):
            
            lista_tuplas_VR = []
            nomes = []
            for indv in population[i][1]:
                
                while indv[0] in nomes: #adionar X se o nome já estiver dentro
                    indv[0] = indv[0]+'X'
                nomes.append(indv[0])
                
                lista_tuplas_VR.append((indv[0],indv[1])) #aqui vai pegando cada regressor do indivíduo (lista de regressores),
                                                          #que é formado pelo nome do regressor e o objeto.
                
            Voting_regressor = VotingRegressor(lista_tuplas_VR)
            Voting_regressor.fit(self.X_train, self.y_train)
            
            mae_vr = mae(Voting_regressor.predict(self.X_test), self.y_test)
            population[i][-1] = mae_vr
            population[i][-2] = Voting_regressor
            
        return population
    
    def next_population(self, population):
        # In this algorithm there is no mutation. 
        # no need of mutation, since the algorithm already have stochastics inside.

        # in this for loop change only the last half (worst) of the population.
        for i in range(int(len(population)/2), len(population)-1):
            qt_regs_pai1 = population[i][0]
            qt_regs_pai2 = population[i+1][0]
            
            #cruzamento
            if qt_regs_pai1<=qt_regs_pai2:    
                population[i][1][:int(qt_regs_pai1/2)] = population[2*i][1][:int(qt_regs_pai1/2)]
            else:
                population[i][1][:int(qt_regs_pai2/2)] = population[2*i][1][:int(qt_regs_pai2/2)]
                
            #modificar nomes dos regressores se houver repetido
            nomes = []
            for reg in population[i][1]:
                while reg[0] in nomes: #adionar X se o nome já estiver dentro
                    reg[0] = reg[0]+'X'
                nomes.append(reg[0])
        
        return population
    
    def early_stop(self):
        array = self.fitness_array_
        to_break=False
        if len(array) > 4:
            array_diff1_1 = array[1:] - array[:-1]
            array_diff2 = array_diff1_1[1:] - array_diff1_1[:-1]
            
            if (self.verbose_):
                print('second derivative: ', array_diff2[-2:].mean()) 
                print('first derivative: ', abs(array_diff1_1[-2:].mean()))
                print('featness: ', array[-1])
                
            if (array_diff2[-4:].mean()) > 0 and (abs(array_diff1_1[-4:].mean()) < self.alpha_stop):
                to_break = True
        
        return to_break

    def search_best(self):
        population = self.gen_population()
        population = self.set_fitness(population)
        population.sort(key = lambda x: x[-1])  
        self.fitness_array_ = np.append(self.fitness_array_, population[0][-1])
        self.best_of_all_ = population[0][-2]
        
        for i in tqdm(range(self.epochs)):
            population = self.next_population(population)
            population = self.set_fitness(population)
            population.sort(key = lambda x: x[-1])
            
            #pegar o melhor de todas as épocas
            
            if population[0][-1] < min(self.fitness_array_):
                self.best_of_all_ = population[0][-2]
            
            #adicionar ao array de fitness o atual
            self.fitness_array_ = np.append(self.fitness_array_, population[0][-1])

            if self.early_stop():
                break
            
        return self