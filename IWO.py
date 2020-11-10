import numpy as np
from numpy.random import choice
import random
from tqdm import tqdm
import numpy as np
import sys

class Weed:
    def __init__(self, dim, minx, maxx, function, functionArgs=[]):
        self.Position = np.random.rand(dim)
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.CostFunction = function
        self.CostFunctionParameters = functionArgs=[]
        self.Cost=None
        self.calc_Cost()

    def calc_Cost(self):
        self.Cost = self.CostFunction(self.Position, self.CostFunctionParameters)

    def update_position(self, Position):
        self.Position = np.clip(Position, a_min=self.minx, a_max=self.maxx)
        self.calc_Cost()

class IWO(Weed):
    def __init__(self, dim, minx, maxx, function, functionArgs=[]):
        super(IWO, self).__init__(dim, minx, maxx, function, functionArgs=[])
        self.historic_best_pos = []
        self.historic_best_error = []
        self.best_iwo_cost = sys.float_info.max
    
    def Reproduction(self, population, Sigma, minCost, maxCost, Smin=0, Smax=5):
        # Initialize Offsprings Population
        newpop = []

        for i in range(len(population)):
            ratio = (population[i].Cost - minCost)/(maxCost - minCost)
            S = int(np.floor(Smin + (Smax - Smin)*ratio))
            
            for j in range(S):
                # Initialize Offspring
                newsol = Weed(self.dim, self.minx, self.maxx, self.CostFunction, self.CostFunctionParameters)
                
                # Generate Random Location
                newsol.update_position(population[i].Position + Sigma * np.random.randn(self.dim))
                
                # Add Offpsring to the Population
                newpop.append(newsol)

        return newpop

    def MergePopulation(self, population, newpopulation, weed_qtz):
        appended_pop = population + newpopulation
        sorted_pop = sorted(appended_pop, key=lambda x: x.Cost, reverse=False)
        return sorted_pop[:weed_qtz]
    
    def search(self, weed_qtz_i=10, weed_qtz_f=100, MaxIt=200, print_at_every = 10, Smin=0, Smax=5, Exponent = 2, sigma_initial = 0.5, sigma_final = 0.001):
        plot_follow = print_at_every
        population = [Weed(self.dim, self.minx, self.maxx, self.CostFunction, self.CostFunctionParameters) for _ in range(weed_qtz_i)]
        weed_qtz = weed_qtz_i
        
        for it in tqdm(range(MaxIt)):
            
            # Update Standard Deviation
            Sigma = ((MaxIt - it)/(MaxIt - 1))**Exponent * (sigma_initial - sigma_final) + sigma_final
            
            # Get Best and Worst Cost Values
            Costs = [pop.Cost for pop in population]            
            newpopulation = self.Reproduction(population, Sigma, min(Costs), max(Costs), Smin, Smax)
            population = self.MergePopulation(population, newpopulation, weed_qtz)
            
            # increase max population size
            weed_qtz += int((weed_qtz_f - weed_qtz)*it/MaxIt)
            
            self.historic_best_pos.append(population[0].Position)
            self.historic_best_error.append(population[0].Cost)

            if it >= plot_follow:
                plot_follow += print_at_every
                # mostrando a melhor rota a cada iteracao
                print("######## iteracao {0} ##########".format(it))
                print("Best Point {0}, Best Cost: {1}, Sigma: {2}, Weeds: {3}".format(population[0].Position, population[0].Cost, Sigma, len(population)))

        return self