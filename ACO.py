import sparse
import numpy as np
from tqdm import tqdm
from scipy.stats import mode
import sys
import itertools as it

class ACO(object):
    """
        antNumber : number of ants
        
        alpha : parameter for probabilities matrix
        
        beta : parameter for probabilities matrix
        
        rho : for pherormone
        
        Q : for pherormone
        
        dimentionsRanges : must be a list of itretables
    
        fitenessFunction : must be like, and returns a float from 0 to inf, the smaller means the better
            result = fitenessFunction(self.Space(self.antsVertice[k_ant]), *fitnessFunctionArgs)
            
        fitnessFunctionArgs : args Diferent than the antsVertice in Space.    
    """

    fitnessFunctionArgs = None

    def __init__(self, alpha, beta, rho, Q):
        self._alpha = alpha
        self._beta = beta
        self._rho = rho
        self._Q = Q
        self._Dij = None
        self._Pij = None
        self._Tij = None
        self._Space = None
        self._antsVertice = None
        self._oldAntsVertice = None
        self._verticesFitness = None
        self._allBest = None
        self._allBestFitness = sys.maxsize
        self._ants_History = None
        self._antNumber = None
        self._antTours = None
        self._dimentionsRanges = None
        self.fitnessFunction = None
        self._fitnessFunctionArgs = None
    
    def setSpace(self):
        """
            Dimentions_Ranges: is a list of ranges. E.g:
                p = d = q = range(0, 2)
                Dimentions_Ranges = [p, d, q]
                
            The vertices of the graph will be a line of Space
        """

        Space = np.array(list(it.product(*self._dimentionsRanges)), dtype = np.int)

        return Space    
    
    def initializeVerticesFitness(self):
        return 1/np.zeros(self._Space.shape[0])
    
    def initializeMatricesAndAntsPosition(self):
        # TODO make with dynamic programming to prevent uncessary allocation of a lot o memory.
        # maybe use float 32 dtype...    
        self._Space = self.setSpace()
        self._verticesFitness = self.initializeVerticesFitness()
        self._Dij = 1/np.zeros((self._Space.shape[0], self._Space.shape[0]), dtype=np.float32)
        self._Pif = sparse.COO(np.ones((self._Space.shape[0], self._Space.shape[0]), dtype=np.float32))
        self._Tij = sparse.COO(np.ones((self._Space.shape[0], self._Space.shape[0]), dtype=np.float32))
        
        self._antsVertice = np.random.choice(range(self._Space.shape[0]), size=self._antNumber)
        self._oldAntsVertice = np.zeros(self._antNumber, dtype=int)
        self._ants_History = [None]*self._antTours
    
    def updateDij(self, Dij, verbose=False):
        """
            Dij and Pij will be only the matrix for the current possibilities
            Tij will be the pherormonen matrix for the whole graph
            
            fitnessFunction - lesser the better, so a good path should, Cj lesser than Ci
            Dij_ij = Exp((Cj-Ci)/Ci) + max 10% random
            Dij_ji = Exp((Ci-Cj)/Cj) + max 10% random
            
            the random idea is like the ants cant get the distance perfectly
        """
        for k_ant in range(self._antNumber):
            i_index = self._antsVertice[k_ant]
            j_index = np.random.choice(range(0, self._Space.shape[0]))
            
            if i_index != j_index: # ant should not stay at the point

                if verbose:
                    print("Setting fitness for")
                    print(self._Space[i_index, :])

                Ci = self.fitnessFunction(self._Space[i_index, :], self._fitnessFunctionArgs)
                self._verticesFitness[i_index] = Ci

                if verbose:
                    print("fitness is")
                    print(Ci)
                    print("Setting fitness for")
                    print(self._Space[j_index, :])
                
                Cj = self.fitnessFunction(self._Space[j_index, :], self._fitnessFunctionArgs)
                self._verticesFitness[j_index] = Cj

                if verbose:
                    print("fitness is")
                    print(Cj)

                Dij_ij = np.exp((Cj-Ci)/Ci)
                Dij[i_index, j_index] = Dij_ij #+ Dij_ij*np.random.rand(1)/10
    
                Dij_ji = np.exp((Ci-Cj)/Cj)
                Dij[j_index, i_index] = Dij_ji #+ Dij_ji*np.random.rand(1)/10
            
            else:
                Dij[j_index, i_index] = sys.maxsize
        
        return Dij
    
                                          
    def updateTij(self, Tij, Dij, Ants, last_Ants, rho=0.5, Q=1):
        # Dij_inf = Dij == sys.maxsize
        # Dij_notinf = Dij != sys.maxsize
        # Dij[Dij_inf] = Dij_notinf.max()

        sumdeltaTij = np.zeros(Tij.shape, dtype=np.float32)

        for kij in zip(last_Ants, Ants):
            sumdeltaTij[kij] += Q/Dij[kij]

        Tij = (1-rho)*Tij + sumdeltaTij

        Tij += np.random.randint(1, size=Tij.shape)/10

        return Tij

                                          
    def updatePij(self, Pij, Tij, Dij, alpha=1, beta=1):
        # Dij_inf = Dij == sys.maxsize
        # Dij_notinf = Dij != sys.maxsize
        # Dij[Dij_inf] = Dij_notinf.max()

        Pij = (Tij**alpha)/(Dij**beta)
        Pij += np.random.randint(1, size=Pij.shape)/10
        
        row_sums = Pij.sum(axis=1)
        Pij = Pij / row_sums[:, np.newaxis]
        
        return Pij


    def getHistorySolutions(self):
        self._ants_History = list(filter(lambda x: not x is None, self._ants_History))
        return self._ants_History, 
    
    def plotHistorySolutions(self):
        #TODO
        return 0

    def updateAntsPosition(self, Ants, Pij, verbose=False):
        last_Ants = Ants.copy()

        for i in range(Ants.shape[0]):
            k = Ants[i]

            possible_move = np.argwhere(Pij[k,:] > 0).flatten()

            if possible_move.shape[0] != 0:
                weights = Pij[k, possible_move]/Pij[k, possible_move].sum()
                Ants[i] = np.random.choice(possible_move, p=weights)
            else:
                Ants[i] = np.random.choice(np.array(range(Pij.shape[1])))

            if verbose:
                print("Ant {} possibilities:".format(i))
                print(possible_move)
                print("Ant {} move from {} to {}".format(i, k, Ants[i]))

        return Ants, last_Ants
    
                                          
    def optimize(self, antNumber, antTours, dimentionsRanges, function, functionArgs=[], verbose=False):
        """
            antNumber : Number of ants
            
            antTours : Number of tours each ant will make on the graph
            
            dimentionsRanges : Dimentions of the Graph, [[x1_min:x1_max],[x2_1, x2_2, x2_3, ...],...]
            
            function : function to be optimized
            
            functionArgs : *args of the function
        """
        self._antNumber = antNumber
        self._antTours = antTours
        self._dimentionsRanges = dimentionsRanges
        self.fitnessFunction = function
        self._fitnessFunctionArgs = functionArgs
        self.initializeMatricesAndAntsPosition()

        if verbose:
            print("dimentions Ranges passed: ", self._dimentionsRanges)
            print("Space Created: ", self._Space)
            print("number of Space Possibilities (rows): ", self._Space.shape[0])
        
        for it in tqdm(range(self._antTours)):
            self._Dij = self.updateDij(self._Dij, verbose)
            if verbose:
                print("Dij: ")
                print(self._Dij)

            self._Tij = self.updateTij(self._Tij, self._Dij, self._antsVertice, self._oldAntsVertice, self._rho, self._Q)
            if verbose:
                print("Tij: ")
                print(self._Tij)

            self._Pij = self.updatePij(self._Pij, self._Tij, self._Dij)
            if verbose:
                print("Pij:")
                print(self._Pij)
            
            self._antsVertice, self._oldAntsVertice = self.updateAntsPosition(self._antsVertice.copy(), self._Pij, verbose)
            self._ants_History[it] = self._antsVertice.copy()

            if verbose:
                print("Dij: ")
                print(self._Dij)
                print("Tij: ")
                print(self._Tij)
                print("Pij:")
                print(self._Pij)
                print("Ants now - then")
                print(self._antsVertice, "-", self._oldAntsVertice)

        self._allBest = self._Space[np.argmin(self._verticesFitness)]
        self._allBestFitness = self._verticesFitness.min()
        print("ACO-All Best Response: {0}. Fitness: {1}".format(self._allBest, self._allBestFitness))

        self._ants_History = list(filter(lambda x: not x is None, self._ants_History))

        return self._allBest, self._allBestFitness