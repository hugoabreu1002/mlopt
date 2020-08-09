import numpy as np
from tqdm import tqdm
from scipy.stats import mode

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

    def __init__(self, antNumber, alpha, beta, rho, Q, dimentionsRanges, fitnessFunction, fitnessFunctionArgs=None):
        self._alpha = alpha
        self._beta = beta
        self._dimentionsRanges = dimentionsRanges
        self._rho = rho
        self._Q = Q
        self._antNumber = antNumber
        self._Dij = None
        self._Pij = None
        self._Tij = None
        self._Space = None
        self._antsVertice = None
        self._oldAntsVertice = None
        self.fitnessFunction = fitnessFunction
        if fitnessFunctionArgs is None:
            fitnessFunctionArgs = []
        self._fitnessFunctionArgs = fitnessFunctionArgs
    
    
    def setSpace(self):
        """
        Dimentions_Ranges: is a list of ranges. E.g:
            p = d = q = range(0, 2)
            Dimentions_Ranges = [p, d, q]
            
        The vertices of the grap will be a line of Space
        """
                
        rows = 1
        for d in self._dimentionsRanges:
            rows = len(d)*rows

        print("dimentions Ranges passed: ", self._dimentionsRanges)
        print("number o Space rows: ", rows)
        
        Space = np.zeros((rows, len(self._dimentionsRanges)))

        for r in range(rows):
            for di, dobjtec in enumerate(self._dimentionsRanges):
                if (r >= len(dobjtec)):
                    Space[r, di] = dobjtec[r%len(dobjtec)]
                else:
                    Space[r, di] = dobjtec[r]

        return Space    
    
    
    def initializeMatricesAndAntsPosition(self):    
        self._Space = self.setSpace()
        self._Dij = 1/np.zeros((self._Space.shape[0], self._Space.shape[0])) #np.ones((self._Space.shape[0], self._Space.shape[0]))#
        self._Pif = np.ones((self._Space.shape[0], self._Space.shape[0]))
        self._Tij = np.ones((self._Space.shape[0], self._Space.shape[0]))
        
        self._antsVertice = np.random.choice(list(range(self._Space.shape[0])), size=self._antNumber)
        self._oldAntsVertice = np.zeros(self._antNumber, dtype=int)
    
    
    def updateDij(self, Dij):
        """
        Dij and Pij will be only the matrix for the current possibilities
        Tij will be the pherormonen matrix for the whole graph
        
        fitnessFunction - lesser the better, so a good path should, Cj lesser than Ci
        Dij = Exp((Cj-Ci)/Ci)
        """
        
        for k_ant in range(self._antNumber):
            i = self._antsVertice[k_ant]
            j = np.random.choice(list(range(self._Space.shape[0])))

            if Dij[i,j] == np.inf: #np.inf
                Ci = self.fitnessFunction(self._Space[self._antsVertice[k_ant], :], self._fitnessFunctionArgs)
                Cj = self.fitnessFunction(self._Space[j, :], self._fitnessFunctionArgs)
                Dij[i,j] = np.exp((Cj-Ci)/Ci)
                Dij[j,i] = Dij[i,j]
        
        return Dij
    
                                          
    def updateTij(self, Tij, Dij, Ants, last_Ants, rho=0.5, Q=1):
        Dij_inf = Dij == np.inf
        Dij_notinf = Dij != np.inf
        Dij[Dij_inf] = Dij_notinf.max()

        sumdeltaTij = np.zeros(Tij.shape)

        for kij in zip(last_Ants, Ants):
            sumdeltaTij[kij] += Q/Dij[kij]

        Tij = (1-rho)*Tij + sumdeltaTij

        Tij += np.random.randint(1, size=Tij.shape)/10

        return Tij

                                          
    def updatePij(self, Pij, Tij, Dij, alpha=1, beta=1):
        Dij_inf = Dij == np.inf
        Dij_notinf = Dij != np.inf
        Dij[Dij_inf] = Dij_notinf.max()

        Pij = (Tij**alpha)/(Dij**beta)
        Pij += np.random.randint(1, size=Pij.shape)/10
        
        row_sums = Pij.sum(axis=1)
        Pij = Pij / row_sums[:, np.newaxis]
        
        return Pij

                                          
    def updateAntsPosition(self, Ants, Pij, verbose=False):
        last_Ants = Ants.copy()

        for i in range(Ants.shape[0]):
            k = Ants[i]

            possible_move = np.argwhere(Pij[k,:] > 0).flatten()
            weights = Pij[k,possible_move]/Pij[k,possible_move].sum()
            Ants[i] = np.random.choice(possible_move, p=weights)

            if verbose:
                print("Ant {} possibilities:".format(i))
                print(possible_move)
                print("Ant {} move from {} to {}".format(i, k, Ants[i]))

        return Ants, last_Ants
    
                                          
    def search(self, verbose=False):
        
        self.initializeMatricesAndAntsPosition()
        self.antsVertice = np.zeros(self._antNumber)
        self.oldAntsVertice = np.zeros(self._antNumber)
                
        for it in tqdm(range(self._antNumber)):
            self._Dij = self.updateDij(self._Dij)
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
            self._antsVertice, self._oldAntsVertice = self.updateAntsPosition(self._antsVertice, self._Pij, verbose)
            if verbose:
                print("Dij: ")
                print(self._Dij)
                print("Tij: ")
                print(self._Tij)
                print("Pij:")
                print(self._Pij)
                print("Ants now - then")
                print(self._antsVertice, self._oldAntsVertice)

        print("Most Frequent Response")
        print(self._Space[mode(self._antsVertice)[0][0],:])
        print("Minimun Found")
        print(self.fitnessFunction(self._Space[mode(self._antsVertice)[0][0],:], self._fitnessFunctionArgs))