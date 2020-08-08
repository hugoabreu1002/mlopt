import numpy as np

class ACO(object):
    
    def __init__(self, antNumber, alpha, beta, rho, Q, dimentionsRanges, fitnessFunction, fitnessFunctionArgs):
        """
        antNumber : number of ants
        
        alpha : parameter for probabilities matrix
        
        beta : parameter for probabilities matrix
        
        rho : for pherormone
        
        Q : for pherormone
        
        dimentionsRanges : must be a list of iretables
    
        fitenessFunction : must be like, and returns a float from 0 to inf, the smaller means the better
            result = fitenessFunction(self.Space(self.antsVertice[k_ant]), *fitnessFunctionArgs)
            
        fitnessFunctionArgs : args Diferent than the antsVertice in Space.    
        """
        self.alpha = alpha
        self.beta = beta
        self.dimentionsRanges = dimentionsRanges
        self.rho = rho
        self.Q = Q
        self.antNumber = antNumber
        self.Dij = None
        self.Pij = None
        self.Tij = None
        self.Space = None
        self.antsVertice = None
        self.oldAntsVertice = None
        self.fitnessFunction = fitnessFunction
        self.fitnessFunctionArgs = fitnessFunctionArgs
    
    
    def setSpace(self):
        """
        Dimentions_Ranges: is a list of ranges. E.g:
            p = d = q = range(0, 2)
            Dimentions_Ranges = [p, d, q]
            
        The vertices of the grap will be a line of Space
        """
                
        rows = 1
        for d in self.dimentionsRanges:
            rows = len(d)*rows

        Space = np.zeros(rows, len(self.dimentionsRanges))

        for r in rows:
            for di, dobjtec in enumerate(self.dimentionsRanges):
                if (r > len(dobjtec)):
                    Space[r, di] = dobjtec[r%len(dobjtec)]
                else:
                    Space[r, di] = dobjtec[r]

        
        return Space    
    
    
    def initializeMatrices(self):    
        self.Space = self.setSpace()
        self.Dij = np.zeros((self.Space.shape[0], self.Space.shape[0]))
        self.Pif = np.zeros((self.Space.shape[0], self.Space.shape[0]))
        self.Tij = np.zeros((self.Space.shape[0], self.Space.shape[0]))
    
    
    def updateDij(self, Dij):
        """
        Dij and Pij will be only the matrix for the current possibilities
        Tij will be the pherormonen matrix for the whole graph
        
        fitnessFunction - lesser the better, so a good path should, Cj lesser than Ci
        Dij = Exp((Cj-Ci)/Ci)
        """
        
        for k_ant in range(self.antNumber):
            i = self.antsVertice[k_ant]
            j = np.random.choice(self.Space.shape[0])
            
            if Dij[i,j] == np.inf:
                Ci = self.fitnessFunction(self.Space(self.antsVertice[k_ant]), self.fitnessFunctionArgs)
                Cj = self.fitnessFunction(self.Space(j, self.fitnessFunctionArgs))
                Dij[i,j] = np.exp((Cj-Ci)/Ci)
        
        return Dij
    
                                          
    def updateTij(self, Tij, Dij, Ants, last_Ants, rho=0.5, Q=1):
    
        Dij_inf = Dij == np.inf

        sumdeltaTij = np.zeros(Tij.shape)

        for kij in zip(last_Ants, Ants):
            sumdeltaTij[kij] += Q/Dij[kij]

        Tij = (1-rho)*Tij + sumdeltaTij

        Tij[Dij_inf] = 0

        return Tij

                                          
    def updatePij(self, Pij, Tij, Dij, alpha=1, beta=1):
        Dij_inf = Dij == np.inf

        Pij = (Tij**alpha)/(Dij**beta)
        Pij[Dij_inf] = 0

        row_sums = Pij.sum(axis=1)
        Pij = Pij / row_sums[:, np.newaxis]

        Pij[Dij_inf] = 0 #lidar com ficar na mesma cidade

        return Pij

                                          
    def updateAntsPosition(self, Ants, Pij):
        last_Ants = Ants.copy()

        for i in range(Ants.shape[0]):
            k = Ants[i]

            possible_move = np.argwhere(Pij[k,:] > 0).flatten()
            weights = Pij[k,possible_move]/Pij[k,possible_move].sum()

            print("Ant {} possibilities:".format(i))
            print(possible_move)

            Ants[i] = np.random.choice(possible_move, p=weights)

            print("Ant {} move from {} to {}".format(i, k, Ants[i]))

        return Ants, last_Ants
    
                                          
    def search(self):
        
        self.initializeMatrices()
        
        self.antsVertice = np.zeros(self.antNumber)
        self.oldAntsVertice = np.zeros(self.antNumber)
        
        it = 0
        while(it<self.antNumber):
            print("iteration {0}".format(it))

            self.Dij = self.updateDij(self.Dij)
            print("Dij: ")
            print(self.Dij)

            self.Tij = self.updateTij(self.Tij, self.Dij, self.antsVertice, self.oldAntsVertice, self.rho, self.Q)
            print("Tij: ")
            print(self.Tij)

            Pij = self.updatePij(self.Pij, self.Tij, self.Dij)
            print("Pij:")
            print(Pij)

            self.antsVertice, self.oldAntsVertice = self.updateAntsPosition(self.antsVertice, self.Pij)
            print("Ants now - then")
            print(self.antsVertice, self.oldAntsVertice)

            it+=1