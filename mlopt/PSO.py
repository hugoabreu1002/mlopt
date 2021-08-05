import numpy as np
from tqdm import tqdm
import random
import math    # cos() for Rastrigin
import copy    # array-copying convenience
import sys     # max float

class particle(object):
    """
    fitenessFunction : must be like, and returns a float from 0 to inf, the smaller means the better
        result = fitenessFunction(self.Space(self.antsVertice[k_ant]), *fitnessFunctionArgs)
        
    fitnessFunctionArgs : args Diferent than the antsVertice in Space.    
    """
    def __init__(self, dim, minx, maxx, seed, function, functionArgs=[], verbose=False):
        self.rnd = random.Random(seed)
        self.position = [0.0]*dim
        self.velocity = [0.0]*dim
        self.best_part_pos = [0.0]*dim
        self.fitnessFunction = function
        self._fitnessFunctionArgs = functionArgs

        for i in range(dim):
            self.position[i] = ((maxx - minx) * self.rnd.random() + minx)
            self.velocity[i] = ((maxx - minx) *self.rnd.random() + minx)

        self.error = self.error_func() # curr error
        self.best_part_pos = copy.copy(self.position) 
        self.best_part_err = self.error # best error

    def error_func(self):
        self.error = self.fitnessFunction(self.position, self._fitnessFunctionArgs)
        return self.error


class swarm:
    def __init__(self, number_of_particles, dim, minx, maxx, seed, function, functionArgs=[], verbose=False):
        self.dim = dim
        self.minx = minx
        self.maxx = maxx
        self.seed = seed
        self.function = function
        self.functionArgs = functionArgs
        self.best_swarm_pos = None
        self.best_swarm_err = sys.float_info.max
        self.number_of_particles = number_of_particles
        self.swarm = self.initializeSwarm()

    def initializeSwarm(self):
        Particle = particle(self.dim, self.minx, self.maxx, self.seed, self.function, self.functionArgs, verbose=False)
        particles = [Particle for i in range(self.number_of_particles)]
        return particles

    def findGlobalBestInSwarm(self):
        for i in range(self.number_of_particles): # check each particle
            if self.swarm[i].error < self.best_swarm_err:
                self.best_swarm_err = self.swarm[i].error
                self.best_swarm_pos = copy.copy(self.swarm[i].position) 

    def getDistanceBetweenParticles(self, p1, p2):
        return sum([np.sqrt((p1.position[d]-p2.position[d])**2) for d in range(self.dim)])

    def getLocalNeighbors(self, target_particle, N_neighbors):
        distances_particle_list = []
        for pts in self.swarm:
            distances_particle_list.append((pts, self.getDistanceBetweenParticles(target_particle, pts)))
        
        distances_particle_list = sorted(distances_particle_list, key=lambda x: x[1])

        return list(map(lambda tup_p_d: tup_p_d[0], distances_particle_list[1:]))

    def findLocalBestForParticle(self, particle, numberOfClusters):
        local_swarm = self.getLocalNeighbors(particle, numberOfClusters)
        best_local_pos = local_swarm[0].position
        local_best_swarm_err = local_swarm[0].error
        
        for i, pt in enumerate(local_swarm): # check each particle
            if pt.error < local_best_swarm_err:
                local_best_swarm_err = pt.error
                best_local_pos = copy.copy(pt.position) 
        
        return best_local_pos

class PSO(swarm):
    def __init__(self, number_of_particles, dim, minx, maxx, seed, function, functionArgs=[], verbose=False):
        """
        number_of_particle - number of particles to chose
        dim - dimention of the problem, space of search
        minx - bottom boundary for the space of search
        maxx - topper boundary for the space of search
        seed - random seed generator
        function - function to be optimized (must be of sabe dimention as dim)
        functionArgs - Extra arguments for the function to be optimized
        """
        super(PSO, self).__init__(number_of_particles, dim, minx, maxx, seed, function, functionArgs=[], verbose=False)
        self.historic_best_pos = []
        self.historic_best_error = []

    def update_particle_velocity(self, p_v, p_best, p_pos, g_best, c1, r1, c2, r2, w):
        return (w * p_v) + (c1 * r1 * (p_best - p_pos)) +  (c2 * r2 * (g_best - p_pos)) 

    def Solver(self, max_epochs, plot_at_every=50, w=0.8, c1=2.05, c2=2.05, topology='G'):
        """
        max_epochs - the number of epochs to search
        w - the inertial coefficient. It must be under 1, and float or tuple type. If tuple, (wi, wf), the algorithm will linearly decay the inertial coefficient over the epochs til the final value.
        c1 - the cognitive coefficient
        c2 - the social coefficient
        topology - choose between chars 'G':"Global" 'L':"Local or Ring" 'F':"Focal or Wheel".
        """
        
        self.historic_best_error = []

        if isinstance(w, tuple):
            w_array = np.arange(w[0], w[1], (w[1]-w[0])/max_epochs)
        elif isinstance(w, float):
            if w > 1 or w < 0:
                raise Exception("w must be lower than 1 and bigger than 0")
            else:
                w_array = [w]*max_epochs
        
        if topology == 'G':
            foo = "bar"
        elif topology == 'F':
            foo = "bar"
        elif topology == 'L':
            numberOfClusters = int(self.number_of_particles/3)
        else:
            raise Exception("Wrong char for topology choice, please choose between chars 'G':'Global' 'L':'Local or Ring' 'F':'Focal or Wheel'.")
        
        plot_follow = plot_at_every
        for epoch in tqdm(range(max_epochs)):
            # Topology
            self.findGlobalBestInSwarm()
            self.historic_best_error.append(self.best_swarm_err)
            self.historic_best_pos.append(self.best_swarm_pos)

            for i in range(self.number_of_particles): # process each particle
                # compute new velocity of curr particle
                pt = self.swarm[i]
                
                # Topology
                if topology == 'G':
                    reference_pos = copy.copy(self.best_swarm_pos)
                elif topology == 'F':
                    reference_pos = copy.copy(self.swarm[0].position)
                elif topology == 'L':
                    reference_pos = copy.copy(self.findLocalBestForParticle(pt, numberOfClusters))

                for k in range(self.dim): 
                    r1 = pt.rnd.random()    # randomizations
                    r2 = pt.rnd.random()
                
                    pt.velocity[k] = self.update_particle_velocity(pt.velocity[k], pt.best_part_pos[k], pt.position[k], reference_pos[k], c1,r1,c2,r2, w_array[epoch])

                    if pt.velocity[k] < self.minx:
                        pt.velocity[k] = self.minx
                    elif pt.velocity[k] > self.maxx:
                        pt.velocity[k] = self.maxx

                # compute new position using new velocity
                for k in range(self.dim): 
                    pt.position[k] += pt.velocity[k]
            
                # compute error of new position
                pt.error_func()

                # is new position a new best for the particle?
                if pt.error < pt.best_part_err:
                    pt.best_part_err = pt.error
                    pt.best_part_pos = copy.copy(pt.position)

                # update swarm
                self.swarm[i] = copy.copy(pt)   
                # is new position a new best overall?
                if self.swarm[i].error < self.best_swarm_err:
                    self.best_swarm_err = self.swarm[i].error
                    self.best_swarm_pos = copy.copy(self.swarm[i].position) 

            # self.findGlobalBestInSwarm()
            # self.historic_best_error.append(self.best_swarm_err)
            # self.historic_best_pos.append(self.best_swarm_pos)
            
            if epoch > plot_follow:
                print("Epoch: {0}, best error: {1:.3f}, best pos: {2}".format(epoch, self.best_swarm_err, self.best_swarm_pos))
                plot_follow += plot_at_every
                
        return self
