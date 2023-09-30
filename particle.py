import numpy as np

class Particle:

    def __init__(self, position, velocity ):
        '''position and velocity are vectors (numpy array)'''

        self.position = position
        self.velocity = velocity
        self.bestp = position

        self.iteration = 0


    def FitnessCalculator(self, position, accuracy):
        '''Computes the fitness related to a specific position, update the particle fitness with it and returns it.
           position: position of the particle. It must be a numpy array made with floats.
           accuracy: function that computes the fitness of a particle. It takes position as argument and returns the relative fitness (a float value)'''

        self.fitness = accuracy(position)
        return self

    def inertia_coefficient(self, c1, c2, random_1, random_2, max_iter = None, old_w = None, schedule_type = 'constant'):
        '''Computes the inertia coefficient (typically named w), necessary for updating the particle velocity.
           c1: positive costant value. Float type
           c2: positive constant value. Float type
           random_1: random float value
           random_2: random float value
           max_iter: (Optional) maximum number of iteration. Integer type
           old_w: old inertia coefficient value
           schedule_type: strategy you want to use to compute the inertia coefficient in PSO.
           It can be <<constant>>, <<random>>, <<linearly decreasing>>'''

        min_w = (c1+c2)*(1/2)-1
        
        if schedule_type == 'costant':
            # w is set to be constant
            w = c1*random_1 + c2*random_2

        if schedule_type == 'random':
            # w is randomly selected ad each iteration from a gaussian distribution with center 0.72 and σ small enough to ensure that w is not predominantly greater than one
            w = np.random.normal(0.72, 0.4)

        if schedule_type == 'linearly decreasing':
            if max_iter != None:
                w = ((0.9-min_w)*(max_iter-self.iteration)/max_iter) + min_w
            else: 
                raise Exception('ERROR YOU MUST SPECIFY THE MAXIMUM NUMBER OF ITERATION')
        
        if schedule_type == 'nonlinearly decreasing':
            if old_w != None:
                w = 0.975*old_w
            else: 
                raise Exception('ERROR YOU MUST SPECIFY W AT PREVIOUS ITERATION')
        
        if schedule_type not in ['constant','random','linearly decreasing', 'nonlinearly decreasing']:
            raise Exception('You must specify a valid type for w')

        # w > min_w guarantees convergent particle trajectories. If this condition is not satisfied, divergent or cyclic behavior may occur.
        if w < min_w:
            w = min_w
        return w


    def VelocityCalculator(self, c1, c2, best_glob_pos, w_schedule, w = 0.9, v_max = None):
        '''Computes and update the particle velocity.
           c1: positive costant value. Float type
           c2: positive constant value. Float type
           best_glob_pos: numpy array
           w_schedule: strategy you want to use to compute the inertia coefficient in PSO.
                       String object
           w: positive constant value. Float type (is the starting value for w)
           v_max: vector (each element is the maximum velocity for that dimension)'''

        random_1 = np.random.random(len(self.position))
        random_2 = np.random.random(len(self.position))

        velocity = w*self.velocity + c1*random_1*(self.bestp - self.position) + c2*random_2*(best_glob_pos - self.position)

        # velocity quickly explodes to large values, especially for particles far from the neighborhood best and personal best positions. Consequently, particles have large position updates, which result in particles leaving the boundaries of the search space – the particles diverge. To control the global exploration of particles, velocities are clamped to stay within boundary constraints.

        if v_max != None:
            new_velocity = np.zeros(len(velocity)) #I inizialize an array with all zeros and then I change the elements

            for i in range (len(velocity)):
                if velocity[i] > v_max[i]:
                    new_velocity[i] = v_max[i]
                else:
                    new_velocity[i] = velocity[i]
            self.velocity = new_velocity

        else:
            self.velocity = velocity
        
        #let's update w
        w = self.inertia_coefficient(c1, c2, random_1, random_2, old_w = w, schedule_type = w_schedule)

        return self

    def BoundaryConstraints(self, lower_bound, upper_bound, scheme = 'reflecting'):
        '''This function calculates the new position of a particle when the actual position is outside the boundary.
        Three main scheme are take in account: 
        1) random:  if a particle flies outside of the boundary of a parameter, a random value drawn from a uniform distribution between the lower and upper boundaries of the parameter is assigned.
        2) absorbing: a particle flying outside of a parameter’s boundary is relocated at the boundary in that dimension.
        3) reflecting: when a particle flies outside of a boundary of a parameter, the boundary acts like a mirror and reflects the projection of the particle’s displacement'''
        
        new_position = []

        # we iterate for each component of the position
        for dim, lower, upper in zip(self.position, lower_bound, upper_bound):
            
            while ((dim < lower) or (dim > upper)): #check if in that dimension the position component is outside of the boundary

                if scheme == 'random':
                    dim = np.random.uniform(lower, upper)

                elif scheme == 'absorbing':
                    if dim < lower:
                        dim = lower
                    else:
                        dim = upper
                
                elif scheme == 'reflecting':
                    if dim < lower:
                        dim = (lower - dim) + lower
                    else:
                        dim = upper - (dim-upper)
                
            new_position.append(dim)
        self.position = np.array(new_position)

    def BestLocal(self, problem):
        '''Takes as input the particle and the type of optimization problem (problem could be minimum or maximum) and calculates best fitness and best position'''
        if self.iteration == 0:
            self.bestfit = self.fitness

        if problem == 'minimum':
            if self.fitness < self.bestfit:
                self.bestfit = self.fitness
                self.bestp = self.position
        elif problem == 'maximum':
            if self.fitness > self.bestfit:
                self.bestfit = self.fitness
                self.bestp = self.position
        else:
            return "Error! problem must be: 'minimum' or 'maximum'"    
        return self

    def PositionCalculator(self, lower_bound, upper_bound, evaluation_funct, problem_type):
        '''Calculates the new position and the relative fitness and in case update the new best local position.

        lower_bound: is a vector in which the i-th element is the lower bound of the position for the i-th dimension
        upper_bound: is a vector in which the i-th element is the upper bound of the position for the i-th dimension
        evaluation_funct: is the function used for evaluating the goodness of the position
        problem_type: can be 'minimum' or 'maximum'.'''
        
        self.iteration += 1

        # First we calculate the new position
        self.position = self.position + self.velocity

        #Then we check if the new_position is inside the boundaries
        self.BoundaryConstraints(lower_bound, upper_bound)
        
        # We need to calculate the fitness function for the new position
        self.FitnessCalculator(self.position, evaluation_funct)

        # With the new position calculated we have to update the local best position:
        self.BestLocal(problem_type)

        return self