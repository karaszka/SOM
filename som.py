#Karolina Stosio

import numpy as np
import random
from math import exp, log


class onlineSOM():
    '''
    Online implementation of SOM algorithm.
    The NxX map is created based on N^2 first samples,
    then the regular SOM algorithm begins.
    '''

    
    def __init__(self, size, dim):
        '''
        This method creates a map vector. 
        The map is 1D vector (flattened square 2D map of a given size).
        In order to bring it to the 2D shape for the visualisation purpouse
        call the reshape_map method.
        
        '''
        self.size = size
        self.dim = dim
        self.somap = np.zeros((size**2,dim),dtype='float64')
        self.iteration = 0
        self.init_eta = 1
        self.init_sigma = 5
        self.eta = self.init_eta
        self.sigma = self.init_sigma
        self.samples = []
        self.errors = []
        self.maps_log = {} #maps stored here are already reshaped
       
   
    def reshape(self):
        '''
        Returns 2D version of the map
        '''
        return self.somap.reshape((self.size,self.size,self.dim))
        
    
    def update(self, sample):
        '''
        Updating a map.
        If the whole map has been initialized - do the regular SOM step
        else use the new sample as a prototype.
        '''
        
        sample = sample.astype('float64')
        
        if self.iteration%100==0:
            self.maps_log[str(self.iteration)]=np.copy(self.somap.reshape((self.size,self.size,self.dim)))
        
        if self.iteration < self.size**2:
            self.errors.append(0)
            self.somap[self.iteration] = sample
        else:
            self.samples.append(sample)
            self.errors.append(np.min(np.sqrt(np.sum((sample - self.somap)**2, axis=1))))
            self._update(sample)
            
        self.iteration += 1  
       
    
    def q_lookup(self, index):
        '''
        Computing the position of the 1D index in 2D kxk space
        '''
        
        return np.array([index/self.size, index%self.size])
    
    
    def h_vec(self, q, p):
        '''
        Neighbourhood function
        '''
        
        return exp(-((q-p).dot(q-p))/(2*self.sigma**2))    
    
    
    def _update(self, sample):
        '''
        The update of the SOM - main part
        finding the closest prototype
        update of the whole map
        '''
        
        error = np.argmin(np.sum((sample - self.somap)**2, axis=1))
        p_vec = self.q_lookup(error)
        
        for q in range(self.size**2):
            q_vec = self.q_lookup(q)
            temp_diff = (sample - self.somap[q]) # this accounts for the distance in the samples space
            temp_h =  self.h_vec(q_vec, p_vec)             # this accounts for the distance in the map space
            update = self.eta * temp_h * temp_diff
            self.somap[q] += update.astype('float64')
            
        
        self.sigma *= 0.9995
        self.eta *= 0.9997


class onlineSOM_learning(onlineSOM):
    '''
    Online implementation of SOM algorithm.
    The NxX map is created based on N^2 first samples,
    than the regular SOM algorithm begins.
    In case the distance between new sample and a prototype
    is higher than the 95th percentile of the approx error distribution.
    '''

    
    def __init__(self, size, dim, hard_reset=False):
        '''
        this method creates a map vector. the map is 1D vector 
        (flattened square 2D map of a given size)
        in order to bring it to the 2D shape for the visualisation purpouse
        call the reshape_map method
        
        '''
        
        super(onlineSOM_learning,self).__init__(size,dim)
        if hard_reset:
            self.restart_params=self.restart_params_hard
        else:
            self.restart_params=self.restart_params_soft
        self.err_condition = []
        self.T = 5 * size**2
    
    def restart_params_soft(self):
        '''
        Restarts initial parameters, so that map is plastic again.
        '''
        self.eta = self.init_eta
        self.sigma = self.init_sigma    
    
    def restart_params_hard(slef):
        '''
        Restarts initial parameters, so that map is plastic again.
        Restarting the iteration attr will cause the reinitialization of the map with new samples
        '''
        self.restart_params_soft()
        self.iteration = 0
      
    def error_condition(self,error):
        '''
        Constructs and checks the error condition
        '''
        window = signal.gaussian(6, std=2)
        percentile = np.percentile(np.convolve(self.errors[-self.T:-1], window, mode='full'),95) 
        self.err_condition.append(percentile)
            
        if percentile<error: 
            return True
        else:
            return False
    
    def update(self, sample):
        '''
        if the whole map has been initialized - do SOM step
        else use the new sample as a prototype.
        the SOM step is divided such that the new error is first computed 
        and the error condition is checked before the update. if the error condition yeilds True,
        initial parameters are restarted according to the policy selected 
        at the initialization of the class object.
        '''
        sample = sample.astype('float64')
        
        if self.iteration%100==0:
            self.maps_log[str(self.iteration)]=np.copy(self.somap.reshape((self.size,self.size,self.dim)))
        
        
        if self.iteration < self.size**2:
            self.somap[self.iteration] = sample
            self.errors.append(0)
            self.err_condition.append(0)
        else:
            self.samples.append(sample)
            min_error = np.min(np.sqrt(np.sum((sample - self.somap)**2, axis=1)))
            if self.iteration > self.size**2 + self.T:    
                if self.error_condition(min_error):
                    self.restart_params()
                    print('parameters restarted at iteration ',self.iteration)
            else:
                self.err_condition.append(0)
            self.errors.append(min_error)
            self._update(sample)

        self.iteration += 1
     
    
    
    def _update(self, sample):
        '''
        the update of the SOM - main part
        finding the closest prototype
        update of the whole map
        '''
        
        error = np.argmin(np.sum((sample - self.somap)**2, axis=1))
        p_vec = self.q_lookup(error)
        for q in range(self.size**2):
            q_vec = self.q_lookup(q)
            temp_diff = (sample - self.somap[q]) # this accounts for the distance in the samples space
            temp_h =  self.h_vec(q_vec, p_vec)             # this accounts for the distance in the map space
            update = self.eta * temp_h * temp_diff
            self.somap[q] += update.astype('float64')
        self.sigma *= 0.9997
        self.eta *= 0.9997
                

class UbiSOM(onlineSOM):
    '''
    Implementation of UbiSOM algorithm without the normalisation.
    Algorithm as described in 
    Silva 2015: The ubiquitous self-organizing map for non-stationary data streams.
    The NxX map is created based on N^2 first samples,
    than the regular UbiSOM algorithm with begins: 
    ordering phase that transist to the learning phase, etc.
    '''


    def __init__(self, size, dim, T=2000):
        '''
        This method creates a map vector. 
        The map is 1D vector (flattened square 2D map of a given size).
        In order to bring it to the 2D shape for the visualisation purpouse
        call the reshape_map method.
        '''

        super(UbiSOM, self).__init__(size,dim)
        
        # this variables are used to calculate the drift
        self.T = T
        self.generalization_errors = []
        self.activities = np.zeros((size**2,T),dtype='float64')
        self.drift_values = []
        
        # flags and counters for switching between ordering/learning phases
        self.iteration_ordering = 0
        self.iteration_learning = 0
        self.ordering = True
        
        # values as in Silva'15
        self.dist = np.sqrt(2)*(size-1)
        self.eta_0 = 0.1
        self.eta_f = 0.01
        self.sigma_0 = 2*np.sqrt(size)
        self.sigma_f = 1
        self.neighborhood_threshold = 0.01
        self.beta = 0.7
        self._init_learning_params()
        
    def _init_learning_params(self):
        '''
        Learning parameters annealing scheme for the ordering phase (as proposed in Silva'15)
        '''
        time = np.arange(1,self.T)
        self.sigma_annealing = self.sigma_0*(self.sigma_f/self.sigma_0)**(time/self.T)
        self.eta_annealing = self.eta_0*(self.eta_0/self.eta_0)**(time/self.T)
    
    def update(self, sample):
        '''
        if the whole map has been initialized - go to ubiSOM
        else use the new sample as a prototype.
        ubiSOM consists of ordering phase that lasts for T samples 
        and the learning phase that continues until the newly computed error exeeds the deafult value
        '''
        
        sample = sample.astype('float64')

        if self.iteration%100==0: #just for the purpouse of visualizing the nodes
            self.maps_log[str(self.iteration)]=np.copy(self.somap.reshape((self.size,self.size,self.dim)))
            
        if self.iteration < self.size**2:
            
            self.somap[self.iteration] = sample
            self.generalization_errors.append(0)
        
        elif self.ordering:                  #the ordering step
            
            self.eta =  self.eta_annealing[self.iteration_ordering]
            self.sigma = self.sigma_annealing[self.iteration_ordering]
            self.iteration_ordering += 1
            
            if self.iteration_ordering == self.T-1:
                self.ordering = False
                self.learning_index = 0
                self._update_drift()
                
            self.samples.append(sample)
            min_error = np.min(np.sum((sample - self.somap)**2, axis=1))
            self.generalization_errors.append(min_error)
            self._update(sample)
            
        else:                                #the learning step
            
            #set up learning parameters
            if self.drift_values[0] > self.drift_values[-1]:
                self.eta =  self.eta_f * self.drift_values[-1]/self.drift_values[0]
                self.sigma = self.sigma_f * self.drift_values[-1]/self.drift_values[0]
            else:
                self.eta =  self.eta_f
                self.sigma = self.sigma_f    
                
            self.samples.append(sample)
            min_error = np.min(np.sqrt(np.sum((sample - self.somap)**2, axis=1)))
            self.generalization_errors.append(min_error)
            self._update(sample)
            self._update_drift()
            
            # check for drastical distribution changes
            if self.iteration_learning > self.T-1 and np.sum(self.drift_values[-self.T:]>self.T):
                self.ordering = True
                self.ordering_index = 0
                    
        self.iteration += 1
    
    def _update_drift(self):
        mean_utility = np.sum(self.activities, axis=1)
        mean_utility[mean_utility!=0] = 1
        mean_utility = np.sum(mean_utility)/self.T
        mean_error = np.sum(self.generalization_errors[-self.T:])/self.T
        d = self.beta*mean_error + (1-self.beta)*(1-mean_utility)
        self.drift_values.append(d) 
    
    def _update(self, sample):
        '''
        the update of the SOM - main part
        finding the closest prototype
        update of the whole map
        '''
        error = np.argmin(np.sum((sample - self.somap)**2, axis=1))
        p_vec = self.q_lookup(error)
        activities = np.zeros(self.size**2)
        
        for q in range(self.size**2):
            q_vec = self.q_lookup(q)
            temp_diff = (sample - self.somap[q]) # this accounts for the distance in the samples space
            temp_h = self.h_vec(q_vec, p_vec)        # this accounts for the distance in the map space
            
            if temp_h<self.neighborhood_threshold:
                temp_h = 0
            else:
                activities[q] = 1
            
            update = self.eta * temp_h * temp_diff
            self.somap[q] += update.astype('float64')
        
        self.activities = np.concatenate([self.activities[:,1:],activities[:,None]],axis=1) #the first (oldest) activity is discarded
                                                                                  #the recent value is stacked on the other end
            

class PLSOM(onlineSOM):
    '''
    Online implementation of PLSOM algorithm.
    From The Parameterles Self Organising Map.
    Learning rate and neighbourhood function are calculated based on errors.
    The NxX map is created based on N^2 first samples,
    than the regular algorithm begins.
    '''

    
    def __init__(self, size, dim, beta=0.93,t=2):
        '''
        This method creates a map vector. 
        The map is 1D vector (flattened square 2D map of a given size).
        In order to bring it to the 2D shape for the visualisation purpouse
        call the reshape_map method.
        
        '''
        super(PLSOM,self).__init__(size,dim)
        self.beta = beta
        self.r = 0
        self.epsilon = 0
        self.errors=[]
        
        if t==1: 
            self.theta=self.theta1
            self.theta_min = 1
        if t==2: 
            self.theta=self.theta2
            self.theta_min = 1
        if t==3: 
            self.theta=self.theta3
            self.theta_min = 0
            
    def theta1(self):
        return self.beta*self.epsilon
    
    def theta2(self):
        return (self.beta-self.theta_min)*self.epsilon+self.theta_min
    
    def theta3(self):
        return (self.beta-self.theta_min)*np.log(1+self.epsilon*(np.e-1))+self.theta_min

    def h_vec(self, q, p):
        '''
        neighbourhood function
        '''
        
        return np.exp(-((q-p).dot(q-p))/(self.theta()**2))   
    
    
    def update(self, sample):
        '''
        if the whole map has been initialized - do SOM step
        else use the new sample as a prototype.
        the SOM step is divided such that the new error is first computed 
        and the error condition is checked before the update. if the error condition yeilds True,
        initial parameters are restarted according to the policy selected 
        at the initialization of the class object.
        '''
        sample = sample.astype('float64')
        
        if self.iteration%100==0:
            self.maps_log[str(self.iteration)]=np.copy(self.somap.reshape((self.size,self.size,self.dim)))
        
        if self.iteration < self.size**2:
            self.somap[self.iteration] = sample
            self.errors.append(0)
        else:
            self.samples.append(sample)
            min_error = np.min(np.sqrt(np.sum((sample - self.somap)**2, axis=1)))
            self.errors.append(min_error)
            
            if min_error>self.r: 
                self.r=min_error
            
            self._update(sample)

        self.iteration += 1
     
    
    def _update(self, sample):
        '''
        the update of the SOM - main part
        finding the closest prototype
        update of the whole map
        '''
        
        error = np.argmin(np.sum((sample - self.somap)**2, axis=1))
        er = np.min(np.sqrt(np.sum((sample - self.somap)**2, axis=1)))
        p_vec = self.q_lookup(error)
        self.r = np.max((self.r,er))
        self.epsilon = er/self.r
        
        for q in range(self.size**2):
            q_vec = self.q_lookup(q)
            temp_diff = (sample - self.somap[q]) # this accounts for the distance in the samples space
            temp_h =  self.h_vec(q_vec, p_vec)       # this accounts for the distance in the map space
            update = self.epsilon * temp_h * temp_diff
            self.somap[q] += update.astype('float64')


