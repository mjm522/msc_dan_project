from plot_graphs import visualize_plots
import numpy as np
import pandas as pd
from numpy import matrix, array, asmatrix, eye
from data_collection_setup import DataCollectionSetup


class LearningCorrection():

    def __init__(self, ideal_system, ideal_data, real_system):

        '''
        data structure is 
        10001 by 5
        column 1 =  cart position
        column 2 =  cart velocity
        column 3 =  pendulum position
        column 4 =  pendulum velocity
        column 5 =  control command
        '''

        self.ideal_data  = ideal_data
        self.ideal_model = ideal_system
        self.real_model  = real_system

        #This is the ideal
        self.A = self.ideal_model.A
        self.B = self.ideal_model.B


        #Trying to change it to the perturbed model with the disturbance
        self.A = self.real_model.A
        self.B = self.real_model.B        


        self.data_setup = DataCollectionSetup(self.real_model)


    def collect_data(self, visualize_data):
        self.dist_data_collection = self.data_setup.gather_data(K=5)
        #a = numpy.asarray(self.dist_data_collection)
        #This is how you access indivisual compnenets
        #visualize_plots("Data collected", self.dist_data_collection[k,:,:])
        #try something like this to access.....
        CartPosition = []
        CartVelocity = []
        PendulumPosition = []
        PendulumVelocity = []
        ControlCommand = []
        i =0
        for k  in range(self.dist_data_collection.shape[0]):    
            #print(self.dist_data_collection[k,0])
            CartPosition.append(self.dist_data_collection[k,0])
            CartVelocity.append(self.dist_data_collection[k,1])
            PendulumPosition.append(self.dist_data_collection[k,2])
            PendulumVelocity.append(self.dist_data_collection[k,3])
            ControlCommand.append(self.dist_data_collection[k,4])
            i = i + 1
        #print CartPosition
        #np.savetxt('cartPosition.txt', CartPosition)
        #np.savetxt('cartVelocity.txt', CartVelocity)
        #np.savetxt('pendulumPosition.txt', PendulumPosition)
        #np.savetxt('PendulumVelocity.txt', PendulumVelocity)
        #np.savetxt('controlCommand.txt', ControlCommand)
        
        


        if visualize_data:
            for k  in range(self.dist_data_collection.shape[0]):
                visualize_plots("Data collected", self.dist_data_collection[k,:,:])
                print "Data collected index :=", k
                raw_input("Press enter to see next")



    #self.dist_data_collection - this is the call for accessing the data 
    #need to use the following variables which i can then do some kind of learning on    
     # '''
      #  data structure is 
       # 10001 by 5
        #column 1 =  cart position
        #column 2 =  cart velocity
        #column 3 =  pendulum position
        #column 4 =  pendulum velocity
        #column 5 =  control command
        #'''

    def compute_correction(self, visualize_data=True):

       

        '''
        this is the function you will have to implement.

        '''
        self.collect_data(visualize_data=visualize_data)


       

        #This needs changing so as it implement a learning method
        model_correction = asmatrix(eye(4)) 
        #model_correction = asmatrix(eye(4)*300.234)
        #print(model_correction)

       # model_correction = [[300,0,0,0],
        #                     [0,300,0,0],
         #                    [0,0,300,0],
          #                   [0,0,0,300]]

       




        return model_correction