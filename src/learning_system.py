from plot_graphs import visualize_plots
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

        self.A = self.ideal_model.A
        self.B = self.ideal_model.B

        self.data_setup = DataCollectionSetup(self.real_model)


    def collect_data(self, visualize_data):
        self.dist_data_collection = self.data_setup.gather_data(K=5)
        
        if visualize_data:
            for k  in range(self.dist_data_collection.shape[0]):
                visualize_plots("Data collected", self.dist_data_collection[k,:,:])
                print "Data collected index :=", k
                raw_input("Press enter to see next")

    def compute_correction(self, visualize_data=False):

        '''
        this is the function you will have to implement.

        '''
        self.collect_data(visualize_data=visualize_data)

        model_correction = asmatrix(eye(4)*-2.234)

        return model_correction