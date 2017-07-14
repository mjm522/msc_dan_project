from numpy import matrix, array, asmatrix, eye


class LearningCorrection():

    def __init__(self, ideal_system_model, ideal_data, disturbed_data):

        '''
        data structure is 
        100001 by 5
        column 1 =  cart position
        column 2 =  cart velocity
        column 3 =  pendulum position
        column 4 =  pendulum velocity
        column 5 =  control command
        '''

        self.ideal_data = ideal_data
        self.disturbed_data = disturbed_data
        self.model = ideal_system_model

        self.A = self.model.A
        self.B = self.model.B


    def compute_correction(self):

        '''
        this is the function you will have to implement.

        '''

        model_correction = asmatrix(eye(4)*-2.234)

        return model_correction