from numpy import matrix, array, asmatrix, eye


class LearningCorrection():

    def __init__(self, ideal_system_model, ideal_data, disturbed_data):

        self.ideal_data = ideal_data
        self.disturbed_data = disturbed_data
        self.model = ideal_system_model

        self.A = self.model.A
        self.B = self.model.B


    def compute_correction(self):

        '''
        this is the function you will have to implement.
        
        '''

        return asmatrix(eye(4)*-2.234)