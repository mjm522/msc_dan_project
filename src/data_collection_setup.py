from numpy import pi, array, zeros, random
from cart_pole import CartPole
random.seed(0)

class DataCollectionSetup():

    def __init__(self, system_model):
        self.system_model = system_model
        self.state_dim = self.system_model.state_dim
        self.ctrl_dim = self.system_model.ctrl_dim

    def get_peturbation(self):
        '''
        this could be even made better , a better type of exploration
        can be implemented.
        '''
        return random.rand(self.state_dim,self.state_dim)

    def gather_data(self, K):
        data = zeros([K, 10001, self.state_dim+self.ctrl_dim])

        for k in range(K):
            self.system_model.reset()
            self.system_model.peturb_model(self.get_peturbation())
            self.system_model.compute_lqr_gain()
            data[k,:,:] = self.system_model.integrate()

        return data

