import numpy as np
import scipy.linalg as la

np.random.seed(0)

'''
Install SCIPY library to make this code work
pip install scipy
'''

'''
This funciton implements a discretized lqr controller
it takes in system matrix A, the control matrix B
The state penalization Q, and the control penalization R
It formualates the ricatti equation and solves it
Please read the materials given in the doc folder
'''

def dlqr(A, B, Q, R):
    """Solve the discrete time lqr controller.
    
    x[k+1] = A x[k] + B u[k]
     
    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    #ref Bertsekas, p.151
 
    #first, try to solve the ricatti equation
    X = la.solve_discrete_are(A, B, Q, R)
     
    #compute the LQR gain
    K = la.pinv( np.dot( (np.dot(np.dot(B.T, X), B) + R) , np.dot(np.dot(B.T, X), A) ).T )

    eigVals, eigVecs = la.eig( A - np.dot(B,K) )
     
    return K, eigVals


'''
This is a class that encapsulates the dynamics of a point mass object.
It has the A matrix, B matrix 
It also has an option to compute the non_linear_term when a configuration is passed
the compute_nxt_state funciton computes the next state if the current state and control is passed
the state q = [x, y, dx, dy]
the control  u = [Fx, Fy]
'''

class Dynamics():

    def __init__(self, dt=0.01):
        '''
        the system matrix
        '''
        self.A = np.array([[1.,0.,dt, 0.],
                           [0.,1.,0., dt],
                           [0.,0.,1., 0.],
                           [0.,0.,0., 1.]])
        '''
        the control matrix
        '''
        self.B = np.array([[0.,0.],
                           [0.,0.],
                           [1.,0.],
                           [0.,1.]])

    def non_linear_term(self, q):
        return q + np.random.randn(4)


    def compute_nxt_state(self, q, u, disturb=False):

        q = np.dot((np.eye(4) + self.A), q) + np.dot(self.B,u)

        if disturb:
            q += self.non_linear_term(q)

        return q



def compute_ctrl(q0, qf, dynamics, correction_dynamics=None):
    '''
    find a control command using the start, final and the dynamics
    at present it is not doing anything, just returning a random control input
    so initially the correction dyanmics is none and later it learns it
    The state penalization Q, and the control penalization R
    '''

    Q = np.eye(4)*0.1
    R = np.eye(2)*0.01

    K, eigVals = dlqr(dynamics.A, dynamics.B, Q, R)

    if correction_dynamics is not None:
        '''
        add the corrected dyanmics to the computed dyanmics
        '''
        pass

    return np.dot(-K, (qf-q0))


def collect_data_and_learn_correction(dynamics, total_data_points=100):
    
    for k in range(total_data_points):

        u = compute_ctrl(q, qf, dynamics)

        q_nxt = dynamics.compute_nxt_state(q=q, u=u, disturb=True)

        data.append(np.r_[q, u, q_nxt])

        q = q_nxt

    np.savetxt('data.txt', data)

    '''
    implement some method that will use this data to learn the correction model
    for example, 
    '''

    return correction_dynamics


def main():
    start = 0.
    goal  =  np.pi

    total_data_points = 100

    '''
    This is a dyanmics object of class Dynamics
    '''
    dynamics = Dynamics()

    data = []

    q = start
    dq = 0.
    qf = goal
    dqf = 0.

    u = compute_ctrl(q, dq, qf, dqf, dynamics)

    '''
    the first is the ideal case where the system is able to compute the
    right control command to take it from start to goal
    '''
    q_nxt, dq_nxt = dynamics(q=q, dq=dq, u=u, disturb=False)
    print "Error in q \t", abs(q_nxt-qf)
    print "Error in dq \t", abs(dq_nxt-dqf)

    '''
    this is the actual case and will show show that the computed control wont take the system to the goal 
    due to the presence of non linear term
    '''

    q_nxt, dq_nxt = dynamics(q=q, dq=dq, u=u, disturb=True)

    print "Error in q \t", abs(q_nxt-qf)
    print "Error in dq \t", abs(dq_nxt-dqf)

    correction_dynamics = collect_data_and_learn_correction()


    u = compute_ctrl(q, dq, qf, dqf, dynamics, correction_dynamics)

    '''
    this is the corrected case and will show show that the computed control wont take the system to the goal 
    due to the presence of non linear term
    '''

    q_nxt, dq_nxt = dynamics(q=q, dq=dq, u=u, disturb=True)

    print "Error in q \t", abs(q_nxt-qf)
    print "Error in dq \t", abs(dq_nxt-dqf)


if __name__ == '__main__':
    main()