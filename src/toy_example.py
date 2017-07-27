
'''
Install SCIPY library to make this code work
pip install scipy
'''

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

#This makes the random numbers predictable
np.random.seed(0)

#This creates a graph which is larger for the viewer
#plt.figure(1, figsize=(15,15))
#plt.ion()
#plt.show(False)


'''
This funciton implements a discretized lqr controller
it takes in system matrix A, the control matrix B
The state penalization Q, and the control penalization R
It formualates the ricatti equation and solves it
Don't understand the materials given in the doc folder
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
        self.B = np.array([[0., 0.],
                           [0., 0.],
                           [dt, 0.],
                           [0., dt]])

    
    def non_linear_term(self, q):
        return q + np.random.randn(4)


    def compute_nxt_state(self, q, u, disturb=False):
        
        q = np.dot(self.A, q) + np.dot(self.B, u)

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
    Q = np.array([[1.,0.,0.,0.],
                  [0.,1.,0.,0.],
                  [0.,0.,10.,0.],
                  [0.,0.,0.,10.]])
    R = np.eye(2)*0.06

    K, eigVals = dlqr(dynamics.A, dynamics.B, Q, R)

    if correction_dynamics is not None:
        '''
        add the corrected dyanmics to the computed dyanmics
        '''
        pass

    print "U \n", np.dot(-K, (q0-qf))

    return np.dot(-K, (q0-qf))


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


def visualize(start, goal, point_mass_trajectory, error_list):
    plt.clf()
    plt.subplot(221)
    plt.scatter(start[0], start[1], color='r')
    plt.scatter(goal[0],  goal[1],  color='g')

    plt.plot(point_mass_trajectory[0,:], point_mass_trajectory[1,:], color='b')
    plt.xlim([-0.1, 1.])
    plt.ylim([-0.1, 1.])
    plt.xlabel("X location")
    plt.ylabel("Y location")

    plt.subplot(222)
    plt.plot(error_list, color='m')
    plt.xlabel("time steps")
    plt.ylabel("error magnitude")

    plt.subplot(223)
    plt.plot(point_mass_trajectory[2,:], color='r')
    plt.xlabel("time steps")
    plt.ylabel("x velocity magnitude")

    plt.subplot(224)
    plt.plot(point_mass_trajectory[3,:], color='g')
    plt.xlabel("time steps")
    plt.ylabel("y velocity magnitude")

    plt.draw()
    plt.pause(0.0001)


def main():
    start =  np.array([0.,0.,0.,0.]) #x,y,dx,dy
    goal  =  np.array([.5682151,   0.5682151,   0.36304926,  0.36304926]) #x,y,dx,dy

    total_data_points = 100

    time_steps = 100

    '''
    This is a dyanmics object of class Dynamics
    '''
    q  = start.copy()
    qf = goal.copy()

    dynamics = Dynamics()

    data = []

    point_mass_trajectory = np.zeros([4, time_steps])
    point_mass_trajectory[point_mass_trajectory==0.] = np.nan
    point_mass_trajectory[:, 0] = start
    
    error_list = np.asarray([np.nan for _ in range(time_steps)])

    error_list[0] = np.linalg.norm(start-goal)

    for t in range(1, time_steps):

        u = compute_ctrl(q0=q, qf=qf, dynamics=dynamics)

        '''
        the first is the ideal case where the system is able to compute the
        right control command to take it from start to goal
        '''
        q_nxt = dynamics.compute_nxt_state(q=q, u=u, disturb=False)
        q = q_nxt
        print "State \n", q

        point_mass_trajectory[:, t] = q_nxt
        error_list[t] = np.linalg.norm(q_nxt-qf)

        visualize(start, goal, point_mass_trajectory, error_list)

    print q_nxt

    '''
    this is the actual case and will show show that the computed control wont take the system to the goal 
    due to the presence of non linear term
    '''

    # q_nxt, dq_nxt = dynamics(q=q, dq=dq, u=u, disturb=True)

    # print "Error in q \t", abs(q_nxt-qf)
    # print "Error in dq \t", abs(dq_nxt-dqf)

    # correction_dynamics = collect_data_and_learn_correction()


    # u = compute_ctrl(q, dq, qf, dqf, dynamics, correction_dynamics)

    # '''
    # this is the corrected case and will show show that the computed control wont take the system to the goal 
    # due to the presence of non linear term
    # '''

    # q_nxt, dq_nxt = dynamics(q=q, dq=dq, u=u, disturb=True)

    # print "Error in q \t", abs(q_nxt-qf)
    # print "Error in dq \t", abs(dq_nxt-dqf)

    raw_input("Press enter to exit...")


if __name__ == '__main__':
    main()