import numpy as np


def non_linear_term(q, dq):
    a0 = 1e-5
    a1 = 1e-4

    return a0*(q**2) +  a1*(dq**2)


def dynamics(q, dq, u, dt=0.01, disturb=False):
    mass    = 1.
    gravity = -9.81
    length  =  1.

    ddq = (1./mass)*(u-gravity)

    dq += ddq*dt
    q  += dq*dt

    if disturb:
        q += non_linear_term(q, dq)

    return q, dq



def compute_ctrl(q0, dq0, qf, dqf, dynamics, correction_dynamics=None):
    '''
    find a control command using the start, final and the dynamics
    at present it is not doing anything, just returning a random control input
    so initially the correction dyanmics is none and later it learns it
    '''

    if correction_dynamics is not None:
        '''
        add the corrected dyanmics to the computed dyanmics
        '''

    return np.random.rand()


def collect_data_and_learn_correction(total_data_points=100):
    
    for k in range(total_data_points):

        u = compute_ctrl(q, dq, qf, dqf, dynamics)

        q_nxt, dq_nxt = dynamics(q=q, dq=dq, u=u, disturb=True)

        data.append(np.array([q, dq, u, q_nxt, dq_nxt]))

        q = q_nxt
        dq = dq_nxt

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