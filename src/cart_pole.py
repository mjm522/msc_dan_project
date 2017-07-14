'''
code adapted from
https://github.com/toddsifleet/inverted_pendulum
'''

from math import sin, cos, pi
from numpy import matrix, array, asmatrix, eye
from control.matlab import *


#various utility functions
def constrain(theta):
    theta = theta % (2*pi)
    if theta > pi:
        theta = -2*pi+theta
    return theta

def sat(Vsat, V):
    if abs(V) > Vsat:
        return Vsat * cmp(V, 0)
    return V

def average(x):
    x_i, k1, k2, k3, k4 = x
    return x_i + (k1 + 2.0*(k3 + k4) +  k2) / 6.0

theta = []


class LQR():

    def __init__(self, A, B):
        self.Q =  matrix([
                            [10000, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 10000, 0],
                            [0, 0, 0, 1]
                          ])
        self.R = 6.

        self.A = A
        self.B = B

    def compute_lqr_gain(self):
        (K, X, E) = lqr(self.A, self.B, self.Q, self.R)
        return K

class CartPole(object):
    def __init__(self, dt, init_conds, end, disturb=False):

        self.M  = .6  # mass of cart+pendulum
        self.m  = .3  # mass of pendulum
        self.Km =  2  # motor torque constant
        self.Kg = .01  # gear ratio
        self.R  =  6  # armiture resistance
        self.r  = .01  # drive radiu3
        self.K1 = self.Km*self.Kg/(self.R*self.r)
        self.K2 = self.Km**2*self.Kg**2/(self.R*self.r**2)
        self.l  = .3  # length of pendulum to CG
        self.I  = 0.006  # inertia of the pendulum
        self.L  = (self.I + self.m*self.l**2)/(self.m*self.l)
        self.g  = 9.81  # gravity
        self.Vsat = 20.  # saturation voltage

        A11 = -1 * self.Km**2*self.Kg**2 / ((self.M - self.m*self.l/self.L)*self.R*self.r**2)
        A12 = -1*self.g*self.m*self.l / (self.L*(self.M - self.m*self.l/self.L))
        A31 = self.Km**2*self.Kg**2 / (self.M*(self.L - self.m*self.l/self.M)*self.R*self.r**2)
        A32 = self.g/(self.L-self.m*self.l/self.M)

        self.A = matrix([
            [0, 1, 0, 0],
            [0, A11, A12, 0],
            [0, 0, 0, 1],
            [0, A31, A32, 0]
        ])

        B1 = self.Km*self.Kg/((self.M - self.m*self.l/self.L)*self.R*self.r)
        B2 = -1*self.Km*self.Kg/(self.M*(self.L-self.m*self.l/self.M)*self.R*self.r)

        self.B = matrix([
            [0],
            [B1],
            [0],
            [B2]
        ])

        self.state_dim = 4
        self.ctrl_dim  = 1
        self.dt = dt
        self.init_conds = init_conds
        self.end = end
        self.reset()

        if disturb:
            self.add_model_disturbance(2.234)

    def reset(self):
        self.t = 0.0
        self.x = self.init_conds[:]
        self.ctrlr = None

    def add_model_disturbance(self, value):
        A_disturb = asmatrix(eye(4))*value
        self.A += A_disturb

    def correct_model_disturbance(self, correction):
        if correction.shape[0] == 4 and correction.shape[1] == 4:
            self.A += correction
        else:
            raise("Correction matrix passed has a dimention mismatch with system model A")

    def peturb_model(self, peturbation):
        if peturbation.shape[0] == 4 and peturbation.shape[1] == 4:
            self.A += peturbation
        else:
            raise("Peturbation matrix passed has a dimention mismatch with system model A")

    def compute_lqr_gain(self):
        self.ctrlr = LQR(self.A,self.B)

    def derivative(self, u):
        V = sat(self.Vsat, self.control(u))
   
        #x1 = x, x2 = x_dt, x3 = theta, x4 = theta_dt
        x1, x2, x3, x4 = u
        x1_dt, x3_dt =  x2, x4
        x2_dt = (self.K1*V - self.K2*x2 - self.m*self.l*self.g*cos(x3)*sin(x3)/self.L + 
                 self.m*self.l*sin(x3)*x4**2) / (self.M - self.m*self.l*cos(x3)**2/self.L)
        x4_dt = (self.g*sin(x3) - self.m*self.l*x4**2*cos(x3)*sin(x3)/self.L 
                - cos(x3)*(self.K1*V + self.K2*x2)/self.M) / (self.L - self.m*self.l*cos(x3)**2/self.M)
        x = [x1_dt, x2_dt, x3_dt, x4_dt]
        return x

    def control(self, u):
        c = constrain(u[2])
        if c>-pi/5 and c<pi/5:
            K = self.ctrlr.compute_lqr_gain()
            return float(-K*matrix(u[0:2]+[c]+[u[3]]).T)
        else:
            return self.swing_up(u)

    def swing_up(self, u):        
        E0 = 0.
        k = 1
        w = (self.m*self.g*self.l/(4*self.I))**(.5)
        E = self.m*self.g*self.l*(.5*(u[3]/w)**2 + cos(u[2])-1)
        a = k*(E-E0)*cmp(u[3]*cos(u[2]), 0)
        F = self.M*a
        V = (F - self.K2*constrain(u[2]))/self.K1
        return sat(self.Vsat, V)

    def rk4_step(self, dt):
        dx = self.derivative(self.x)
        k2 = [ dx_i*dt for dx_i in dx ]

        xv = [x_i + delx0_i/2.0 for x_i, delx0_i in zip(self.x, k2)]
        k3 = [ dx_i*dt for dx_i in self.derivative(xv)]

        xv = [x_i + delx1_i/2.0 for x_i,delx1_i in zip(self.x, k3)]
        k4 = [ dx_i*dt for dx_i in self.derivative(xv) ]

        xv = [x_i + delx1_2 for x_i,delx1_2 in zip(self.x, k4)]
        k1 = [self.dt*i for i in self.derivative(xv)]

        self.t += dt
        self.x = map(average, zip(self.x, k1, k2, k3, k4))
        theta.append(constrain(self.x[2]))


    def integrate(self):
        x = []
        while self.t <= self.end:
            self.rk4_step(self.dt)
            x.append([self.t] + self.x)
        return array(x)