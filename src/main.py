import os
from math import sin, cos, pi
import matplotlib.pyplot as plt
from cart_pole import CartPole
from learning_system import LearningCorrection

ideal_cart_pole = CartPole(
    dt=.001,
    init_conds=[0, 0., pi, 0.], #x, dx, phi, dphi
    end=10,
)

ideal_cart_pole.compute_lqr_gain()
ideal_data = ideal_cart_pole.integrate()

dist_cart_pole = CartPole(
    dt=.001,
    init_conds=[0, 0., pi, 0.], #x, dx, phi, dphi
    end=10,
    disturb=True,
)

dist_cart_pole.compute_lqr_gain()
dist_data = dist_cart_pole.integrate()


plt.figure('Ideal system', figsize=(15,15))
plt.subplot(311)
plt.plot(ideal_data[:,0], color='r')
plt.plot(ideal_data[:,1], color='g')
plt.xlabel("time steps")
plt.ylabel("cart magnitudes")

plt.subplot(312)
plt.plot(ideal_data[:,2], color='r')
plt.plot(ideal_data[:,3], color='g')
plt.xlabel("time steps")
plt.ylabel("pendulum magnitudes")

plt.subplot(313)
plt.plot(ideal_data[:,4], color='b')
plt.xlabel("time steps")
plt.ylabel("control magnitudes")


plt.figure('Disturbed system', figsize=(15,15))
plt.subplot(311)
plt.plot(dist_data[:,0], color='r')
plt.plot(dist_data[:,1], color='g')
plt.xlabel("time steps")
plt.ylabel("cart magnitudes")

plt.subplot(312)
plt.plot(dist_data[:,2], color='r')
plt.plot(dist_data[:,3], color='g')
plt.xlabel("time steps")
plt.ylabel("pendulum magnitudes")

plt.subplot(313)
plt.plot(dist_data[:,4], color='b')
plt.xlabel("time steps")
plt.ylabel("control magnitudes")


# ## Learn the system model
lc = LearningCorrection(ideal_system_model=ideal_cart_pole, 
	                    ideal_data=ideal_data, 
	                    disturbed_data=dist_data)


model_correction = lc.compute_correction()

corr_cart_pole = CartPole(
    dt=.001,
    init_conds=[0, 0., pi, 0.], #x, dx, phi, dphi
    end=10,
    disturb=True,
)
corr_cart_pole.correct_model_disturbance(model_correction)
corr_cart_pole.compute_lqr_gain()
corrected_data = corr_cart_pole.integrate()

plt.figure('Corrected system', figsize=(15,15))
plt.subplot(311)
plt.plot(corrected_data[:,0], color='r')
plt.plot(corrected_data[:,1], color='g')
plt.xlabel("time steps")
plt.ylabel("cart magnitudes")

plt.subplot(312)
plt.plot(corrected_data[:,2], color='r')
plt.plot(corrected_data[:,3], color='g')
plt.xlabel("time steps")
plt.ylabel("pendulum magnitudes")

plt.subplot(313)
plt.plot(corrected_data[:,4], color='b')
plt.xlabel("time steps")
plt.ylabel("control magnitudes")


plt.show()