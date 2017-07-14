
import matplotlib.pyplot as plt


def visualize_plots(figure_name, data):

    plt.figure(figure_name, figsize=(15,15))
    plt.clf()
    plt.subplot(311)
    plt.plot(data[:,0], color='r') #cart position
    plt.plot(data[:,1], color='g') #cart velocity
    plt.xlabel("time steps")
    plt.ylabel("cart magnitudes")

    plt.subplot(312)
    plt.plot(data[:,2], color='r') #pendulum position
    plt.plot(data[:,3], color='g') #pendulum velocity
    plt.xlabel("time steps")
    plt.ylabel("pendulum magnitudes")

    plt.subplot(313)
    plt.plot(data[:,4], color='b') #control commands
    plt.xlabel("time steps")
    plt.ylabel("control magnitudes")

    plt.show()