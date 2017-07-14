from numpy import pi
from cart_pole import CartPole
from plot_graphs import visualize_plots
from render_movie import visualize_movie
from learning_system import LearningCorrection


play_movie = False

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

# ## Learn the system model
lc = LearningCorrection(ideal_system=ideal_cart_pole,
	                    ideal_data=ideal_data, 
	                    real_system=dist_cart_pole)


model_correction = lc.compute_correction(visualize_data=False)
dist_cart_pole.reset()
dist_cart_pole.correct_model_disturbance(model_correction)
dist_cart_pole.compute_lqr_gain()
corrected_data = dist_cart_pole.integrate()


# if play_movie:
#     visualize(ideal_data)
# else:
#     visualize_plots('Ideal data', ideal_data)
#     visualize_plots('Real data', dist_data)
#     visualize_plots('Corrected data', corrected_data)

raw_input("Press enter to exit...")