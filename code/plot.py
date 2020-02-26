import numpy as np
import matplotlib.pyplot as plt

def pltdecorator(function):
	def inner(*args, **kwargs):
		retour = function(*args)
		if 'title' in kwargs:
			plt.title(kwargs['title'])
		plt.legend()
		plt.tight_layout()
		plt.show()
		return retour
	return inner

@pltdecorator
def plot_p(_ps):
	"""
	plot the history of the strategy
	"""
	ps = np.array(_ps)
	T, K = ps.shape
	absc = np.linspace(1, T, T)
	for k in range(K):
		plt.plot(absc, ps[:, k], label='$p_' + str(k + 1) + '$')
	plt.title('History of weights as a function of t')
	plt.xlabel('t')
	plt.ylabel('weights')

@pltdecorator
def plot_average_loss(average_loss, method : str = ""):
	"""
	plot the average loss w/ K simulation trials
	Assumes the cumulative averages are already computed
	"""
	K, T = average_loss.shape
	for k in range(K):
		plt.plot(np.linspace(1, T, T), average_loss[k], label = str(k))
	plt.title('Average loss as a function of t. Algorithm : ' + method)
	plt.xlabel('t')
	plt.ylabel('$\\bar{l_t}$')

@pltdecorator
def plot_cumulative_regret(cumulative_regrets):
	"""
	plot cumumative regret w/ K simulation trials
	"""
	K, T = cumulative_regrets.shape
	for k in range(K):
		plt.plot(np.linspace(1, T, T), cumulative_regrets[k], label=str(k))
	plt.title('Cumulative regret')
	plt.xlabel('t')
	plt.ylabel('$\\bar{R_t}$')

@pltdecorator
def plot_min_max_avg_losses(average_loss, method : str = ""):
	"""
	plot various stats on cumulative average losses w/ K trials 
	"""
	K, T = average_loss.shape
	min_loss = np.amin(average_loss, axis=0)
	max_loss = np.amax(average_loss, axis=0)
	avg_loss = np.mean(average_loss, axis=0)
	X = np.linspace(1, T, T)
	for k in range(K):
		plt.plot(X, average_loss[k], color='#CCCCCC')
	plt.plot(X, min_loss, label='Min average loss', color='b')
	plt.plot(X, max_loss, label='Max average loss', color='r')
	plt.plot(X, avg_loss, label='Mean average loss', color='k')
	plt.title('Average loss as a function of t. Algorithm : ' + method)
	plt.xlabel('t')
	plt.ylabel('$\\bar{l_t}$')

@pltdecorator
def plot_avg_losses_std(average_loss : np.ndarray, method : str = ""):
	K, T = average_loss.shape
	X = np.linspace(1, T, T)
	mean = np.mean(average_loss, axis=0)
	deviation = np.std(average_loss, axis=0)
	plt.plot(X, mean)
	plt.fill_between(X, mean-deviation, mean+deviation, alpha=.25)
	plt.title("Mean average loss over " + str(K) + " trials and standard deviation. Algorithm : " + method)

@pltdecorator
def plot_regret_eta(etas, regrets, method : str = ""):
	"""
	Compare the regret at time T w/ different learning rates
	"""
	plt.title("Regret $R_T$ as a function of the learning rate $\\eta$. Algorithm : " + method)
	plt.xlabel("$\\eta$")
	plt.ylabel("Regret $R_T$")
	plt.plot(etas, regrets)
	

@pltdecorator
def plot_average_p_distance(ps : np.ndarray, p_star : np.ndarray, method : str = ""):
	"""
	plots the distance between the average weights p at time t 
	and a given weight vector p_star (only 1 trial allowed)
	"""
	ps = np.array(ps)
	# works as expected
	average_ps = np.cumsum(ps, axis=0) / np.arange(1, len(ps) + 1 )[:, None]
	distance = [np.linalg.norm(average_ps[i] - p_star) for i in range(len(average_ps))]
	plt.plot(distance)

	p_star_print = np.round(p_star, 2)
	plt.title("Distance between average weights $\\bar{p}_t$ and " + str(p_star_print) + ". Algorithm : " + method)
	plt.xlabel("t")
	plt.ylabel("$ \| \\bar{p}_t - " + str(p_star_print) + "\|_2$")