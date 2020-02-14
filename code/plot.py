import numpy as np
import matplotlib.pyplot as plt

def plot_p(ps):
	x = np.linspace(1, len(ps), len(ps))
	for i in range(3):
		y = [ps[_][i] for _ in range(len(ps))]
		plt.plot(x, y, label='$p_' + str(i+1) + '$')
	plt.legend()
	plt.title('Evolution of p during simulation')
	plt.xlabel('t')
	plt.ylabel('$p_i$')
	plt.show()

def plot_average_loss(average_loss):
	K, T = average_loss.shape
	for k in range(K):
		plt.plot(np.linspace(1, T, T), average_loss[k])
	plt.title('Average loss as a function of t')
	plt.xlabel('t')
	plt.ylabel('$\\bar{l_t}$')
	plt.show()

def plot_cumulative_regret(cumulative_regrets):
	K, T = cumulative_regrets.shape
	for k in range(K):
		plt.plot(np.linspace(1, T, T), cumulative_regrets[k])
	plt.title('Cumulative regret')
	plt.xlabel('t')
	plt.ylabel('$\\bar{R_t}$')
	plt.show()

def min_max_avg_losses(average_loss):
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
	plt.title('Average loss as a function of t')
	plt.xlabel('t')
	plt.ylabel('$\\bar{l_t}$')
	plt.legend()
	plt.show()

def plot_regret_eta(etas, regrets):
	plt.plot(etas, regrets)
	plt.show()

def plot_average_p_distance(ps):
	ps = np.array(ps)
	# works as expected
	average_ps = np.cumsum(ps, axis=0) / np.arange(1, len(ps) + 1 )[:, None]
	p_star = np.ones(3) / 3.
	distance = [np.linalg.norm(average_ps[i] - p_star) for i in range(len(average_ps))]
	plt.plot(distance) ; plt.show()