from util import *
from EWA import *
from plot import *

import sys

# ===========Main functions================

L_RPS = np.array([[0., 1., -1.], [-1., 0., 1.], [1., -1., 0.]])

def several_trials():
	# RPS loss matrix
	L = L_RPS
	# fixed stragegy of the opponent
	q = np.array([0.5, 0.25, 0.25])

	T, K = 100, 10

	average_loss = np.zeros((K, T))
	cumulative_regrets = np.zeros((K, T))
	for k in range(K):
		# weights, losses and opponent's actions
		ps, ls, js = EWA(L, T = T, eta = 1.0, q = q)
		average_loss[k] = get_average_loss(ls)
		cumulative_regrets[k] = get_cumulative_regret(L, ls, js)
	min_max_avg_losses(average_loss)

def eta_comparison():
	# RPS loss matrix
	L = L_RPS
	# fixed stragegy of the opponent
	q = np.array([0.5, 0.25, 0.25])
	etas = [0.01, 0.05, 0.1, 0.5, 1.]
	T, M = 100, 100 # M = 1 dans le DM
	regrets = np.zeros((len(etas), M))
	for k in range(len(etas)):
		for m in range(M):
			ps, ls, js = EWA(L, T = T, eta = etas[k], q = q)
			regrets[k, m] = get_regret(L, ls, js)
	plot_regret_eta(etas, np.mean(regrets, axis=1))

def adaptive_adversary():
	L = L_RPS
	q = np.array([0.5, 0.25, 0.25])
	T, K = 100, 1
	average_losses = np.zeros((K, T))
	for k in range(K):
		ps, qs, ls = adaptive_EWA(L, T, eta = 0.05, adv_eta=0.05,q0=q)
		average_losses[k] = get_average_loss(ls)
		plot_p(ps)
		plot_average_p_distance(ps)
	
def main():
	functions = {'eta_comparison' : eta_comparison,
				 'adaptive_adversary' : adaptive_adversary,
				 'several_trials' : several_trials}

	function = functions[sys.argv[1]]
	function()

if __name__ == '__main__':
	main()