import numpy as np

# question 2.a : tirer un entier 0 <= i < M-1 a partir de p \in \Delta_M
# pour avoir une fonction en une ligne, on ne verifie pas la validite de p
rand_weighted = lambda p : np.argwhere(np.cumsum(p) > np.random.rand())[0, 0]

# question 2.b : met a jour le vecteur des poids p avec la loss l (information
# complete ) et le learning rate eta
def EWA_update(p : np.ndarray, l : np.ndarray, eta : float):
	unnormalized = p * np.exp( - eta * l)
	return unnormalized / np.sum(unnormalized)
    
# question 3 : implemente EWA, prend en argument la loss matrix L et le temps T
# 	+ le learning rate eta + strategie de l'adversaire q
def EWA(L : np.ndarray, T : int, eta : float, q : np.ndarray):
	M, N = L.shape
	ps = [np.ones(M) / float(M)]
	# player and opponent actions
	js = []
	ls = []
	for t in range(T):
		# opponent choses action w/ q
		js.append(rand_weighted(q))
		# chose random action and suffer regret
		ls.append(L[rand_weighted(ps[-1]), js[-1]])
		ps.append(EWA_update(ps[-1], L[:, js[-1]], eta = eta))
	# return list of policies, list of player regret
	return ps, ls, js

# question 4 : opponent also applies EWA update with different learning rates
def adaptive_EWA(L : np.ndarray, T : int, eta : float, *, adv_eta : float = None, q0 : np.ndarray = None):
	M, N = L.shape
	# if not specified, opponent has uniform strategy
	adv_eta = adv_eta or eta
	if q0 is None:
		q0 = np.ones(N) / N
	ps, qs = [np.ones(M) / float(M)], [ q0 ]
	actions, js = [], []
	# player loss
	ls = []
	for t in range(T):
		# random actions
		js.append(rand_weighted(qs[-1]))
		actions.append(rand_weighted(ps[-1]))
		# player suffers a loss, opponent suffers -loss
		ls.append(L[actions[-1], js[-1]])
		# update strategies
		ps.append(EWA_update(ps[-1], L[:, js[-1]], eta = eta))
		# careful : minus sign before L because player loss is opponent's gain
		qs.append(EWA_update(qs[-1], -L[actions[-1]], eta = adv_eta ))
	return ps, qs, ls
