# util.py 
# various functions useful for all scripts in the project

import numpy as np

# question 2.a : tirer un entier 0 <= i < M-1 a partir de p \in \Delta_M
# pour avoir une fonction en une ligne, on ne verifie pas la validite de p
def rand_weighted(p):
	value = 0.
	try:
		value = np.argwhere(np.cumsum(p) > np.random.rand())[0, 0]
	except Exception as e:
		print(e)
		print(np.cumsum(p))
		input()
	return value

# question 2.b : met a jour le vecteur des poids p avec la loss l (information
# complete ) et le learning rate eta
def EWA_update(p : np.ndarray, l : np.ndarray, eta : float):
	unnormalized = p * np.exp( - eta * l)
	return unnormalized / np.sum(unnormalized)

# returns cumulative average loss FOR 1 SIMULATION. works only if ls is a 1D array
get_average_loss = lambda ls : np.cumsum(ls) / np.arange(1, len(ls) + 1)

# min_i {sum_t L(i, j_t)}
# js : opponent actions during the simulation
get_min_loss = lambda L, js : np.min(np.sum(L.transpose()[js], axis=0))

# returns regret at final time R_T given list of L(i_t, j_t)
def get_regret(L, ls, js, T = -1):
	return np.sum(ls[:T]) - get_min_loss(L, js[:T])

# returns list of cumulative regret for 1 SIMULATION
def get_cumulative_regret(L : np.ndarray, ls : np.ndarray, js : list):
	return [0] + [get_regret(L, ls, js, T) for T in range(1, len(ls))]