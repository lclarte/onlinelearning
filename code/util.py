import numpy as np

# question 2.a : tirer un entier 0 <= i < M-1 a partir de p \in \Delta_M
# pour avoir une fonction en une ligne, on ne verifie pas la validite de p
rand_weighted = lambda p : np.argwhere(np.cumsum(p) > np.random.rand())[0, 0]

# returns cumulative average loss. works only if ls is a 1D array
get_average_loss = lambda ls : np.cumsum(ls) / np.arange(1, len(ls) + 1)

# min_i {sum_t L(i, j_t)}
get_min_loss = lambda L, js : np.min(np.sum(L.transpose()[js], axis=0))

# returns regret at final time R_T given list of L(i_t, j_t)
def get_regret(L, ls, js, T = -1):
	return np.sum(ls[:T]) - get_min_loss(L, js[:T])

# returns list of cumulative regret
def get_cumulative_regret(L, ls, js):
	return [0] + [get_regret(L, ls, js, T) for T in range(1, len(ls))]
