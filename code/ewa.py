import numpy as np

from strategy import Strategy

# question 2.b : met a jour le vecteur des poids p avec la loss l (information
# complete ) et le learning rate eta
def EWA_update(p : np.ndarray, l : np.ndarray, eta : float):
	unnormalized = p * np.exp( - eta * l)
	return unnormalized / np.sum(unnormalized)

# question 3 : Exponentially Weighted Average
class EWA(Strategy):
	
	def __init__(self, p : np.ndarray, eta : float):
		super(). __init__(p, N=None, complete_info=True)
		self.eta = eta
		# history of all strategies
		self.ps = [ self.p ]

	def update(self, loss):
		self.p = EWA_update(self.p, loss, self.eta)
		self.ps.append(self.p)