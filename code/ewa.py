import numpy as np

import strategy
import util

# question 3 : Exponentially Weighted Average
class EWA(strategy.Strategy):
	
	def __init__(self, p : np.ndarray, eta : float):
		super(). __init__(p, complete_info=True)
		self.eta = eta
		
	@strategy.update
	def update(self, loss):
		self.p = util.EWA_update(self.p, loss, self.eta)