import numpy as np

import strategy
import util

# question 3 : Exponentially Weighted Average
class EWA(strategy.Strategy):
       
        __name__ = "EWA"

        def __init__(self, p : np.ndarray, eta : float):
                super(). __init__(p, complete_info=True)
                self.eta = eta
                
        @strategy.update
        def update(self, loss):
            self.p = util.EWA_update(self.p, loss, self.eta)
            
        def upper_bound_regret(self, eta, T, K):
            """
            Plots the upper bound for the regret for EWA with fixed adversary
            Here, the upper bound is \eta T + log(K) / \eta
            """
            eta = float(eta)
            return eta * T + np.log(K) / eta 
