# exp3.py
# Exponential-weight algorithm for Exploration and Exploitation

import strategy
import util

import numpy as np

class Exp3(strategy.Strategy):
    def __init__(self, p : np.ndarray, eta : float):
        super().__init__(p, complete_info=False)
        self.eta = eta
        self.K = len(p)
        # sum of loss + number of time we visit the arms
        self.last_action = None

    def sample(self):
        self.last_action = super().sample()
        return self.last_action

    @strategy.update
    def update(self, loss):
        k = self.last_action
        # cf http://www.spadro.eu/sites/default/files/SebastienG.pdf
        self.update_loss = np.zeros(self.K)
        self.update_loss[k] = loss / self.p[k]
        # cf site : https://banditalgs.com/2016/10/01/adversarial-bandits/
        self.p = util.EWA_update(self.p, self.update_loss, self.eta)

class Exp3IX(Exp3):
    """
    Exp3 with variance reduction. Remark : optimal parameters are 
    eta = eta1 = sqrt(2 * log(K+1) / (nk))
    and gamma = eta1 / 2
    """

    def __init__(self, p : np.ndarray, eta : float, gamma : float):
        super().__init__(p, eta)
        self.gamma = gamma

    @strategy.update
    def update(self, loss):
        k = self.last_action
        self.update_loss = np.zeros(self.K)
        self.update_loss[k] = loss / (self.p[k] + self.gamma)
        self.p = util.EWA_update(self.p, self.update_loss, self.eta) 
        