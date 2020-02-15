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
        self.sums, self.visits = np.zeros(self.K), np.zeros(self.K)
        self.last_action = None

    def sample(self):
        self.last_action = super().sample()
        self.visits[self.last_action] += 1.0
        return self.last_action

    @strategy.update
    def update(self, loss):
        k = self.last_action
        self.sums[k] = self.sums[k] + loss
        # update average loss
        self.update_loss = np.zeros(self.K)
        # use temporal difference
        self.update_loss[k] = (-1.0 / self.visits[k]) * (loss - self.sums[k])
        self.p = util.EWA_update(self.p, self.update_loss, self.eta)