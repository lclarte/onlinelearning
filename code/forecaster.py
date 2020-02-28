import numpy as np
from pandas import read_csv

from actor import Actor
from ewa import EWA
from ogd import OGD

class Forecaster:
    def __init__(self, ids : np.array, votes : np.array, strat_factory : callable):
        """
        arguments : 
            - ids : 1D-array with list of ids of choices (ideas, votes, etc.)
            - votes : 2D-array of shape (_, 3) : last axis is id1, id2, victory of id1 in {0, 1}
        """
        self.ids = np.copy(ids)
        # number of possible choices
        self.N = len(self.ids)
        self.votes = np.copy(votes)
        # total number of votes
        self.T = len(votes)
        # define the loss and its gradient 
        self.loss = lambda y_true, y_hat : (1 - y_true)*y_hat + y_true * (1 - y_hat)
        self.loss_grad = lambda y_true, y_hat : (1 - 2 * y_true)
        self.player = None
        self.strat_factory = strat_factory
        self.init(strat_factory)

    def init(self, strat_factory):
        self.t = 0
        self.player = Actor(True)
        self.player.set_strategy(strat_factory())

    def strategies(self, z_1, z_2, fill_value = 0.0):
        f = np.full(2*self.N, fill_value)
        f[z_1] = f[z_2 + self.N] = 1.
        f[z_2] = f[z_1 + self.N] = 0.
        return f
    
    def predict(self, z_1, z_2):
        """
        Return win probability of choice 1 given ids of the 2 choices
        """
        p_t = self.player.get_strategy().p
        return (p_t[z_1] + p_t[z_2 + self.N]) # \
                # / (p_t[z_1] + p_t[z_1 + self.N] + p_t[z_2] + p_t[z_2 + self.N])

    def play(self, t):
        """
        Forecast the result of vote number t in updates the strategy consequently
        """
        z_1, z_2, result = self.votes[t]
        self.current_loss = np.vectorize(lambda y : self.loss(result, y))
        self.current_loss_grad = np.vectorize(lambda y : self.loss_grad(result, y))
        
        f_t = self.strategies(z_1, z_2, 0.0)
        hat_y = self.predict(z_1, z_2 )
        # sleeping trick : change strategy to hat_y
        f_t = self.strategies(z_1, z_2, hat_y)
        loss = self.current_loss(f_t)
        gradient_loss = self.current_loss_grad(f_t)

        if not self.player.strategy.b_gradient_loss:
            self.player.update(loss)
        else:
            self.player.update(gradient_loss)

    def test(self):
        """
        Compute precision and loss on all test
        """
        total_loss = 0.0
        correct = 0
        for t in range(self.T):
            vote = self.votes[t]
            hat_y = self.predict(vote[0], vote[1])
            prediction = 1.0 if (hat_y > 0.5) else 0.0
            correct += int(prediction == vote[2])
            total_loss += self.loss(vote[2], hat_y)
        return correct, total_loss
