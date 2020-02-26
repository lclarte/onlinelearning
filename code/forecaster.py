import numpy as np
from pandas import read_csv

from actor import Actor
from ewa import EWA

class Forecaster:
    def __init__(self, ids : np.array, votes : np.array):
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
        self.init()

    def init(self):
        self.t = 0
        self.player = Actor(True)
        # for now, we use EWA
        self.eta = 0.125
        self.player.set_strategy(EWA(np.ones(2*self.N) / (2*self.N), self.eta))

    def predict(self, z_1, z_2):
        z_1_win, z_1_lose, z_2_win, z_2_lose = z_1, z_1 + self.N, z_2, z_2 + self.N
        p_t = self.player.get_strategy().p
        return (p_t[z_1_win] + p_t[z_2_lose]) / (p_t[z_1_win]+ p_t[z_1_lose] + p_t[z_2_win]+ p_t[z_2_lose])

    def play(self, t):
        """
        Forecast for 
        """
        vote = self.votes[t]
        z_1_win, z_1_lose, z_2_win, z_2_lose = vote[0], vote[0] + self.N, vote[1], vote[1] + self.N
        result = vote[2]
        # 1) le joueur donnes les probas actuelles p_t
        p_t = self.player.get_strategy().p
        # 3) Calcul de la prediction du joueur 
        # hat_y : prediction de notre joueur, proba de gagner
        # -> Necessite de normaliser : \hat{y}_t = (p_t(z_1) + p_t(z_2 + N)) / (p_t(z_1) + p_t(z_2 + N) + p_t(z_2) + p_t(z_1 + N))
        hat_y = (p_t[z_1_win] + p_t[z_2_lose]) / (p_t[z_1_win]+ p_t[z_1_lose] + p_t[z_2_win]+ p_t[z_2_lose])
        # vector loss of the player 
        # sleeping strategies have same loss as our prediction
        losses = np.full(2*self.N, self.loss(hat_y, result))
        # loss of predictors that predict win of choice 1 
        losses[z_1_win] = losses[z_2_lose] = self.loss(result, 1.0)
        # loss of predictors that predict loss of choice 2
        losses[z_1_lose] = losses[z_2_win] = self.loss(result, 0.0)
        # 5) Application de la loss sur chacun des p_t(k) du joueur
        self.player.update(losses)

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