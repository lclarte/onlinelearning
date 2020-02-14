# strategy.py
# classes representant les algos (UCB, Exp3, etc.) qui ont tous une interface commune 
# sample() pour jouer un coup 
# update() pour recuperer les informations de la simulation et MAJ les infos de la strat

import numpy as np
import util

class Strategy:
    def __init__(self, p, N = None, complete_info = True):
        # N is discarded if p is not None
        if p is None:
            if N is None:
                raise ValueError()
            self.p = np.ones(N) / float(N)
        else:
            self.p = p
        self.complete_info = complete_info
        # history of strategies
        self.ps = []

    def sample(self):
        # in all strategies used here, we sample an action by playing randomly
        return util.rand_weighted(self.p)

    def update(self, loss):
        # to define in children classes
        raise NotImplementedError()