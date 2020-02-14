# strategy.py
# classes representant les algos (UCB, Exp3, etc.) qui ont tous une interface commune 
# sample() pour jouer un coup 
# update() pour recuperer les informations de la simulation et MAJ les infos de la strat

import numpy as np
import util

class Strategy:
    def __init__(self, p, N = None):
        if p is None:
            if N is None:
                raise ValueError()
            self.p = np.ones(N) / float(N)

    def set_simulation(self, simulation):
        self.simulation = simulation

    def sample(self):
        return util.rand_weighted(self.p)

    def update(self):
        raise NotImplementedError()
        