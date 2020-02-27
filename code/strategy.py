
# classes representant les algos (UCB, Exp3, etc.) qui ont tous une interface commune 
# sample() pour jouer un coup 
# update() pour recuperer les informations de la simulation et MAJ les infos de la strat

import numpy as np
import util

# decorator for update function : adds current weights to weights history
def update(f):
    def inner(self, loss):
        result = f(self, loss)
        self.ps.append(self.p)
        return result
    return inner


class Strategy:
    def __init__(self, p, complete_info = True):
        self.p = np.copy(p)
        # remember of initial strategy 
        self.complete_info = complete_info
        # history of strategies 
        self.ps = [self.p]
        # boolean if the strategy requires loss or its gradient
        self.b_gradient_loss = False

    def get_p(self):
        return self.p

    def sample(self):
        # in all strategies used here, we sample an action by playing randomly
        return util.rand_weighted(self.p)

    @update
    def update(self, loss):
        # to define in children classes
        raise NotImplementedError()
