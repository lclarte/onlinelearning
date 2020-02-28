# ogd.py
# strategy for online gradient descent

import numpy as np

import strategy 

class OGD(strategy.Strategy):
    def __init__(self, p, eta): 
        super().__init__(p, complete_info = True)
        self.b_gradient_loss = True
        self.eta = eta

    def project_simplex(self, v, z=1):
        n_features = v.shape[0]
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u) - z
        ind = np.arange(n_features) + 1
        cond = u - cssv / ind > 0
        rho = ind[cond][-1]
        theta = cssv[cond][-1] / float(rho)
        w = np.maximum(v - theta, 0)
        return w

    """
    def project_simplex(self, v):
        # Project vector v into the simplex
        n = len(v)
        # c.f https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf
        # 1 ) Sort v in reverse order
        mu = sorted(v, reverse=True) 
        partial_sum = np.cumsum(mu)
        # In the article, z = 1.0
        tmp = np.array([mu[j] - (1. / (j+1))*(partial_sum[j] - 1.) for j in range(n)]) 
        rho = np.argwhere(tmp > 0)[-1, 0]
        theta = (1. / (rho+1))*(partial_sum[rho] - 1)
        retour = np.maximum(v - theta, 0.)
        return retour
    """


    @strategy.update
    def update(self, loss_gradient):
        # gradient descent 
        before = self.p
        self.p = self.p - self.eta * loss_gradient
        # project updated weights on simplex
        self.p =  self.project_simplex(self.p)
        # print(np.linalg.norm(self.p - before))

