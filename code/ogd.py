# ogd.py
# strategy for online gradient descent

import strategy

class OGD(strategy.Strategy):
    def __init__(self):
        super().__init__(p, complete_info=True)
        self.b_gradient_loss = True

    def projection(self):
        pass

    @strategy.update
    def update(self, loss_gradient):
        # 1 ) Descend gradient
        # 2 ) Project on simplex
        pass
