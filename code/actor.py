# actor.py
# a player has a strategy variable and is linked to a simulation

from strategy import Strategy

class Actor:
    def __init__(self, player : bool):
        self.player = player
        self.complete_info = None
        self.b_gradient_loss = None

    def set_strategy(self, strategy : Strategy):
        self.strategy = strategy
        # True if loss is w/ complete information
        self.complete_info = self.strategy.complete_info
        self.b_gradient_loss = self.strategy.b_gradient_loss

    def get_strategy(self):
        return self.strategy

    def set_simulation(self, simulation):
        self.simulation = simulation
    
    def play(self):
        # return self.strategy.rand_weight()
        return self.strategy.sample()

    def update(self, loss):
        # strategy.update(loss incured by player)
        # si complete_info : loss : np.ndarray
        # sinon : loss : int
        self.strategy.update(loss)
