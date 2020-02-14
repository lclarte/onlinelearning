# simulation.py
# this class makes the two players play against one another and manages the statistics 

class Simulation:
    def __init__(self, L, T):
        self.L = L
        self.T = T
        self.player = None
        self.opponent = None

    def set_player(self, player):
        self.player = player
        self.player.set_simulation(self)

    def set_opponent(self, opponent):
        self.opponent = opponent
        self.opponent.set_simulation(self)

    def run(self):
        raise NotImplementedError()