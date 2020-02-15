# main.py
# main file

import numpy as np

from simulation import *
from actor import *
from strategy import *
from plot import *
from ewa import EWA
from exp3 import Exp3

def main():
    pL = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    # zero sum game
    oL = -1 * pL
    main_simulation = Simulation(pL, oL, T = 200)
    player, opponent = Actor(True), Actor(False)
    
    player.set_strategy(Exp3(p=np.ones(3) / 3., eta = 1.0))
    # case where the opponent adapts to the strategy
    opponent.set_strategy(EWA(p=np.array([0.5, 0.25, 0.25]), eta=0.0))

    main_simulation.set_player(player)
    main_simulation.set_opponent(opponent)

    main_simulation.run()

    player_ps = main_simulation.actors['player'].strategy.ps
    plot_p(player_ps)

if __name__ == '__main__':
    main()