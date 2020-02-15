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
    main_simulation = Simulation(pL, oL, T = 10000)
    player, opponent = Actor(True), Actor(False)
    
    player.set_strategy(EWA(N=3, eta = 1.0))
    # case where the opponent adapts to the strategy
    opponent.set_strategy(EWA(N=3, eta=0.05))

    main_simulation.set_player(player)
    main_simulation.set_opponent(opponent)

    main_simulation.run()

    player_ps = main_simulation.actors['player'].strategy.ps
    plot_p(player_ps[int(5000):])
    plot_average_p_distance(player_ps, p_star=np.ones(3) / 3.)

if __name__ == '__main__':
    main()