# main.py
# main file

import numpy as np

from simulation import *
from actor import *
from strategy import *
from plot import *
from ewa import EWA
from exp3 import Exp3

def run_simulation(player_strat_factory : callable, opponent_strat_factory : callable, N : int, _plot = True):
    """
    player_strat_factory : function to create the same strategy at each iteration
    Here, opponent strategy is fixed and hardcoded 
    """
    pL = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    # zero sum game
    oL = -1 * pL

    T = 1000

    losses = np.zeros((N, T))
    regrets = np.zeros((N, T))

    for n in range(N):
        main_simulation = Simulation(pL, oL, T = T)
        
        player, opponent = Actor(True), Actor(False)
        
        player.set_strategy(player_strat_factory())

        # opponent's strategy
        opponent.set_strategy(opponent_strat_factory())

        main_simulation.set_player(player)
        main_simulation.set_opponent(opponent)

        main_simulation.run()

        losses[n] = util.get_average_loss(main_simulation.bandit_losses['player'])
        opponent_actions= main_simulation.actions['opponent'].copy()
        regrets[n]= util.get_cumulative_regret(pL, main_simulation.bandit_losses['player'], opponent_actions)

    if _plot:
        plot_cumulative_regret(regrets)
        plot_average_loss(losses)
        plot_min_max_avg_losses(losses)

def compareetas(strat_factory : callable):
    """
    opponent has a fixed strategy = (0.5, 0.25, 0.25)
    la fonction strat_factory doit etre parametrisee par eta
    """
    pL = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    # zero sum game
    oL = -1 * pL
    T = 100
    etas = [0.01, 0.05, 0.1, 0.5, 1.]
    regrets = [0]*len(etas)
    
    for k in range(len(etas)):
        eta = etas[k]
        simulation = Simulation(pL, oL, T)
        player, opponent = Actor(True), Actor(False)
        player.set_strategy(strat_factory(eta))
        q_init = np.array((0.5, 0.25, 0.25))
        opponent.set_strategy(EWA(q_init, eta = 0.0))
        simulation.set_player(player)
        simulation.set_opponent(opponent)

        simulation.run()

        regrets[k] = util.get_regret(pL, simulation.bandit_losses['player'], simulation.actions['opponent'])
    
    plot_regret_eta(etas, regrets)

# question 6 : player uses Exp3 strategy w/ static opponent
def question6():
    print("Question 6 : player uses Exp3, fixed opponent")
    p_init = np.ones(3) / 3.
    s1 = lambda : Exp3(p_init, eta = 1.0)
    s1eta = lambda eta : Exp3(p_init, eta = eta)

    q_init = np.array((0.5, 0.25, 0.25))
    s2 = lambda : EWA(q_init, eta = 0.0)
    
    run_simulation(s1, s2, 10)
    compareetas(s1eta)

def question7():
    print("Question 7 : opponent plays w/ EWA")
    p_init = np.ones(3) / 3.
    # Exp3, eta = 1.0
    s1 = lambda : Exp3(p_init, eta = 1.0)
    # EWA, eta = 0.05
    s2 = lambda : Exp3(p_init, eta = 1.0)
    run_simulation(s1, s2, 10)

def main():
    question6()
    question7()

if __name__ == '__main__':
    main()