# main.py
# main file

import numpy as np
import pandas as pd

from simulation import *
from actor import *
from strategy import *
from plot import *
from forecaster import *
from ewa import EWA
from exp3 import Exp3, Exp3IX

p_init = np.ones(3) / 3.
q_init = np.array((0.5, 0.25, 0.25))

# T = 100
eta1 = np.sqrt(2 * np.log(4) / (100 * 3))

config_question6 = {
    
    'player_strat_factory' : lambda : Exp3(p_init, eta = 1.0),
    'opponent_strat_factory' : lambda : EWA(q_init, eta = 0.0),
    'player_strat_eta_comparison' : lambda eta : Exp3(p_init, eta = eta)
}

config_question7 = {
    'player_strat_factory' : lambda : Exp3(p_init, eta = 1.0),
    'opponent_strat_factory' : lambda : EWA(p_init, eta = 0.05),
    'player_strat_eta_comparison' : lambda eta : Exp3(p_init, eta = eta)
}

config_question6Exp3IX = {
    'player_strat_factory' : lambda : Exp3IX(p_init, eta = eta1, gamma = eta1 / 2),
    'opponent_strat_factory' : lambda : EWA(q_init, eta = 0.0),
    'player_strat_eta_comparison' : lambda eta : Exp3IX(p_init, eta = eta, gamma = eta / 2)
}

def run_simulation(player_strat_factory : callable, opponent_strat_factory : callable, N : int, _plot = True):
    """
    player_strat_factory : function to create the same strategy at each iteration
    Here, opponent strategy is fixed and hardcoded 
    """
    pL = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    # zero sum game
    oL = -1 * pL

    T = 100

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
        plot_avg_losses_std(losses, player_strat_factory().__class__.__name__)
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

def run_config(config):
    run_simulation(config['player_strat_factory'], config['opponent_strat_factory'], 10)
    compareetas(config['player_strat_eta_comparison'])

def run_forecaster():
    ids_file = 'data/ideas_dataset/ideas_id.csv'
    votes_file = 'data/ideas_dataset/ideas_votes.csv'
    ids_data = pd.read_csv(ids_file)
    votes_data = pd.read_csv(votes_file)
    ids_data = np.array(ids_data.loc[:, 'idea.id'])
    votes_data = np.array(votes_data.loc[:, ['z1', 'z2', 'y']])
    # offset all ids and votes by 1 (begin at 0 now)
    ids_data -= 1
    votes_data[:, 0] -= 1
    votes_data[:, 1] -= 1

    forecaster = Forecaster(ids_data, votes_data)
    for t in range(len(ids_data)):
        forecaster.play(t)
    p = forecaster.player.get_strategy().p
    correct, loss = forecaster.test() 
    print(float(correct) / forecaster.T)

def main():
    # run_config(config_question6Exp3IX)
    # run_config(config_question7)
    run_forecaster()

if __name__ == '__main__':
    main()