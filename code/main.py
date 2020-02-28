# main.py
# main file

import argparse
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

RPS_loss_matrix = np.array([[0.0, 1.0, -1.], [-1., 0., 1.], [1., -1., 0.]])

config_question3 = {

    'player_strat_factory' : lambda :  EWA(p_init, eta = 1.0),
    'opponent_strat_factory' : lambda : EWA(q_init, eta = 0.0),
    'player_strat_eta_comparison' : lambda eta : EWA(p_init, eta = eta),

}

config_question4 = {

    'player_strat_factory' : lambda :  EWA(p_init, eta = 1.0),
    'opponent_strat_factory' : lambda : EWA(p_init, eta = 0.05),
}


config_question7 = {

    'player_strat_factory' : lambda :  Exp3(p_init, eta = 1.0),
    'opponent_strat_factory' : lambda : Exp3(p_init, eta = 0.05),
}

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
    pL = np.copy(RPS_loss_matrix)
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
        method = player_strat_factory().__class__.__name__
        plot_cumulative_regret(regrets, method)
        plot_average_loss(losses, method)
        plot_avg_losses_std(losses, method) 
        plot_min_max_avg_losses(losses, method)


def compareetas(strat_factory : callable):
    """
    opponent has a fixed strategy = (0.5, 0.25, 0.25)
    la fonction strat_factory doit etre parametrisee par eta
    """
    # zero sum game
    pL = np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    oL = -1 * pL
    T = 100
    # number of trials for Monte Carlo
    N = 20
    etas = [0.01, 0.05, 0.1, 0.5, 1.]
    regrets = np.zeros((len(etas), N))
    for k in range(len(etas)):
        for i in range(N):
            eta = etas[k]
            simulation = Simulation(pL, oL, T)
            player, opponent = Actor(True), Actor(False)
            player.set_strategy(strat_factory(eta))
            q_init = np.array((0.5, 0.25, 0.25))
            opponent.set_strategy(EWA(q_init, eta = 0.0))
            simulation.set_player(player)
            simulation.set_opponent(opponent)

            simulation.run()

            regrets[k, i] = util.get_regret(pL, simulation.bandit_losses['player'], simulation.actions['opponent'])
    mean_regrets = np.mean(regrets, axis=1) 
    tmp = strat_factory(0.0) 
    upper_bound_function = getattr(tmp, 'upper_bound_regret', None)
    if upper_bound_function:
        upper_bounds = [upper_bound_function(eta, T, 3) for eta in etas]
        plt.plot(etas, upper_bounds, label='Theoretical upper bound')
    plot_regret_eta(etas, mean_regrets, strat_factory(0.0).__name__)

def run_fixed(config):
    run_simulation(config['player_strat_factory'], config['opponent_strat_factory'], 10)
    compareetas(config['player_strat_eta_comparison'])

def run_simulation2(player_strat_factory : callable, opponent_strat_factory : callable, N : int, _plot = True):
    # zero sum game
    pL = np.copy(RPS_loss_matrix)
    oL = -1 * pL

    T = 100
    Tprime = int(1e4) 

    losses = np.zeros((N, T))
    # plot only one evolution of weights
    ps = None

    # Last simulation is to compute the distance for T = 1000
    for n in range(N+1):
        main_simulation = Simulation(pL, oL, T = T)
        if n == N:
            main_simulation = Simulation(pL, oL, T = Tprime)

        player, opponent = Actor(True), Actor(False)
        
        player.set_strategy(player_strat_factory())

        method = player_strat_factory().__class__.__name__
        # opponent's strategy
        opponent.set_strategy(opponent_strat_factory())

        main_simulation.set_player(player)
        main_simulation.set_opponent(opponent)

        main_simulation.run()

        if n < N:
            losses[n] = util.get_average_loss(main_simulation.bandit_losses['player'])
        else:
            ps = main_simulation.actors['player'].get_strategy().ps

    if _plot:
        method = player_strat_factory().__class__.__name__
        plot_average_loss(losses, title='Average loss with adaptive adversary and strategy ' + method)

        plot_average_p_distance(ps, np.ones(3) / 3., method) 


def run_adaptive(config):
    run_simulation2(config['player_strat_factory'], config['opponent_strat_factory'], 10)

def run_forecaster(config):
    
    # data loading and cleaning

    ids_file = 'data/ideas_dataset/ideas_id.csv'
    votes_file = 'data/ideas_dataset/ideas_votes.csv'
    ids_data = pd.read_csv(ids_file)
    votes_data = pd.read_csv(votes_file)
    ids_data = np.array(ids_data.loc[:, 'idea.id'])
    N = len(ids_data)
    votes_data = np.array(votes_data.loc[:, ['z1', 'z2', 'y']])
    # offset all ids and votes by 1 (begin at 0 now)
    ids_data -= 1
    votes_data[:, 0] -= 1
    votes_data[:, 1] -= 1

    eta = 1.0
    p_init = np.random.uniform(size=2*N)
    p_init /= np.sum(p_init)

    strat_factory = lambda : EWA(p_init, eta)
    strat_factory = lambda : OGD(p_init, eta)

    forecaster = Forecaster(ids_data, votes_data, strat_factory)
    for t in range(len(ids_data)):
        forecaster.play(t)
    p = forecaster.player.get_strategy().p
    correct, _ = forecaster.test() 
    print(float(correct) / forecaster.T)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fonction')
    parser.add_argument('config_file')
    args = parser.parse_args()
    fonctions = {'fixed'   : run_fixed,
                 'adaptive': run_adaptive,
                 'forecaster' : run_forecaster
                 }
    configs = {'question4' : config_question4,
                'question3' : config_question3,
               'question6' : config_question6,
               'question6IX' : config_question6Exp3IX,
               'question7' : config_question7
               }

    fonctions[args.fonction](configs[args.config_file])

if __name__ == '__main__':
    main()
