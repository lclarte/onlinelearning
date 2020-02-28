# simulation.py
# this class makes the two players play against one another and manages the statistics 

import numpy as np

from actor import Actor

class Simulation:
    def __init__(self, pL : np.ndarray, oL : np.ndarray, T):
        # for non zero sum games, oL (opponent loss matrix) != -pL.transpose() (player loss matrix)
        self.T = T

        # pL[i, j] : loss du player quand le joueur joue i, l'adversaire joue j
        # oL[i, j] : loss de l'adv. le joueur joue i et que l'adv. joue j
        # par simplicite, le joueur est donc toujours en premier indice
        # pour un zero sum game, on a donc oL = - pL (si on a pas un zero sum game,
        # les deux matrices doivent donc etre specifiees)

        self.loss_matrices = {'player' : pL, 'opponent' : oL}
        self.minlosses = {'player' : np.min(pL), 'opponent' : np.min(oL)}

        self.actors = {'player' : None, 'opponent' : None}
        self.actions = {'player' : [], 'opponent' : []}
        # losses = depend on (in)complete information
        self.losses = {'player' : [], 'opponent' : []}
        # true loss = loss incured by the player + action
        self.bandit_losses = {'player' : [], 'opponent' : []}
        self.comp_losses = {'player' : [], 'opponent' : []}

    # set player and opponent

    def set_player(self, player : Actor):
        self.actors['player'] = player
        self.actors['player'].set_simulation(self)

    def set_opponent(self, opponent : Actor):
        self.actors['opponent'] = opponent
        self.actors['opponent'].set_simulation(self)

    # "bandit" information : only get loss of the two actions taken

    def get_bandit_loss(self, _actor = 'player'):
        # return L(I_t, J_t)
        it, jt = self.actions['player'][-1], self.actions['opponent'][-1]   
        return self.loss_matrices[_actor][it, jt]

    # complete information : get the losses of all actions given other player's action

    def get_complete_loss(self, _actor = 'player'):
        # return L[:, J_t] if player and L[I_t, :] if opponent
        it, jt = self.actions['player'][-1], self.actions['opponent'][-1]
        if _actor == 'player':
            return self.loss_matrices[_actor][:, jt]
        elif _actor == 'opponent':
            return self.loss_matrices[_actor][it, :] 

    def run(self):
        # alternatively run player and opponent + update their strategies
        for t in range(self.T):
            self.actions['player'].append(self.actors['player'].play())
            self.actions['opponent'].append(self.actors['opponent'].play())
            
            self.update()

    # update losses w.r.t last actions played

    def update(self):
        # each player computes its loss and updates its strategy
        for _actor in ['player', 'opponent']:
            actor = self.actors[_actor]
            loss = None

            # loss depends on bandit or complete information
            # (will either be real number or vector)
            if actor.complete_info:
                loss = self.get_complete_loss(_actor) 
            else:
                loss = self.get_bandit_loss(_actor)
            # if min loss has to be 0 (e.g with Exp3)
            if actor.strategy.b_require_minloss:
                loss -= self.minlosses[_actor]

            self.losses[_actor].append(loss)
            
            self.bandit_losses[_actor].append(self.get_bandit_loss(_actor))
            self.comp_losses[_actor].append(self.get_complete_loss(_actor))
            actor.update(loss)
