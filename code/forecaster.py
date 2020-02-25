from pandas import read_csv

class Forecaster:
    def __init__(self, ids_file, votes_files):
        self.ids = read_csv(ids_file)
        self.votes = read_csv(votes_files)
        self.loss = lambda y_true, y_hat : (1 - y_true)*y_hat + y_true * (1 - y_hat)
        self.player = None

    def set_player(self, player):
        assert not (player is None)
        self.player = player

    def play(self):
        # 1) le joueur donnes les probas actuelles p_t
        # 2) A partir des idees, on calcule le vecteur qui contient les 4 entrees non vides
        # 3) Calcul de la prediction du joueur 
        # -> Necessite de normaliser : \hat{y}_t = (p_t(z_1) + p_t(z_2 + N)) / (p_t(z_1) + p_t(z_2 + N) + p_t(z_2) + p_t(z_1 + N))
        # 4) Mise a jour du vecteur f_t (sleeping strategy)
        # 5) Application de la loss sur chacun des p_t(k) du joueur