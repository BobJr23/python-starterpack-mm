from strategy.gegenpress import GegenpressStrategy

class Strategy:
    def __init__(self):
        self.impl = GegenpressStrategy()

    def decide(self, game_state, team_id):
        return self.impl.decide(game_state, team_id)
