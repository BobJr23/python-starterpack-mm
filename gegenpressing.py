from objects import Action, PlayerRole
import math

def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def norm(v):
    d = math.hypot(v[0], v[1])
    if d == 0:
        return (0, 0)
    return (v[0]/d, v[1]/d)

def add(a, b): return (a[0]+b[0], a[1]+b[1])
def sub(a, b): return (a[0]-b[0], a[1]-b[1])
def mul(a, s): return (a[0]*s, a[1]*s)

class GegenpressStrategy:
    def __init__(self):
        self.params = {
            "press_radius": 20,
            "tackle_dist": 1.5,
            "mark_buffer": 2.0,
            "cover_depth": 10
        }

    def decide(self, game_state, team_id):
        """
        Returns dict: {player_id: Action}
        """
        my_players = [p for p in game_state.players if p.team == team_id]
        opp_players = [p for p in game_state.players if p.team != team_id]
        ball = game_state.ball

        actions = {}

        # --- find ball possessor (if any) ---
        possessor = None
        for opp in opp_players:
            if opp.has_ball:
                possessor = opp
                break
        for me in my_players:
            if me.has_ball:
                possessor = me
                break

        # --- if opponent has the ball: gegenpress ---
        if possessor and possessor.team != team_id:
            poss_pos = (possessor.x, possessor.y)
            # find nearest of my players to press
            presser = min(my_players, key=lambda p: dist((p.x, p.y), poss_pos))
            for me in my_players:
                me_pos = (me.x, me.y)
                if me.id == presser.id:
                    # PRESS
                    d = dist(me_pos, poss_pos)
                    tackle = d < self.params["tackle_dist"]
                    actions[me.id] = Action.move_towards(poss_pos, sprint=True, tackle=tackle)
                else:
                    # others: mark or cut lane
                    opps = [o for o in opp_players if o.id != possessor.id]
                    if opps:
                        # mark nearest opponent
                        target = min(opps, key=lambda o: dist(me_pos, (o.x, o.y)))
                        offset = norm(sub((target.x, target.y), game_state.my_goal(team_id)))
                        mark_spot = add((target.x, target.y), mul(offset, self.params["mark_buffer"]))
                        actions[me.id] = Action.move_towards(mark_spot)
                    else:
                        # fallback: cover center near own goal
                        gx, gy = game_state.my_goal(team_id)
                        cover_spot = (gx + self.params["cover_depth"], gy)
                        actions[me.id] = Action.move_towards(cover_spot)

        # --- if ball is free: intercept ---
        elif not possessor:
            ball_pos = (ball.x, ball.y)
            nearest = min(my_players, key=lambda p: dist((p.x, p.y), ball_pos))
            for me in my_players:
                if me.id == nearest.id:
                    actions[me.id] = Action.move_towards(ball_pos, sprint=True, tackle=True)
                else:
                    # block lanes toward goal
                    gx, gy = game_state.my_goal(team_id)
                    cut_point = ((ball.x + gx)/2, (ball.y + gy)/2)
                    actions[me.id] = Action.move_towards(cut_point)

        # --- if my team has ball: simple support ---
        else:
            # very simple possession logic (can be improved)
            for me in my_players:
                if me.has_ball:
                    actions[me.id] = Action.shoot(game_state.opp_goal(team_id))
                else:
                    actions[me.id] = Action.move_towards((ball.x, ball.y))

        return actions