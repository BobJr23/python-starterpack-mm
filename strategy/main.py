from . import *

def get_strategy(team: int):
    """This function tells the engine what strategy you want your bot to use"""
    
    # team == 0 means I am on the left
    # team == 1 means I am on the right
    
    if team == 0:
        print("Hello! I am team A (on the left)")
        return Strategy(rush_formation, new_strategy)
    else:
        print("Hello! I am team B (on the right)")
        return Strategy(rush_formation, new_strategy)
    
    # NOTE when actually submitting your bot, you probably want to have the SAME strategy for both
    # sides.

def goalee_formation(score: Score) -> List[Vec2]:
    """The engine will call this function every time the field is reset:
    either after a goal, if the ball has not moved for too long, or right before endgame"""
    
    config = get_config()
    field = config.field.bottom_right()
    
    return [
        Vec2(field.x * 0.1, field.y * 0.5),
        Vec2(field.x * 0.4, field.y * 0.4),
        Vec2(field.x * 0.4, field.y * 0.5),
        Vec2(field.x * 0.4, field.y * 0.6),
    ]

def rush_formation(score: Score) -> List[Vec2]:
    """The engine will call this function every time the field is reset:
    either after a goal, if the ball has not moved for too long, or right before endgame"""
    
    config = get_config()
    field = config.field.bottom_right()
    
    # Optimized starting positions for new_strategy roles:
    return [
        Vec2(field.x * 0.3, field.y * 0.5),   # Player 0: Ball rusher - closer to center
        Vec2(field.x * 0.25, field.y * 0.85),  # Player 1: Back corner receiver - in back corner
        Vec2(field.x * 0.5, field.y * 0.9), # Player 2: Side field runner - side position
        Vec2(field.x * 0.5, field.y * 0.05),   # Player 3: Support - defensive position
    ]

def ball_chase(game: GameState) -> List[PlayerAction]:
    """Very simple strategy to chase the ball and shoot on goal"""
    
    config = get_config()
    
    # NOTE Do not worry about what side your bot is on! 
    # The engine mirrors the world for you if you are on the right, 
    # so to you, you always appear on the left.
    
    return [
        PlayerAction(
            game.ball.pos - game.players[i].pos,
            config.field.goal_other() - game.players[i].pos
        ) 
        for i in range(NUM_PLAYERS)
    ]

def do_nothing(game: GameState) -> List[PlayerAction]:
    """This strategy will do nothing :("""
    
    return [
        PlayerAction(Vec2(0, 0), None) 
        for _ in range(NUM_PLAYERS)
    ]

def new_strategy(game: GameState) -> List[PlayerAction]:
    """Fast ball control strategy: Rush ball → Back corner → Side field → Goal shot"""
    
    config = get_config()
    actions = []
    
    # Get field dimensions and key positions
    field = config.field.bottom_right()
    enemy_goal = config.field.goal_other()
    ball_pos = game.ball.pos
    
    # Define player roles:
    # Player 0: Ball rusher (closest to ball at start)
    # Player 1: Back corner receiver
    # Player 2: Side field runner 
    # Player 3: Support/backup
    
    # Calculate distances to ball for all players
    distances_to_ball = [(i, (game.players[i].pos - ball_pos).norm()) for i in range(NUM_PLAYERS)]
    distances_to_ball.sort(key=lambda x: x[1])
    closest_to_ball = distances_to_ball[0][0]
    
    # Determine who has possession
    ball_holder = None
    for i in range(NUM_PLAYERS):
        if (game.players[i].pos - ball_pos).norm() <= config.player.pickup_radius:
            ball_holder = i
            break
    
    for i in range(NUM_PLAYERS):
        player_pos = game.players[i].pos
        movement = Vec2(0, 0)
        pass_target = None
        
        if i == 0:  # Ball rusher
            if ball_holder != 0:
                # Rush to the ball as fast as possible
                to_ball = ball_pos - player_pos
                
                movement = to_ball
            else:
                # Has the ball - pass to nearest teammate
                # nearest_bot = min(
                #     (j for j in range(NUM_PLAYERS) if j != 0),
                #     key=lambda j: (game.players[j].pos - player_pos).norm()
                # )
                # nearest_pos = game.players[nearest_bot].pos
                nearest_pos = game.players[1].pos  # Always pass to player 1 (back corner receiver)
                pass_direction = nearest_pos - player_pos
                if pass_direction.norm() > 0:
                    pass_target = pass_direction.normalize()
                else:
                    pass_target = Vec2(1.0, 0.0)  # Default forward pass
                
                # Move slightly toward goal while passing
                to_goal = enemy_goal - player_pos
                if to_goal.norm() > 0:
                    movement = to_goal * 0.3
        
        elif i == 1:  # Back corner receiver
            if ball_holder != 1:
                # Position in back corner and wait for pass
                back_corner_pos = Vec2(field.x * 0.25, field.y * 0.85)
                to_corner = back_corner_pos - player_pos
                if to_corner.norm() > 0:
                    movement = to_corner
            else:
                # Has the ball - pass to side field runner (player 2)
                side_field_pos = game.players[2].pos
                pass_direction = side_field_pos - player_pos
                if pass_direction.norm() > 0:
                    pass_target = pass_direction.normalize()  # Direction to player 2
                else:
                    pass_target = Vec2(1.0, 0.0)  # Default forward pass
                # Stay in position while passing
                movement = Vec2(0, 0)
        
        elif i == 2:  # Side field runner
            if ball_holder != 2:
                # Move up the side field toward goal
                side_target = Vec2(field.x * 0.8, field.y * 0.92)  # Side field position
                to_side = side_target - player_pos
                if to_side.norm() > 0:
                    movement = to_side
            else:
                # Has the ball - shoot at goal with maximum power
                to_goal = enemy_goal - player_pos
                if to_goal.norm() < 260.0:  # Only shoot if within 100 units of goal
                    pass_target = to_goal.normalize()
                else:
                    pass_target = Vec2(0.0, 0.0)  # Default forward
                    movement = Vec2(1.0, 0.0)  # Move toward goal while shooting
        
        else:  # Player 3: Support/backup
            if ball_holder != 3:
                # Move up the side field toward goal
                side_target = Vec2(field.x * 0.8, field.y * 0.1)  # Side field position
                to_side = side_target - player_pos
                if to_side.norm() > 0:
                    movement = to_side
            else:
                # Has the ball - shoot at goal with maximum power
                to_goal = enemy_goal - player_pos
                if to_goal.norm() > 0:
                    pass_target = to_goal.normalize()  # Shoot at goal
                else:
                    pass_target = Vec2(1.0, 0.0)  # Default forward
                movement = to_goal * 0.5  # Move toward goal while shooting
        
        # Ensure movement doesn't exceed max magnitude
        if movement.norm() > 1.0:
            movement = movement.normalize()
        
        actions.append(PlayerAction(movement, pass_target))
    
    return actions