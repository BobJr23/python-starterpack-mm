from . import *
# from strategy.opposing_strategy import opp_strat

def GetGoalieAction(game: GameState) -> PlayerAction:
    """
    Goalkeeper positioning on the edges of the penalty box rectangle
    """
    config = get_config()
    goalie = game.players[1]  # Assuming goalie is player 0
    ball_pos = game.ball.pos
    goal_center = config.field.goal_self()
    field = config.field.bottom_right()
    
    # Penalty box dimensions
    penalty_box_width = config.goal.penalty_box_width  # Distance from goal line
    penalty_box_height = config.goal.penalty_box_height  # Height of the box
    
    # Define penalty box rectangle edges
    penalty_left_x = goal_center.x
    penalty_right_x = goal_center.x + penalty_box_width
    penalty_top_y = goal_center.y - penalty_box_height / 2
    penalty_bottom_y = goal_center.y + penalty_box_height / 2
    
    # Calculate optimal position on penalty box edge based on ball position
    ball_to_goal = goal_center - ball_pos
    
    # Default to right edge positioning (most common)
    if ball_pos.x <= goal_center.x:
        # Ball behind goal - stay on right edge, centered
        target_pos = Vec2(penalty_right_x, goal_center.y)
    else:
        # Ball in front - find best edge position
        if abs(ball_to_goal.x) < 0.001:  # Ball directly in line with goal
            target_y = goal_center.y
            target_pos = Vec2(penalty_right_x, target_y)
        else:
            # Calculate where ball-to-goal line intersects penalty box right edge
            t = (penalty_right_x - goal_center.x) / ball_to_goal.x
            target_y = goal_center.y + t * ball_to_goal.y
            
            # Check if intersection is within penalty box height
            if penalty_top_y <= target_y <= penalty_bottom_y:
                # Position on right edge
                target_pos = Vec2(penalty_right_x, target_y)
            elif target_y < penalty_top_y:
                # Position on top edge
                if abs(ball_to_goal.y) > 0.001:
                    t_top = (penalty_top_y - goal_center.y) / ball_to_goal.y
                    target_x = goal_center.x + t_top * ball_to_goal.x
                    target_x = max(penalty_left_x, min(penalty_right_x, target_x))
                    target_pos = Vec2(target_x, penalty_top_y)
                else:
                    target_pos = Vec2(penalty_right_x, penalty_top_y)
            else:
                # Position on bottom edge
                if abs(ball_to_goal.y) > 0.001:
                    t_bottom = (penalty_bottom_y - goal_center.y) / ball_to_goal.y
                    target_x = goal_center.x + t_bottom * ball_to_goal.x
                    target_x = max(penalty_left_x, min(penalty_right_x, target_x))
                    target_pos = Vec2(target_x, penalty_bottom_y)
                else:
                    target_pos = Vec2(penalty_right_x, penalty_bottom_y)
    
    # Calculate movement toward target position
    movement = target_pos - goalie.pos
    
    # If ball is very close and goalie can reach it, go for the ball
    ball_distance = (ball_pos - goalie.pos).norm()
        
    # If we have the ball, try simple actions
    if ball_distance < config.player.pickup_radius:
        return PlayerAction(Vec2(0, 0), config.field.goal_other() - goalie.pos)
    
    # Limit movement speed for controlled positioning
    max_movement = 1
    if movement.norm() > max_movement:
        movement = movement.normalize() * max_movement
    
    return PlayerAction(movement, None)

def get_strategy(team: int):
    """This function tells the engine what strategy you want your bot to use                else:
                    
    """
    if team == 0:
        print("Hello! I am team A (on the left)")
        return Strategy(cheese_formation, modified_strategy)
    else:
        print("Hello! I am team B (on the right)")
        return Strategy(cheese_formation, modified_strategy)
    
    # NOTE when actually submitting your bot, you probably want to have the SAME strategy for both
    # sides.

def cheese_formation(score: Score) -> List[Vec2]:
    """The engine will call this function every time the field is reset:
    either after a goal, if the ball has not moved for too long, or right before endgame"""
    
    config = get_config()
    field = config.field.bottom_right()
    
    return [
        Vec2(field.x * 0.3, field.y * 0.5),   # Player 0: Ball rusher - closer to center
        Vec2(field.x * 0.25, field.y * 0.85),  # Player 1: Back corner receiver - in back corner
        Vec2(field.x * 0.5, field.y * 0.9), # Player 2: Side field runner - side position
        Vec2(field.x * 0.5, field.y * 0.9),   # Player 3: Support - defensive position
    ]

def is_passing_lane_clear(passer_pos: Vec2, receiver_pos: Vec2, game: GameState, config) -> bool:
    """Check if the passing lane between two players is clear of opponents"""
    
    # Get all opponent positions (players 4-7 are opponents)
    opponents = [game.players[i].pos for i in range(NUM_PLAYERS, 2 * NUM_PLAYERS)]
    
    # Calculate pass direction and distance
    pass_vector = receiver_pos - passer_pos
    pass_distance = pass_vector.norm()
    
    if pass_distance < 1.0:  # Too close, consider clear
        return True
    
    pass_direction = pass_vector.normalize()
    
    # Check if any opponent is close to the passing lane
    obstruction_radius = config.player.radius * 2.5  # 4x player radius for safety margin
    
    for opp_pos in opponents:
        # Vector from passer to opponent
        to_opponent = opp_pos - passer_pos
        
        # Project opponent position onto the pass line
        projection_length = to_opponent.dot(pass_direction)
        
        # Only consider opponents between passer and receiver
        if 0 <= projection_length <= pass_distance:
            # Calculate perpendicular distance from opponent to pass line
            projection_point = passer_pos + pass_direction * projection_length
            perpendicular_distance = (opp_pos - projection_point).norm()
            
            # If opponent is too close to the pass line, it's obstructed
            if perpendicular_distance < obstruction_radius:
                return False
    
    return True

def find_best_pass_target(passer_pos: Vec2, game: GameState, config) -> int:
    """Find the nearest teammate with an unobstructed passing lane and no opponents nearby"""
    
    # Get all teammates (excluding the passer)
    passer_id = None
    for i in range(NUM_PLAYERS):
        if (game.players[i].pos - passer_pos).norm() < 1.0:  # Find which player is the passer
            passer_id = i
            break
    
    if passer_id is None:
        return 1  # Fallback to player 1
    
    # Check all other teammates
    candidates = []
    for j in range(NUM_PLAYERS):
        if j != passer_id:
            teammate_pos = game.players[j].pos
            distance = (teammate_pos - passer_pos).norm()
            is_lane_clear = is_passing_lane_clear(passer_pos, teammate_pos, game, config)
            
            # Check if receiver has opponents nearby
            nearest_opponent_to_receiver = min(
                (game.players[k].pos - teammate_pos).norm() 
                for k in range(NUM_PLAYERS, 2 * NUM_PLAYERS)  # Opponents are players 4-7
            )
            receiver_is_safe = nearest_opponent_to_receiver > 80.0  # 80 unit safety radius around receiver
            
            candidates.append({
                'id': j,
                'distance': distance,
                'is_clear': is_lane_clear,
                'receiver_safe': receiver_is_safe,
                'opponent_dist': nearest_opponent_to_receiver,
                'pos': teammate_pos
            })
    
    # Sort by priority: clear lane AND safe receiver first, then by distance
    candidates.sort(key=lambda x: (not (x['is_clear'] and x['receiver_safe']), not x['is_clear'], x['distance']))
    
    # Return the best candidate
    if candidates:
        best = candidates[0]
        return best['id']
    
    return 1  # Fallback


def is_shot_blocked(start_pos, target_pos, enemies, config):
    """Simple function to check if anyone is blocking a shot path"""
    shot_vector = target_pos - start_pos
    shot_distance = shot_vector.norm()
    
    if shot_distance == 0:
        return False
    
    shot_direction = shot_vector.normalize()
    
    for enemy in enemies:
        # Vector from shot start to enemy
        to_enemy = enemy.pos - start_pos
        
        # Project enemy onto shot line
        projection = to_enemy.dot(shot_direction)
        
        # Check if enemy is along the shot path (not behind shooter)
        if 0 < projection < shot_distance:
            # Distance from enemy to shot line (2D cross product: ax*by - ay*bx)
            distance_to_line = abs(to_enemy.x * shot_direction.y - to_enemy.y * shot_direction.x)
            
            # If enemy is close enough to block shot
            if distance_to_line < config.player.radius * 2.5:  # 2.5x player radius for safety margin
                return True
    
    return False


def modified_strategy(game: GameState) -> List[PlayerAction]:
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
                if player_pos.x != field.x * 0.25 or player_pos.y != field.y * 0.85:
                    actions.append(GetGoalieAction(game))
                    continue
                      # Stay in position (like a goalie)
                    

            
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
                movement = Vec2(1, 1)
        
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
                if to_goal.norm() < 270.0:  # Only shoot if within 280 units of goal
                    if is_shot_blocked(player_pos, enemy_goal, game.players[NUM_PLAYERS:], config):
                        # Shot is blocked - try wall shot to corner
                        pass_target = (game.players[3].pos - player_pos).normalize()  # Default to passing to player 3 if blocked
                    else:
                        pass_target = to_goal.normalize()
                else:
                    pass_target = Vec2(0.0, 0.0)  # Default forward
                    movement = Vec2(1.0, 0.0)  # Move toward goal while shooting
        
        else:  # Player 3: screener
            if ball_holder != 3:
                # Move up the side field toward goal

                ball_distance = (ball_pos - player_pos).norm()
                if ball_distance < 69.0:
                    movement = (ball_pos - player_pos).normalize()
                else:
                    side_target = Vec2(field.x * 0.93, field.y * 0.69)  # Side field position
                    to_side = side_target - player_pos
                    if to_side.norm() > 0:
                        movement = to_side

                    if player_pos.x >= field.x * 0.90 and player_pos.y < field.y * 0.8:
                        # Try for a wall shot to top right corner
                        movement = Vec2(0,-1)  # Stay in position to shoot
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