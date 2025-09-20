from . import *

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
        return Strategy(modified_formation, modified_strategy)
    else:
        print("Hello! I am team B (on the right)")
        return Strategy(modified_formation, modified_strategy)
    
    # NOTE when actually submitting your bot, you probably want to have the SAME strategy for both
    # sides.

def modified_formation(score: Score) -> List[Vec2]:
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


    """The engine will call this function every time the field is reset:
    either after a goal, if the ball has not moved for too long, or right before endgame"""
    
    config = get_config()
    field = config.field.bottom_right()
    
    return [
        Vec2(field.x * 0.1, field.y * 0.65),
        Vec2(field.x * 0.4, field.y * 0.5),
        Vec2(field.x * 0.05, field.y * 0.8),
        Vec2(field.x * 0.4, field.y * 0.6),
    ]

    """The engine will call this function every time the field is reset:
    either after a goal, if the ball has not moved for too long, or right before endgame"""
    
    config = get_config()
    field = config.field.bottom_right()
    
    # Optimized starting positions for new_strategy roles:
    return [
        Vec2(field.x * 0.1, field.y * 0.5),  # Player 1: Back corner receiver - in back corner
        Vec2(field.x * 0.4, field.y * 0.5),   # Player 0: Ball rusher - closer to center
        Vec2(field.x * 0.4, field.y * 0.9), # Player 2: Side field runner - side position
        Vec2(field.x * 0.4, field.y * 0.1),   # Player 3: Support - defensive position
    ]

    """Enhanced strategy with intelligent support players"""
    
    config = get_config()
    actions = []
    
    # Goalkeeper action
    actions.append(GetGoalieAction(game))
    
    # Main striker (player 1)
    player_pos = game.players[1].pos
    field = config.field.bottom_right()
    enemy_goal = config.field.goal_other()
    goal_height = config.goal.normal_height
    ball_pos = game.ball.pos
    
    # Check who has the ball
    ball_holder = None
    ball_distance_threshold = config.player.pickup_radius * 1.2
    
    # Check our team for ball possession
    for i in range(NUM_PLAYERS // 2):
        if (game.players[i].pos - ball_pos).norm() <= ball_distance_threshold:
            ball_holder = ('our_team', i)
            break
    
    # Check enemy team for ball possession
    if ball_holder is None:
        for i in range(NUM_PLAYERS // 2, NUM_PLAYERS):
            if (game.players[i].pos - ball_pos).norm() <= ball_distance_threshold:
                ball_holder = ('enemy_team', i)
                break
    
    # Main striker behavior (player 1)
    goal_corners = [
        Vec2(enemy_goal.x, enemy_goal.y - goal_height/2),  # Top goal corner
        Vec2(enemy_goal.x, enemy_goal.y + goal_height/2),  # Bottom goal corner
    ]
    import random
    target_corner = random.choice(goal_corners)
    
    enemies = [game.players[i] for i in range(NUM_PLAYERS // 2, NUM_PLAYERS)]
    wall_shot = calculate_wall_shot_to_corner(player_pos, target_corner, field, enemies, config)
    
    if wall_shot is not None:
        actions.append(PlayerAction(ball_pos - player_pos, wall_shot))
    else:
        # Fallback to direct shot
        direct_shot = (target_corner - player_pos).normalize()
        actions.append(PlayerAction(ball_pos - player_pos, direct_shot))
    
    # Support players (2 and 3) behavior
    for support_idx in [2, 3]:
        actions.append(PlayerAction(Vec2(0, 0), None))  # Placeholder to maintain order
        continue
        if support_idx >= NUM_PLAYERS:
            actions.append(PlayerAction(Vec2(0, 0), None))
            continue
            
        support_pos = game.players[support_idx].pos
        
        if ball_holder is None:
            # Ball is free - run towards it
            to_ball = ball_pos - support_pos
            if to_ball.norm() > 0:
                movement = to_ball.normalize()
            else:
                movement = Vec2(0, 0)
            actions.append(PlayerAction(movement, None))
            
        elif ball_holder[0] == 'our_team':
            # Our team has the ball
            if ball_holder[1] == support_idx:  # This support player has the ball
                # 1. Check for open shot opportunities
                goal_corners = [
                    Vec2(enemy_goal.x, enemy_goal.y - goal_height/2),  # Top corner
                    Vec2(enemy_goal.x, enemy_goal.y + goal_height/2),  # Bottom corner
                ]
                
                has_open_shot = False
                shot_direction = None
                
                # Check direct shots to corners
                for corner in goal_corners:
                    if not is_shot_blocked(support_pos, corner, enemies, config):
                        has_open_shot = True
                        shot_direction = (corner - support_pos).normalize()
                        break
                
                # If no direct shot, try wall shots
                if not has_open_shot:
                    for corner in goal_corners:
                        wall_shot = calculate_wall_shot_to_corner(support_pos, corner, field, enemies, config)
                        if wall_shot is not None:
                            has_open_shot = True
                            shot_direction = wall_shot
                            break
                
                if has_open_shot:
                    # Take the shot
                    actions.append(PlayerAction(Vec2(0, 0), shot_direction))
                else:
                    # 2. No open shot - count nearby enemies
                    nearby_enemies = []
                    enemy_threshold = config.player.radius * 8  # Define "nearby"
                    
                    for enemy in enemies:
                        if (enemy.pos - support_pos).norm() <= enemy_threshold:
                            nearby_enemies.append(enemy)
                    
                    if len(nearby_enemies) >= 2:
                        # 3. Two or more enemies nearby - look for pass
                        best_pass_target = None
                        best_pass_score = -1
                        
                        # Check teammates for passing options
                        for teammate_idx in range(NUM_PLAYERS // 2):
                            if teammate_idx == support_idx:  # Don't pass to self
                                continue
                            
                            teammate_pos = game.players[teammate_idx].pos
                            
                            # Check if pass is clear
                            if not is_shot_blocked(support_pos, teammate_pos, enemies, config):
                                # Calculate pass score (closer to goal is better, more open is better)
                                goal_distance = (enemy_goal - teammate_pos).norm()
                                goal_score = 1.0 - (goal_distance / field.norm())
                                
                                # Check how open teammate is
                                min_enemy_dist = min((enemy.pos - teammate_pos).norm() for enemy in enemies)
                                openness_score = min(1.0, min_enemy_dist / (config.player.radius * 6))
                                
                                total_score = goal_score * 0.6 + openness_score * 0.4
                                
                                if total_score > best_pass_score:
                                    best_pass_score = total_score
                                    best_pass_target = teammate_pos
                        
                        if best_pass_target is not None:
                            # Make the pass
                            pass_direction = (best_pass_target - support_pos).normalize()
                            actions.append(PlayerAction(Vec2(0, 0), pass_direction))
                        else:
                            # No good pass - dribble toward goal
                            dribble_direction = (enemy_goal - support_pos).normalize() * 0.8
                            actions.append(PlayerAction(dribble_direction, None))
                    else:
                        # 4. One or no enemies nearby - move up and try to score
                        # Move toward goal while looking for shooting opportunity
                        to_goal = (enemy_goal - support_pos).normalize()
                        
                        # Try to get to a better shooting position
                        ideal_shooting_x = field.x * 0.75
                        if support_pos.x < ideal_shooting_x:
                            # Move forward toward ideal shooting range
                            movement = to_goal * 0.8
                        else:
                            # Close to goal - look for angle
                            if support_pos.y < field.y * 0.4:
                                # Move toward center-top
                                movement = Vec2(to_goal.x * 0.5, 0.5)
                            elif support_pos.y > field.y * 0.6:
                                # Move toward center-bottom  
                                movement = Vec2(to_goal.x * 0.5, -0.5)
                            else:
                                # Move straight forward
                                movement = to_goal
                        
                        if movement.norm() > 1.0:
                            movement = movement.normalize()
                        actions.append(PlayerAction(movement, None))
                        
            elif ball_holder[1] == 1:  # Main striker (player 1) has ball
                # Get open and move down field to receive pass
                # Move towards enemy goal but spread out
                offset = 0.3 if support_idx == 2 else -0.3  # Spread vertically
                target_x = min(field.x * 0.8, support_pos.x + field.x * 0.2)  # Move forward
                target_y = field.y * 0.5 + field.y * offset  # Spread out
                target_pos = Vec2(target_x, target_y)
                
                # Avoid getting too close to enemies while getting open
                min_enemy_distance = float('inf')
                for enemy in enemies:
                    dist = (enemy.pos - support_pos).norm()
                    if dist < min_enemy_distance:
                        min_enemy_distance = dist
                
                # If too close to enemy, create space
                if min_enemy_distance < config.player.radius * 4:
                    # Find direction away from nearest enemy
                    nearest_enemy = min(enemies, key=lambda e: (e.pos - support_pos).norm())
                    away_from_enemy = (support_pos - nearest_enemy.pos).normalize()
                    target_pos = support_pos + away_from_enemy * config.player.radius * 2
                
                movement = target_pos - support_pos
                if movement.norm() > 1.0:
                    movement = movement.normalize()
                actions.append(PlayerAction(movement, None))
            else:
                # Another teammate has the ball - support positioning
                movement = (enemy_goal - support_pos).normalize() * 0.5
                actions.append(PlayerAction(movement, None))
                
        else:
            # Enemy has the ball - mark free enemies and intercept passes
            enemy_with_ball_idx = ball_holder[1]
            enemy_with_ball = game.players[enemy_with_ball_idx]
            
            # Find unmarked enemies (excluding the one with ball)
            unmarked_enemies = []
            for i in range(NUM_PLAYERS // 2, NUM_PLAYERS):
                if i != enemy_with_ball_idx:
                    enemy_pos = game.players[i].pos
                    # Check if this enemy is already being marked by other support player
                    other_support_idx = 3 if support_idx == 2 else 2
                    if other_support_idx < NUM_PLAYERS:
                        other_support_pos = game.players[other_support_idx].pos
                        distance_to_other_support = (enemy_pos - other_support_pos).norm()
                        if distance_to_other_support > config.player.radius * 3:  # Not being marked
                            unmarked_enemies.append(game.players[i])
            
            if unmarked_enemies:
                # Mark the nearest unmarked enemy
                target_enemy = min(unmarked_enemies, key=lambda e: (e.pos - support_pos).norm())
                
                # Position between the enemy and the ball to intercept passes
                ball_to_enemy = target_enemy.pos - ball_pos
                if ball_to_enemy.norm() > 0:
                    # Position slightly closer to ball than the enemy
                    intercept_pos = ball_pos + ball_to_enemy.normalize() * (ball_to_enemy.norm() * 0.7)
                    movement = intercept_pos - support_pos
                else:
                    # If enemy is on ball, just get close to mark them
                    movement = target_enemy.pos - support_pos
                    
                if movement.norm() > 1.0:
                    movement = movement.normalize()
                actions.append(PlayerAction(movement, None))
            else:
                # All enemies marked or none to mark - move toward ball
                to_ball = ball_pos - support_pos
                if to_ball.norm() > 0:
                    movement = to_ball.normalize() * 0.8
                else:
                    movement = Vec2(0, 0)
                actions.append(PlayerAction(movement, None))
    
    # Fill remaining players if any
    while len(actions) < NUM_PLAYERS:
        actions.append(PlayerAction(Vec2(0, 0), None))
    
    return actions


    """This strategy will do nothing :("""
    
    return [
        PlayerAction(Vec2(0, 0), None) 
        for _ in range(NUM_PLAYERS)
    ]

    """Goalie strategy: Player 0 is goalie, others chase ball"""
    
    actions = [GetGoalieAction(game)]
    
    for i in range(1, NUM_PLAYERS):
        player_pos = game.players[i].pos
        ball_pos = game.ball.pos
        movement = ball_pos - player_pos
        
        # Always move at full speed (normalize movement vector)
        if movement.norm() > 0:
            movement = movement.normalize()
        else:
            # If no movement direction, default to moving forward
            movement = Vec2(1.0, 0.0)
        
        actions.append(PlayerAction(movement, None))
    
    return actions

    """Calculate the optimal point to intercept a moving ball"""
    
    # If ball isn't moving much, go to current position
    if ball_velocity.norm() < 0.001:
        return ball_pos
    
    # Simple interception: try a few future times and pick the best one
    best_intercept = ball_pos
    best_ratio = float('inf')  # ratio of player_time / ball_time (want this close to 1.0)
    
    # Check multiple time points in the future
    for t in [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
        # Calculate where ball will be at time t
        ball_speed_at_t = max(0, ball_velocity.norm() - ball_friction * t)
        
        if ball_speed_at_t <= 0:
            # Ball has stopped due to friction
            stopping_time = ball_velocity.norm() / ball_friction if ball_friction > 0 else 0
            if t >= stopping_time:
                ball_direction = ball_velocity.normalize() if ball_velocity.norm() > 0 else Vec2(1, 0)
                stopping_distance = (ball_velocity.norm() ** 2) / (2 * ball_friction) if ball_friction > 0 else 0
                predicted_ball_pos = ball_pos + ball_direction * stopping_distance
            else:
                ball_direction = ball_velocity.normalize()
                distance_traveled = ball_velocity.norm() * t - 0.5 * ball_friction * t * t
                predicted_ball_pos = ball_pos + ball_direction * max(0, distance_traveled)
        else:
            # Ball is still moving
            ball_direction = ball_velocity.normalize()
            distance_traveled = ball_velocity.norm() * t - 0.5 * ball_friction * t * t
            predicted_ball_pos = ball_pos + ball_direction * max(0, distance_traveled)
        
        # Calculate how long it takes player to reach that position
        distance_to_intercept = (predicted_ball_pos - player_pos).norm()
        player_time = distance_to_intercept / player_speed
        
        # We want player_time to be close to t (ball arrival time)
        time_ratio = abs(player_time - t) / max(t, 0.1)  # normalized difference
        
        if time_ratio < best_ratio:
            best_ratio = time_ratio
            best_intercept = predicted_ball_pos
    
    # If we can't find a good intercept, aim ahead of the ball
    if best_ratio > 2.0:  # If no good intercept found
        # Aim ahead of ball by predicting 0.5 seconds into future
        ball_direction = ball_velocity.normalize() if ball_velocity.norm() > 0 else Vec2(1, 0)
        future_distance = ball_velocity.norm() * 0.5 - 0.5 * ball_friction * 0.25
        best_intercept = ball_pos + ball_direction * max(0, future_distance)
    
    return best_intercept

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
        
        if i == 1:  # Ball rusher
            if ball_holder not in (0, 1,2,3):  # Neither we nor our teammate has the ball
                # Debug the ball movement
                ball_velocity = game.ball.vel
                ball_speed = ball_velocity.norm()
                
                
                if ball_speed < 0.05:
                    # Ball is stationary, go directly to it
                    target_pos = ball_pos
                else:
                    # Calculate intersection point where player can intercept ball
                    ball_direction = ball_velocity.normalize()
                    player_speed = config.player.speed
                    ball_friction = config.ball.friction
                    
                    # Try different distances ahead to find where player can intercept
                    best_distance = 800.0  # fallback - increased from 500
                    best_time_diff = float('inf')
                    
                    # Test longer distances from 200 to 1200 units ahead (increased range)
                    for test_distance in range(0, 1201, 75):
                        # Calculate ball position at this distance
                        # Using kinematic equation: distance = v*t - 0.5*friction*t^2
                        # Solve for time when ball travels test_distance
                        if ball_friction > 0:
                            # Quadratic formula to solve: test_distance = ball_speed*t - 0.5*friction*t^2
                            a = -0.5 * ball_friction
                            b = ball_speed
                            c = -test_distance
                            discriminant = b*b - 4*a*c
                            
                            if discriminant >= 0:
                                t1 = (-b + (discriminant**0.5)) / (2*a)
                                t2 = (-b - (discriminant**0.5)) / (2*a)
                                ball_time = min([t for t in [t1, t2] if t > 0], default=test_distance/ball_speed)
                            else:
                                ball_time = test_distance / ball_speed
                        else:
                            ball_time = test_distance / ball_speed if ball_speed > 0 else float('inf')
                        
                        # Calculate predicted ball position
                        predicted_pos = ball_pos + ball_direction * test_distance
                        
                        # Calculate time for player to reach that position
                        player_distance = (predicted_pos - player_pos).norm()
                        player_time = player_distance / player_speed
                        
                        # Find the distance where player and ball arrive at similar times
                        # Add a slight bias toward longer distances (more aggressive leading)
                        time_diff = abs(player_time - ball_time)
                        lead_bonus = test_distance / 10000.0  # Small bonus for longer distances
                        adjusted_time_diff = time_diff - lead_bonus
                        
                        if adjusted_time_diff < best_time_diff:
                            best_time_diff = adjusted_time_diff
                            best_distance = test_distance
                    
                    target_pos = ball_pos + ball_direction * best_distance
                    
                
                to_target = target_pos - player_pos
                movement = to_target
                
            else:
                # Has the ball - always pass immediately (ball rusher's job is to get ball to teammates)
                best_target_id = find_best_pass_target(player_pos, game, config)
                # If ball is within 10 units of the center (500, 300), fallback to player 0
                if (ball_pos - Vec2(500, 300)).norm() < 10:
                    best_target_id = 0  # Fallback if ball position is near center
                target_pos = game.players[best_target_id].pos
                
                pass_direction = target_pos - player_pos
                if pass_direction.norm() > 0:
                    pass_target = pass_direction.normalize()
                else:
                    pass_target = Vec2(1.0, 0.0)  # Default forward pass
                
                # Move slightly toward goal while passing
                to_goal = enemy_goal - player_pos
                if to_goal.norm() > 0:
                    movement = to_goal * 0.3
                    
        
        elif i == 0:  # Back corner receiver
            if ball_holder != 0:
                actions.append(GetGoalieAction(game))  # Stay in position (like a goalie)
                continue

            else:
                # Has the ball - check if opponents are nearby (under pressure)
                nearest_opponent_distance = min(
                    (game.players[j].pos - player_pos).norm() 
                    for j in range(NUM_PLAYERS, 2 * NUM_PLAYERS)  # Opponents are players 4-7
                )
                
                if nearest_opponent_distance < 100.0:  # Pass under pressure
                    # Pass to side field runner (player 2) or best available
                    best_target_id = find_best_pass_target(player_pos, game, config)
                    target_pos = game.players[best_target_id].pos
                    pass_direction = target_pos - player_pos
                    if pass_direction.norm() > 0:
                        pass_target = pass_direction.normalize()
                    else:
                        pass_target = Vec2(1.0, 0.0)  # Default forward pass
                    # Stay in position while passing
                    movement = Vec2(0, 0)
                    
                else:
                    # No pressure - dribble toward goal
                    to_goal = enemy_goal - player_pos
                    movement = to_goal
                    pass_target = None  # Don't pass
                    
        
        elif i == 2:  # Side field runner
            if ball_holder != 2:
                # Check if ball is close enough to chase
                ball_distance = (ball_pos - player_pos).norm()
                chase_radius = 180.0  # Chase ball if within 180 units (larger for aggressive player)
                
                if ball_distance < chase_radius and ball_holder is None:
                    # Ball is close and free - go for it!
                    to_ball = ball_pos - player_pos
                    movement = to_ball
                else:
                    # Move up the side field toward goal
                    side_target = Vec2(field.x * 0.4, field.y * 0.90)  # Side field position
                    to_side = side_target - player_pos
                    if to_side.norm() > 0:
                        movement = to_side
            else:
                # Has the ball - check if opponents are nearby (under pressure)
                nearest_opponent_distance = min(
                    (game.players[j].pos - player_pos).norm() 
                    for j in range(NUM_PLAYERS, 2 * NUM_PLAYERS)  # Opponents are players 4-7
                )
                
                to_goal = enemy_goal - player_pos
                
                if nearest_opponent_distance < 130.0:  # Under pressure - try to shoot or pass
                    shooting_lane_clear = is_passing_lane_clear(player_pos, enemy_goal, game, config)
                    
                    if shooting_lane_clear and to_goal.norm() < 400.0:  # Shoot if clear and within range
                        pass_target = to_goal.normalize()
                        movement = to_goal * 0.3  # Move toward goal while shooting
                        
                       
                    else:
                        # Shooting lane blocked - pass to best available teammate
                        best_target_id = find_best_pass_target(player_pos, game, config)
                        target_pos = game.players[best_target_id].pos
                        
                        pass_direction = target_pos - player_pos
                        if pass_direction.norm() > 0:
                            pass_target = pass_direction.normalize()
                        else:
                            pass_target = Vec2(1.0, 0.0)  # Default forward pass
                        
                        # Continue moving toward goal position while passing
                        movement = to_goal * 0.5
                        
                else:
                    # No pressure - dribble toward goal
                    movement = to_goal
                    pass_target = None  # Don't pass or shoot
                    
        
        else:  # Player 3: Support/backup
            if ball_holder != 3:
                # Check if ball is close enough to chase
                ball_distance = (ball_pos - player_pos).norm()
                chase_radius = 160.0  # Chase ball if within 160 units
                
                if ball_distance < chase_radius and ball_holder is None:
                    # Ball is close and free - go for it!
                    to_ball = ball_pos - player_pos
                    movement = to_ball
                else:
                    # Move up the side field toward goal
                    side_target = Vec2(field.x * 0.4, field.y * 0.1)  # Side field position
                    to_side = side_target - player_pos
                    if to_side.norm() > 0:
                        movement = to_side
            else:
                # Has the ball - check if opponents are nearby (under pressure)
                nearest_opponent_distance = min(
                    (game.players[j].pos - player_pos).norm() 
                    for j in range(NUM_PLAYERS, 2 * NUM_PLAYERS)  # Opponents are players 4-7
                )
                
                to_goal = enemy_goal - player_pos
                
                if nearest_opponent_distance < 130.0:  # Under pressure - try to shoot or pass
                    shooting_lane_clear = is_passing_lane_clear(player_pos, enemy_goal, game, config)
                    
                    if shooting_lane_clear and to_goal.norm() < 400.0:  # Shoot if clear and within range
                        pass_target = to_goal.normalize()
                        movement = to_goal * 0.3  # Move toward goal while shooting
                        
                    else:
                        # Shooting lane blocked - pass to best available teammate
                        best_target_id = find_best_pass_target(player_pos, game, config)
                        target_pos = game.players[best_target_id].pos
                        
                        pass_direction = target_pos - player_pos
                        if pass_direction.norm() > 0:
                            pass_target = pass_direction.normalize()
                        else:
                            pass_target = Vec2(1.0, 0.0)  # Default forward pass
                        
                        # Continue moving toward goal position while passing
                        movement = to_goal * 0.5
                        
                else:
                    # No pressure - dribble toward goal
                    movement = to_goal
                    pass_target = None  # Don't pass or shoot
                    
        # Always move at full speed (normalize movement vector)
        if movement.norm() > 0:
            movement = movement.normalize()
        else:
            # If no movement direction, default to moving forward
            movement = Vec2(1.0, 0.0)
        
        actions.append(PlayerAction(movement, pass_target))
    
    return actions


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
    """Calculate where to shoot on wall to bounce into corner, checking for blocks"""
    
    # Try top wall bounce (y = 0)
    if player_pos.y > field.y * 0.3 and corner_pos.y != 0:
        wall_hit_x = (corner_pos.x * player_pos.y + player_pos.x * corner_pos.y) / (player_pos.y + corner_pos.y)
        if 0 <= wall_hit_x <= field.x:
            bounce_point = Vec2(wall_hit_x, 0)
            shot_direction = (bounce_point - player_pos).normalize()
            
            # Check if shot is blocked (if enemies provided)
            if enemies and config and is_shot_blocked(player_pos, bounce_point, enemies, config):
                pass  # Try next wall
            else:
                return shot_direction
    
    # Try bottom wall bounce (y = field.y)
    if player_pos.y < field.y * 0.7 and corner_pos.y != field.y:
        denominator = (field.y - player_pos.y) + (corner_pos.y - field.y)
        if abs(denominator) > 0.001:
            wall_hit_x = (corner_pos.x * (field.y - player_pos.y) + player_pos.x * (field.y - corner_pos.y)) / denominator
            if 0 <= wall_hit_x <= field.x:
                bounce_point = Vec2(wall_hit_x, field.y)
                shot_direction = (bounce_point - player_pos).normalize()
                
                # Check if shot is blocked
                if enemies and config and is_shot_blocked(player_pos, bounce_point, enemies, config):
                    pass  # Try next wall
                else:
                    return shot_direction
    
    # Try left wall bounce (x = 0)
    if player_pos.x > field.x * 0.3 and corner_pos.x != 0:
        wall_hit_y = (corner_pos.y * player_pos.x + player_pos.y * corner_pos.x) / (player_pos.x + corner_pos.x)
        if 0 <= wall_hit_y <= field.y:
            bounce_point = Vec2(0, wall_hit_y)
            shot_direction = (bounce_point - player_pos).normalize()
            
            # Check if shot is blocked
            if enemies and config and is_shot_blocked(player_pos, bounce_point, enemies, config):
                pass  # Try next wall
            else:
                return shot_direction
    
    # Try right wall bounce (x = field.x)  
    if player_pos.x < field.x * 0.7 and corner_pos.x != field.x:
        denominator = (field.x - player_pos.x) + (corner_pos.x - field.x)
        if abs(denominator) > 0.001:
            wall_hit_y = (corner_pos.y * (field.x - player_pos.x) + player_pos.y * (field.x - corner_pos.x)) / denominator
            if 0 <= wall_hit_y <= field.y:
                bounce_point = Vec2(field.x, wall_hit_y)
                shot_direction = (bounce_point - player_pos).normalize()
                
                # Check if shot is blocked
                if enemies and config and is_shot_blocked(player_pos, bounce_point, enemies, config):
                    pass  # Try next wall
                else:
                    return shot_direction
    
    # Fallback: shoot toward corner directly (only if not blocked)
    direct_shot = (corner_pos - player_pos).normalize()
    if enemies and config and is_shot_blocked(player_pos, corner_pos, enemies, config):
        return None  # All shots blocked
    
    return direct_shot


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