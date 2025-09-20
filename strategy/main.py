from . import *
import tensorflow as tf
import numpy as np
import json
import os
import random
import math
from typing import Dict, List, Tuple, Optional

class Soccer4v4MLStrategy:
    def __init__(self):
        self.model = None
        self.feature_scaler_params = None
        self.is_trained = False
        
        # Model architecture parameters
        self.input_size = 35  # Comprehensive feature set
        self.hidden_layers = [128, 64, 32]
        self.output_size = 6  # [move_x, move_y, kick_x, kick_y, action_type, confidence]
        
    def build_model(self):
        """Build the neural network for 4v4 soccer strategy"""
        model = tf.keras.Sequential([
            # Input layer with normalization
            tf.keras.layers.Dense(self.hidden_layers[0], 
                                input_shape=(self.input_size,),
                                activation='relu',
                                kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            # Hidden layers
            tf.keras.layers.Dense(self.hidden_layers[1], 
                                activation='relu',
                                kernel_initializer='he_normal'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(self.hidden_layers[2], 
                                activation='relu',
                                kernel_initializer='he_normal'),
            tf.keras.layers.Dropout(0.1),
            
            # Output layer
            tf.keras.layers.Dense(self.output_size, activation='tanh')
        ])
        
        # Custom loss function for soccer strategy
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        return model
    
    def extract_comprehensive_features(self, game_state: Dict, player_idx: int, situation: Dict) -> np.ndarray:
        """Extract comprehensive features for ML model"""
        features = []
        our_players = situation['our_players']
        opponents = situation['opponents']
        player = our_players[player_idx]
        ball_pos = situation['ball_pos']
        field = situation['field']
        
        # === PLAYER STATE FEATURES ===
        # Player position (normalized)
        features.extend([
            player.pos.x / field.x,
            player.pos.y / field.y,
        ])
        
        # === BALL FEATURES ===
        # Ball position and relative position
        features.extend([
            ball_pos.x / field.x,
            ball_pos.y / field.y,
            (ball_pos.x - player.pos.x) / field.x,
            (ball_pos.y - player.pos.y) / field.y,
            (ball_pos - player.pos).norm() / (field.x + field.y),  # Distance to ball
        ])
        
        # === POSSESSION FEATURES ===
        ball_possession = situation['ball_possession']
        features.extend([
            1.0 if ball_possession['team'] == 'us' else 0.0,
            1.0 if ball_possession['team'] == 'opponent' else 0.0,
            1.0 if ball_possession['team'] == 'free' else 0.0,
            1.0 if ball_possession.get('player_idx') == player_idx else 0.0,
            1.0 if ball_possession.get('contested', False) else 0.0,
        ])
        
        # === SPATIAL FEATURES ===
        # Distance to goals
        our_goal = situation['our_goal']
        opp_goal = situation['opponent_goal']
        features.extend([
            (player.pos - our_goal).norm() / (field.x + field.y),
            (player.pos - opp_goal).norm() / (field.x + field.y),
            (ball_pos - our_goal).norm() / (field.x + field.y),
            (ball_pos - opp_goal).norm() / (field.x + field.y),
        ])
        
        # === TEAMMATE FEATURES (3 closest teammates) ===
        teammate_distances = []
        for i, teammate in enumerate(our_players):
            if i != player_idx:
                dist = (teammate.pos - player.pos).norm()
                teammate_distances.append((i, teammate, dist))
        
        teammate_distances.sort(key=lambda x: x[2])
        
        for i in range(3):  # Include 3 closest teammates
            if i < len(teammate_distances):
                _, teammate, dist = teammate_distances[i]
                features.extend([
                    (teammate.pos.x - player.pos.x) / field.x,
                    (teammate.pos.y - player.pos.y) / field.y,
                    dist / (field.x + field.y),
                ])
            else:
                features.extend([0.0, 0.0, 1.0])  # Default values
        
        # === OPPONENT FEATURES (3 closest opponents) ===
        opponent_distances = []
        for opponent in opponents:
            dist = (opponent.pos - player.pos).norm()
            opponent_distances.append((opponent, dist))
        
        opponent_distances.sort(key=lambda x: x[1])
        
        for i in range(3):  # Include 3 closest opponents
            if i < len(opponent_distances):
                opponent, dist = opponent_distances[i]
                features.extend([
                    (opponent.pos.x - player.pos.x) / field.x,
                    (opponent.pos.y - player.pos.y) / field.y,
                    dist / (field.x + field.y),
                ])
            else:
                features.extend([0.0, 0.0, 1.0])  # Default values
        
        # === TACTICAL FEATURES ===
        # Field zones
        features.extend([
            1.0 if situation['ball_zone'] == 'defensive' else 0.0,
            1.0 if situation['ball_zone'] == 'middle' else 0.0,
            1.0 if situation['ball_zone'] == 'attacking' else 0.0,
        ])
        
        # Player role context (derived from position and ball)
        is_goalkeeper = 1.0 if player_idx == 0 else 0.0
        is_closest_to_ball = 1.0 if self._is_closest_to_ball(player_idx, our_players, ball_pos) else 0.0
        
        features.extend([
            is_goalkeeper,
            is_closest_to_ball,
        ])
        
        return np.array(features[:self.input_size], dtype=np.float32)
    
    def _is_closest_to_ball(self, player_idx: int, our_players, ball_pos) -> bool:
        """Check if this player is closest to ball"""
        distances = [(ball_pos - p.pos).norm() for p in our_players]
        return distances[player_idx] == min(distances)
    
    def generate_expert_action(self, game_state: Dict, player_idx: int, situation: Dict) -> np.ndarray:
        """Generate expert action using rule-based strategy"""
        our_players = situation['our_players']
        opponents = situation['opponents']
        player = our_players[player_idx]
        ball_pos = situation['ball_pos']
        ball_possession = situation['ball_possession']
        
        # Use rule-based logic from the advanced strategy
        roles = self._assign_simple_roles(situation)
        action_result = self._execute_expert_role_action(player_idx, roles, situation)
        
        # Convert to normalized output format
        move_x = action_result.dir.x / 100.0  # Normalize movement
        move_y = action_result.dir.y / 100.0
        
        if action_result.has_pass and action_result.ball_pass:
            kick_x = action_result.ball_pass.x / 100.0
            kick_y = action_result.ball_pass.y / 100.0
            action_type = 1.0  # Kick/pass action
        else:
            kick_x = 0.0
            kick_y = 0.0
            action_type = 0.0  # Movement only
        
        confidence = self._calculate_action_confidence(situation, action_result)
        
        return np.array([move_x, move_y, kick_x, kick_y, action_type, confidence], dtype=np.float32)
    
    def _assign_simple_roles(self, situation):
        """Simplified role assignment for training"""
        ball_possession = situation['ball_possession']
        our_players = situation['our_players']
        
        roles = {
            'goalkeeper': 0,
            'ball_carrier': None,
            'support': None,
            'attackers': [],
            'defenders': []
        }
        
        if ball_possession['team'] == 'us' and ball_possession['player_idx'] is not None:
            roles['ball_carrier'] = ball_possession['player_idx']
            
            # Simple support assignment
            if roles['ball_carrier'] != 0:
                for i in range(1, 4):
                    if i != roles['ball_carrier']:
                        roles['support'] = i
                        break
        
        # Assign remaining roles
        for i in range(1, 4):
            if i not in [roles['ball_carrier'], roles['support']]:
                if random.random() < 0.5:  # Random assignment for variety
                    roles['defenders'].append(i)
                else:
                    roles['attackers'].append(i)
        
        return roles
    
    def _execute_expert_role_action(self, player_idx: int, roles, situation) -> 'PlayerAction':
        """Execute simplified expert action"""
        our_players = situation['our_players']
        player = our_players[player_idx]
        ball_pos = situation['ball_pos']
        opp_goal = situation['opponent_goal']
        our_goal = situation['our_goal']
        
        if player_idx == 0:  # Goalkeeper
            if (ball_pos - player.pos).norm() < 80:
                move_dir = (ball_pos - player.pos).normalize() * 50
                clear_target = Vec2(situation['field'].x * 0.3, situation['field'].y * 0.5)
                return PlayerAction(move_dir, clear_target - player.pos)
            else:
                goal_pos = Vec2(our_goal.x + 25, our_goal.y)
                return PlayerAction((goal_pos - player.pos) * 0.5, None)
        
        elif player_idx == roles.get('ball_carrier'):
            # Ball carrier logic
            forward_dir = (opp_goal - player.pos).normalize()
            
            # Check for shot
            if (player.pos - opp_goal).norm() < 150:
                return PlayerAction(Vec2(0, 0), opp_goal - player.pos)
            
            # Move forward or pass
            if random.random() < 0.7:  # Move forward
                return PlayerAction(forward_dir * 80, None)
            else:  # Pass
                teammates = [p for i, p in enumerate(our_players) if i != player_idx and i != 0]
                if teammates:
                    target = random.choice(teammates)
                    return PlayerAction(Vec2(0, 0), target.pos - player.pos)
        
        elif player_idx == roles.get('support'):
            # Support positioning
            if roles.get('ball_carrier') is not None:
                ball_carrier = our_players[roles['ball_carrier']]
                support_pos = ball_carrier.pos + Vec2(-40, 0)
                return PlayerAction(support_pos - player.pos, None)
        
        elif player_idx in roles.get('attackers', []):
            # Attacking movement
            attack_pos = ball_pos + (opp_goal - ball_pos).normalize() * 60
            return PlayerAction(attack_pos - player.pos, None)
        
        else:
            # Defensive movement
            defensive_pos = ball_pos + (our_goal - ball_pos).normalize() * 40
            return PlayerAction(defensive_pos - player.pos, None)
        
        # Default action
        return PlayerAction((ball_pos - player.pos).normalize() * 30, None)
    
    def _calculate_action_confidence(self, situation, action_result) -> float:
        """Calculate confidence in the action"""
        # Simple confidence based on situation
        base_confidence = 0.7
        
        if situation['ball_possession']['team'] == 'us':
            base_confidence += 0.2
        
        if situation['ball_zone'] == 'attacking':
            base_confidence += 0.1
        
        return min(1.0, base_confidence)
    
    def generate_training_data(self, num_samples: int = 15000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate training data using expert system"""
        print(f"Generating {num_samples} training samples...")
        
        X_train = []
        y_train = []
        
        for sample_idx in range(num_samples):
            if sample_idx % 1000 == 0:
                print(f"Generated {sample_idx}/{num_samples} samples")
            
            # Generate random game state
            game_state, situation = self._generate_random_game_state()
            
            # Generate training sample for each non-goalkeeper player
            for player_idx in range(4):  # Include all players
                features = self.extract_comprehensive_features(game_state, player_idx, situation)
                expert_action = self.generate_expert_action(game_state, player_idx, situation)
                
                X_train.append(features)
                y_train.append(expert_action)
        
        return np.array(X_train, dtype=np.float32), np.array(y_train, dtype=np.float32)
    
    def _generate_random_game_state(self) -> Tuple[Dict, Dict]:
        """Generate random but realistic game state"""
        field_width, field_height = 800, 600
        
        # Random ball position
        ball_x = random.uniform(50, field_width - 50)
        ball_y = random.uniform(50, field_height - 50)
        ball_pos = Vec2(ball_x, ball_y)
        
        # Generate player positions
        our_players = []
        for i in range(4):
            if i == 0:  # Goalkeeper
                pos = Vec2(random.uniform(0, 100), random.uniform(200, 400))
            else:
                pos = Vec2(random.uniform(50, field_width - 50), 
                          random.uniform(50, field_height - 50))
            
            # Create mock player object
            player = type('Player', (), {
                'pos': pos,
                'id': i
            })()
            our_players.append(player)
        
        # Generate opponent positions
        opponents = []
        for i in range(4):
            pos = Vec2(random.uniform(50, field_width - 50), 
                      random.uniform(50, field_height - 50))
            opponent = type('Player', (), {'pos': pos, 'id': i + 4})()
            opponents.append(opponent)
        
        # Determine ball possession
        our_distances = [(ball_pos - p.pos).norm() for p in our_players]
        opp_distances = [(ball_pos - p.pos).norm() for p in opponents]
        
        our_closest = min(our_distances)
        opp_closest = min(opp_distances)
        
        if our_closest < opp_closest and our_closest < 30:
            possession = {
                'team': 'us',
                'player_idx': our_distances.index(our_closest),
                'contested': opp_closest < 60
            }
        elif opp_closest < 30:
            possession = {
                'team': 'opponent',
                'player_idx': opp_distances.index(opp_closest),
                'contested': our_closest < 60
            }
        else:
            possession = {'team': 'free', 'player_idx': None, 'contested': True}
        
        # Create situation dictionary
        situation = {
            'our_players': our_players,
            'opponents': opponents,
            'ball_pos': ball_pos,
            'ball_possession': possession,
            'field': Vec2(field_width, field_height),
            'our_goal': Vec2(0, field_height / 2),
            'opponent_goal': Vec2(field_width, field_height / 2),
            'ball_zone': 'defensive' if ball_x < field_width/3 else 'middle' if ball_x < 2*field_width/3 else 'attacking'
        }
        
        return {}, situation  # Empty game_state dict, full situation
    
    def train_model(self, epochs: int = 100, batch_size: int = 64) -> tf.keras.callbacks.History:
        """Train the 4v4 strategy model"""
        if self.model is None:
            self.build_model()
        
        # Generate training data
        X_train, y_train = self.generate_training_data()
        
        print(f"Training model on {len(X_train)} samples...")
        print(f"Feature shape: {X_train.shape}")
        print(f"Target shape: {y_train.shape}")
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=8,
                    min_lr=0.0001
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    'best_soccer_model.keras',
                    monitor='val_loss',
                    save_best_only=True
                )
            ],
            verbose=1
        )
        
        self.is_trained = True
        return history
    
    def save_model(self, filepath: str = 'soccer_4v4_model'):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        # Save model
        self.model.save(f'{filepath}.keras')
        
        # Save weights
        self.model.save_weights(f'{filepath}_weights.h5')
        
        # Save configuration
        config = {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'output_size': self.output_size,
            'is_trained': self.is_trained
        }
        
        with open(f'{filepath}_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model saved: {filepath}.keras")
        print(f"Weights saved: {filepath}.weights.h5")
        print(f"Config saved: {filepath}_config.json")
    
    def load_model(self, filepath: str = 'soccer_4v4_model'):
        """Load trained model"""
        try:
            # Try loading complete model first
            if os.path.exists(f'{filepath}.keras'):
                self.model = tf.keras.models.load_model(f'{filepath}.keras')
                self.is_trained = True  # Always set when loading model
                print(f"Loaded model from {filepath}.keras")
                
                # Load config if available
                if os.path.exists(f'{filepath}_config.json'):
                    with open(f'{filepath}_config.json', 'r') as f:
                        config = json.load(f)
                    # Keep other config but ensure trained status
                    
                return True
            
            # Alternative: build model and load weights
            elif os.path.exists(f'{filepath}_config.json') and os.path.exists(f'{filepath}_weights.h5'):
                with open(f'{filepath}_config.json', 'r') as f:
                    config = json.load(f)
                
                self.input_size = config['input_size']
                self.hidden_layers = config['hidden_layers']
                self.output_size = config['output_size']
                self.is_trained = config.get('is_trained', True)
                
                self.build_model()
                self.model.load_weights(f'{filepath}_weights.h5')
                print(f"Built model and loaded weights from {filepath}")
                return True
                
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
        
        print(f"No model found at {filepath}")
        return False
    
    def predict_action(self, game_state: Dict, player_idx: int, situation: Dict) -> 'PlayerAction':
        """Predict action using trained model"""
        if self.model is None:
            raise ValueError("Model not loaded. Train or load model first.")
        
        # Extract features
        features = self.extract_comprehensive_features(game_state, player_idx, situation)
        features = features.reshape(1, -1)  # Add batch dimension
        
        # Make prediction
        prediction = self.model.predict(features, verbose=0)[0]
        
        # Convert prediction to PlayerAction
        move_x = float(prediction[0]) * 100  # Denormalize
        move_y = float(prediction[1]) * 100
        kick_x = float(prediction[2]) * 100
        kick_y = float(prediction[3]) * 100
        action_type = float(prediction[4])
        confidence = float(prediction[5])
        
        move_direction = Vec2(move_x, move_y)
        
        # Decide whether to kick based on action_type
        if action_type > 0.5 and (abs(kick_x) > 10 or abs(kick_y) > 10):
            kick_direction = Vec2(kick_x, kick_y)
        else:
            kick_direction = None
        
        return PlayerAction(move_direction, kick_direction)

# Integration with existing strategy system
def get_strategy(team: int):
    """ML-powered 4v4 strategy"""
    ml_strategy = Soccer4v4MLStrategy()
    
    # Try to load existing model
    if not ml_strategy.load_model('soccer_4v4_model'):
        print("No trained model found. Training new model...")
        ml_strategy.train_model(epochs=50)
        ml_strategy.save_model('soccer_4v4_model')
    
    if team == 0:
        print("Hello! I am team A (ML-powered 4v4 Strategy)")
    else:
        print("Hello! I am team B (ML-powered 4v4 Strategy)")
    
    def ml_formation(score: Score) -> List[Vec2]:
        config = get_config()
        field = config.field.bottom_right()
        return [
            Vec2(field.x * 0.08, field.y * 0.5),   # GK
            Vec2(field.x * 0.25, field.y * 0.25),  # Defender  
            Vec2(field.x * 0.35, field.y * 0.75),  # Midfielder
            Vec2(field.x * 0.5, field.y * 0.5),    # Forward
        ]
    
    def ml_strategy_main(game: GameState) -> List[PlayerAction]:
        # Convert GameState to our format
        situation = convert_game_state_to_situation(game, get_config())
        
        actions = []
        for i in range(NUM_PLAYERS):
            try:
                action = ml_strategy.predict_action({}, i, situation)
                actions.append(action)
            except Exception as e:
                # Fallback to simple action if prediction fails
                print(f"ML prediction failed for player {i}: {e}")
                our_players = game.team(Team.Self)
                ball_pos = game.ball.pos
                move_dir = (ball_pos - our_players[i].pos).normalize() * 50
                actions.append(PlayerAction(move_dir, None))
        
        return actions
    
    return Strategy(ml_formation, ml_strategy_main)

def convert_game_state_to_situation(game: GameState, config):
    """Convert GameState to situation dict for ML model"""
    our_players = game.team(Team.Self)
    opponents = game.team(Team.Other)
    ball_pos = game.ball.pos
    field = config.field.bottom_right()
    
    # Determine possession
    our_distances = [(ball_pos - p.pos).norm() for p in our_players]
    opp_distances = [(ball_pos - p.pos).norm() for p in opponents]
    
    our_closest = min(our_distances)
    opp_closest = min(opp_distances)
    
    if our_closest < opp_closest and our_closest < 30:
        possession = {
            'team': 'us',
            'player_idx': our_distances.index(our_closest),
            'contested': opp_closest < 60
        }
    elif opp_closest < 30:
        possession = {
            'team': 'opponent', 
            'player_idx': opp_distances.index(opp_closest),
            'contested': our_closest < 60
        }
    else:
        possession = {'team': 'free', 'player_idx': None, 'contested': True}
    
    return {
        'our_players': our_players,
        'opponents': opponents,
        'ball_pos': ball_pos,
        'ball_possession': possession,
        'field': field,
        'our_goal': config.field.goal_self(),
        'opponent_goal': config.field.goal_other(),
        'ball_zone': 'defensive' if ball_pos.x < field.x/3 else 'middle' if ball_pos.x < 2*field.x/3 else 'attacking'
    }

# Training script
if __name__ == "__main__":
    print("Training 4v4 Soccer ML Strategy...")
    
    trainer = Soccer4v4MLStrategy()
    history = trainer.train_model(epochs=100)
    trainer.save_model('soccer_4v4_model')
    
    print("Training complete! Model saved.")
    print("To use in your bot, the model will be loaded automatically.")
