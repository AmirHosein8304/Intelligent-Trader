#Amir Hossein Shakeri
#Uni_Code: 40218593
#Q-learning project

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict
import random
from collections import defaultdict

class AAPLTradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, df, initial_balance = 10000.0):
        super(AAPLTradingEnv, self).__init__()
        
        # historical data
        self.df = df
        self.prices = df['Close'].values
        self.volumes = df['Volume'].values
        self.rsi = df['RSI'].values
        
        # Trading parameters
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.trading_cost = 0.001  # 0.1% commission per trade ---> i added this for more realistic trading simulation
        self.position = 0  # -1=short, 0=neutral, 1=long
        self.entry_price = 0.0
        self.nav_history = []  
        
        # Environment tracking
        self.current_step = 0
        self.max_steps = len(df) - 1 # Total number of steps in the environment
        
        # action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0=short, 1=hold, 2=long
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -1]), 
            high=np.array([2, 2, 2, 1]),
            dtype=np.int32
        )

    def reset(self):
        self.current_step = 0
        self.current_balance = self.initial_balance
        self.position = 0
        self.entry_price = 0.0
        self.nav_history = [self.current_balance]
        return self._get_observation()
    
    def step(self,action):
        if self.current_step >= self.max_steps:
            raise Exception("Episode has ended. Call reset() to start new episode.")
        
        #we take the action and calculate the reward
        self._take_action(action)
        self.current_step += 1
        reward = self._calculate_reward()

        # Update portfolio value
        nav = self._calculate_nav()
        self.nav_history.append(nav)

        done = (self.current_step == self.max_steps) or (nav <= 0)
        return self._get_observation(), reward, done, self._get_info()
    
    def _take_action(self, action):
        current_price = self.prices[self.current_step]
    
        # Close current position if changing
        if (action == 0 and self.position == 1) or (action == 2 and self.position == -1):
            self.current_balance *= (1 - self.trading_cost)  # Pay trading cost ---> commission
            self.position = 0 # we Set to neutral after closing current position
        
        # Open new position
        if action == 0 and self.position != -1:  # Enter short
            self.position = -1
            self.entry_price = current_price
        elif action == 2 and self.position != 1:  # Enter long
            self.position = 1
            self.entry_price = current_price
        # Hold (action == 1) maintains current position no change :)
    
    def _calculate_reward(self):
        prev_price = self.prices[self.current_step - 1]
        current_price = self.prices[self.current_step]
        price_change = (current_price - prev_price) / prev_price
        # Calculate reward based on position and price change
        reward = self.position * price_change
        if self.position != 0:
            reward -= self.trading_cost # commission cost for holding a position !
            
        return reward
    
    def _get_observation(self):
        # Discretize RSI (0=oversold, 1=neutral, 2=overbought)
        current_rsi = self.rsi[self.current_step]
        rsi_bin = 0 if current_rsi < 30 else (2 if current_rsi > 70 else 1)
        
        volume_bin = self.df['Volume_Bin'].iloc[self.current_step] #volume bin (0=low, 1=medium, 2=high)
        
        price_trend = self.df['Price_Trend'].iloc[self.current_step]# price trend (1=up, -1=down, 0=neutral)
        
        return np.array([rsi_bin, volume_bin, price_trend, self.position])

    # here are some helper functions

    def _calculate_nav(self):
        #Calculate net asset value
        current_price = self.prices[self.current_step]
        if self.position == 1:  # we are in Long
            return self.current_balance * (current_price / self.entry_price)
        elif self.position == -1:  # we are in Short
            return self.current_balance * (1 - (current_price / self.entry_price - 1))
        return self.current_balance  # we are Neutral
    
    def _get_info(self):
        # Return additional info for the current step ---> it was for my own debugging
        return {
            'step': self.current_step,
            'date': self.df.index[self.current_step],
            'price': self.prices[self.current_step],
            'position': self.position,
            'nav': self._calculate_nav(),
            'rsi': self.rsi[self.current_step]
        }
    
    def render(self, mode='human'):
        # Prints out human-readable environment info !
        if mode == 'human':
            print(f"Step: {self.current_step}, "
                    f"Date: {self.df.index[self.current_step].date()}, "
                    f"Price: {self.prices[self.current_step]:.2f}, "
                    f"Position: {self.position}, "
                    f"NAV: {self._calculate_nav():.2f}")
            
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, 
        exploration_rate=1.0, exploration_decay=0.995, 
        min_exploration=0.01):
        #storing the provided hyperparameters
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration

        # Initializing our Q-table
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
    
    def _discretize_state(self, state):
        '''Since Our environment already provides discretized states (RSI/Volume bins, etc.)
        We just need to convert the numpy array to an immutable tuple for dictionary keys'''
        return tuple(state.astype(int))
    
    def choose_action(self, state):
        #Epsilon-greedy action selection
        if random.random() < self.exploration_rate:
            return self.env.action_space.sample()
        else:
            discrete_state = self._discretize_state(state)
            return np.argmax(self.q_table[discrete_state])

    def learn(self, state, action, reward, next_state, done):
        #we update the Q-value using the Bellman equation
        discrete_state = self._discretize_state(state)
        discrete_next_state = self._discretize_state(next_state)   

        # Current Q-value estimate
        current_q = self.q_table[discrete_state][action]
        
        # Maximum Q-value for next state
        max_next_q = np.max(self.q_table[discrete_next_state]) if not done else 0
        
        # Bellman equation update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[discrete_state][action] = new_q
        
        # Decay exploration rate
        if not done:
            self.exploration_rate = max(self.min_exploration,self.exploration_rate * self.exploration_decay)

    def train(self, episodes=1000, render_every=100):
        stats = {
            'episode_rewards': [],
            'episode_navs': [],
            'exploration_rate': []
        }
        
        for episode in range(1, episodes + 1):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                # Select and take action
                action = self.choose_action(state)
                next_state, reward, done, info = self.env.step(action)
                
                # Learn from experience
                self.learn(state, action, reward, next_state, done)
                
                # Update tracking
                state = next_state
                episode_reward += reward
            
            # Record stats
            stats['episode_rewards'].append(episode_reward)
            stats['episode_navs'].append(info['nav'])
            stats['exploration_rate'].append(self.exploration_rate)
            
            # Print progress
            if episode % render_every == 0:
                print(f"Episode {episode}/{episodes}, "
                        f"NAV: {info['nav']:.2f}, "
                        f"Total Reward: {episode_reward:.2f}, "
                        f"ε: {self.exploration_rate:.3f}")
        
        return stats
    

# Initialize environment and agent
df = pd.read_csv('AAPL_data.csv', index_col='Date', parse_dates=True)
df['RSI'] = df['RSI'].fillna(50)  # Fill NaN RSI values with 50 (neutral)
df['Volume_Bin'] = df['Volume_Bin'].fillna(1)  # Fill NaN Volume_Bin with 1 (medium)
df['Price_Trend'] = df['Price_Trend'].fillna(0)
df = df.dropna()  # Drop any remaining NaN values
env = AAPLTradingEnv(df)  
agent = QLearningAgent(env)

# Train the agent
stats = agent.train(episodes=1000)

# Evaluate trained policy
def evaluate(agent, env, episodes=10):
    """Evaluate agent without exploration"""
    original_epsilon = agent.exploration_rate
    agent.exploration_rate = 0  # Pure exploitation
    
    total_nav = 0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)
            state, _, done, info = env.step(action)
        total_nav += info['nav']
    
    agent.exploration_rate = original_epsilon
    return total_nav / episodes

avg_nav = evaluate(agent, env)
print(f"Average NAV after training: {avg_nav:.2f}")




