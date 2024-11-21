import gymnasium as gym
import numpy as np
from collections import defaultdict

# initializing the environment
env = gym.make('Blackjack-v1', natural = False, sab = False, render_mode = "human")

### Q-Learning Code
# initializing the parameters
episodes = 1000000
discount = 0.9
rate = 0.001

# initializing the exploration values
exp_val = 1
exp_terminal = 0.1
exp_discount = exp_val / (episodes / 3)

# blackjack player class
class BlackJackBeast:
	def __init__(self, discount, rate,  exp_val, exp_terminal, exp_discount):
		self.discount = discount
		self.rate = rate
		self.exp_val = exp_val
		self.exp_terminal = exp_terminal
		self.exp_discount = exp_discount
		# initialize and populate the table with zeros
		self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
	
	# function to get temporal difference error
	def get_temp_error(self, reward, next_value, action, observed_state):
		temp_error = reward + self.discount * next_value
		temp_error -= self.q_table[observed_state][action]
		return temp_error
	
	def update_table(self, observed_state, action, temp_error):
		self.q_table[observed_state][action] = self.rate * temp_error
		self.q_table[observed_state][action] += self.q_table[observed_state][action]

	def update_qval(self, observed_state, action, reward, terminated, next_observation):
		if terminated: next_value = 0
		else: next_value = np.max(self.q_table[next_observation])

		# temporal difference error
		temp_error = self.get_temp_error(reward, next_value, action, observed_state)

		# update the table
		self.update_table(observed_state, action, temp_error)
	
	# slows down exploration
	def slow_down_exp(self):
		new_exp_val = self.exp_value - exp_discount
		self.exp_val = max(self.exp_terminal, new_exp_val)

	# player decision
	def decision(self, observed_state):
		if self.exp_val > np.random.random(): return env.action_space.sample()
		return int(np.argmax(self.q_table[observed_state]))

BlackJackFein = BlackJackBeast()
### Q-Learning End
