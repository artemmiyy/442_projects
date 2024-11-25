import gymnasium as gym
import numpy as np
from collections import defaultdict

# initializing the environment

# blackjack player class
class BlackJackBeast:
	def __init__(self, discount, rate, exp_val, exp_terminal, exp_discount):
		self.discount = discount
		self.rate = rate
		self.exp_val = exp_val
		self.exp_terminal = exp_terminal
		self.exp_discount = exp_discount
		# initialize and populate the table with zeros
		self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
	
	# private method to get temporal difference error
	def __get_temp_error(self, reward, next_value, action, observed_state):
		temp_error = reward + self.discount * next_value
		temp_error -= self.q_table[observed_state][action]
		return temp_error
	
	# private method to update q table
	def __update_q_table(self, observed_state, action, temp_error):
		self.q_table[observed_state][action] = self.rate * temp_error
		self.q_table[observed_state][action] += self.q_table[observed_state][action]

	def update_qval(self, observed_state, action, reward, terminated, next_observation):
		if terminated: next_value = 0
		else: next_value = np.max(self.q_table[next_observation])

		# temporal difference error
		temp_error = self.__get_temp_error(reward, next_value, action, observed_state)

		# update the table
		self.__update_q_table(observed_state, action, temp_error)
	
	# slows down exploration
	def slow_down_exp(self):
		new_exp_val = self.exp_val - exp_discount
		self.exp_val = max(self.exp_terminal, new_exp_val)

	# player decision
	def decision(self, observed_state):
		if self.exp_val > np.random.random(): return env.action_space.sample()
		return int(np.argmax(self.q_table[observed_state]))

if __name__ == "__main__":
	# initialize the environment
	env = gym.make('Blackjack-v1', natural = False, sab = False, render_mode = "human")

	# initializing the parameters
	episodes = 1000000
	discount = 0.9
	rate = 0.001

	# initializing the exploration values
	exp_val = 1
	exp_terminal = 0.1
	exp_discount = exp_val / (episodes / 3)

	# initialize the agent
	BlackJackFein = BlackJackBeast(discount, rate, exp_val, exp_terminal, exp_discount)
	env = gym.wrappers.RecordEpisodeStatistics(env, episodes)
	wins, losses = 0, 0

	for e in range(episodes):
		game_ended = False
		observation, info = env.reset()

		while not game_ended:
			decision = BlackJackFein.decision(observation)
			new_observation, reward, terminated, truncated, info = env.step(decision)
			BlackJackFein.update_qval(observation, decision, reward, terminated, new_observation)
			observation = new_observation
			game_ended = truncated or terminated

		if reward == 1:
			print("Bot won")
			wins += 1
		
		BlackJackFein.slow_down_exp()
	
	print("The win rate is: " + str(wins / episodes))

### Q-Learning End
