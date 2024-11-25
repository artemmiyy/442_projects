import gymnasium as gym
import numpy as np

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True,
render_mode="human")

# Your code for Q2.2 which Executes Random Policy until 1000 episodes
class RandomPolicy:
	def __init__(self, env, actions, transitions, rewards):
		self.env = env
		self.actions = actions
		self.transitions = transitions
		self.rewards = rewards
		self.reward = None
		self.state = None
		self.next_state = None
		self.next_state_status = None
		self.state_status = None
		self.action_status = None
		self.action = None
		self.episodes = 1000
		self.epsilon = 1e-8
	
	def run_RP(self):
		for episode in range(self.episodes):
			self.state = env.reset()[0]
			finished = False
			while not finished:
				# random action
				self.action = env.action_space.sample()
				self.next_state, self.reward, finished, truncated, info = env.step(self.action_status)
				
				# get state, action, and update transitions and rewards
				self.action_status = int(self.action)
				self.state_status = int(self.state)
				self.next_state_status = int(self.next_state)
				self.transitions[self.state_status, self.action_status, self.next_state_status] += 1
				self.rewards[self.state_status, self.action_status, self.next_state_status] += self.reward

				self.state = self.next_state
		
		positive_transitions = self.transitions > 0
		print("Q2.2, Positive Transitions: ", transitions[positive_transitions])
		print("Q2.2, Positive Rewards: ", rewards[positive_transitions])

		# estimate transition and reward function
		transition_probs = np.zeros(transitions.shape)
		expected_rewards = np.zeros(rewards.shape)
		for i in range(env.observation_space.n):
			for j in range(env.action_space.n):
				total_transitions = np.sum(self.transitions[i, j, :])
				transition_probs[i, j, :] = self.transitions[i, j, :]
				transition_probs[i, j, :] /= total_transitions + self.epsilon
				total_rewards = self.actions[i, j]
				expected_rewards[i, j, :] = self.rewards[i, j, :]
				expected_rewards[i, j, :] /= total_rewards + self.epsilon

# Your code for Q2.3 which implements Value Iteration



#Your code for Q2.4 which implements Policy Extraction



# Your code for Q2.5 which executes the optimal policy


if __name__ == "__main__":
	# initialize the environment
	env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode="human")
	# matrices for transitions and rewards
	actions = np.zeros((env.observation_space.n, env.action_space.n))
	transitions = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
	rewards = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
	
	random_policy = RandomPolicy(actions, transitions, rewards)
