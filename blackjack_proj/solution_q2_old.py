import gymnasium as gym
import numpy as np

# Your code for Q2.2 which Executes Random Policy until 1000 episodes
class RandomPolicy:
	def __init__(self, env, actions, transitions, rewards):
		self.env = env
		self.actions = actions
		self.transitions = transitions
		self.rewards = rewards
		self.next_state = None
		self.next_state_status = None
		self.state_status = None
		self.action_status = None
		self.action = None
		self.episodes = 1000
		self.epsilon = 1e-8
	
	def run_RP(self):
		for episode in range(self.episodes):
			state = self.env.reset()[0]
			finished = False
			while not finished:
				# random action
				self.action = self.env.action_space.sample()
				self.next_state, reward, finished, truncated, info = self.env.step(self.action)
				
				# get state, action, and update transitions and rewards
				self.action_status = int(self.action)
				self.state_status = int(state)
				self.next_state_status = int(self.next_state)
				self.transitions[self.state_status, self.action_status, self.next_state_status] += 1
				self.rewards[self.state_status, self.action_status, self.next_state_status] += reward

				state = self.next_state
		
		positive_transitions = 0 < self.transitions
		print("Q2.2, Positive Transitions: ", self.transitions[positive_transitions])
		print("Q2.2, Positive Rewards: ", self.rewards[positive_transitions])

		# estimate transition and reward functions
		transition_probs = np.zeros(self.transitions.shape)
		expected_rewards = np.zeros(self.rewards.shape)
		for obs in range(self.env.observation_space.n):
			for act in range(self.env.action_space.n):
				total_transitions = np.sum(self.transitions[obs, act, :])
				transition_probs[obs, act, :] = self.transitions[obs, act, :]
				transition_probs[obs, act, :] /= total_transitions + self.epsilon
				total_rewards = self.actions[obs, act]
				expected_rewards[obs, act, :] = self.rewards[obs, act, :]
				expected_rewards[obs, act, :] /= total_rewards + self.epsilon

		return self.env, transition_probs, expected_rewards
		
# Your code for Q2.3 which implements Value Iteration
class ValueIteration:
	def __init__(self, env, transition_probs, expected_rewards):
		self.env = env
		self.policy = np.zeros(self.env.observation_space.n, dtype = int)
		self.space = np.zeros(self.env.observation_space.n)
		# parameters
		self.convergence = 1e-3
		self.gamma = 0.97
		self.total_iterations = 1000
		self.transition_probs = transition_probs
		self.expected_rewards = expected_rewards

	def run_VI(self):
		for iter in range(self.total_iterations):
			curr_convergence = 0
			for sp in range(self.env.observation_space.n):
				curr_space = self.space[sp]
				# rewrite  this
				max_action_value = float('-inf')
				for action in range(self.env.action_space.n):
					action_value = 0
					for s_prime in range(self.env.observation_space.n):
						action_value += (
							self.transition_probs[sp, action, s_prime] *
							(self.expected_rewards[sp, action, s_prime] +
							self.gamma * self.space[s_prime])
						)
					max_action_value = max(max_action_value, action_value)
				self.space[sp] = max_action_value
				new_convergence = abs(curr_space - self.space[sp])
				curr_convergence = max(curr_convergence, new_convergence)

			print(f"Iteration #{iter}: Delta = {curr_convergence}")
			if self.convergence > curr_convergence: break
		
		print(f"Value Iteration took {iter + 1} iterations.")

#Your code for Q2.4 which implements Policy Extraction
	def run_policy_extraction(self):
		# rewrite this
		print("Policy Extraction Staged")
		for s in range(self.env.observation_space.n):
			self.policy[s] = np.argmax([sum([self.transition_probs[s, a, s_prime] * \
				(self.expected_rewards[s, a, s_prime] + self.gamma * self.space[s_prime]) \
					for s_prime in range(self.env.observation_space.n)]) \
						for a in range(self.env.action_space.n)])
		print("Policy Extraction Completed.")
		return self.policy

# Your code for Q2.5 which executes the optimal policy
	def execute_policy(self, policy, episodes, render_mode):
		rewards = list()
		for ep in range(episodes):
			self.env.close()
			self.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4",
						   is_slippery=True, render_mode="human")
			observation = self.env.reset()

			if isinstance(observation, int): state = observation
			else: state = observation[0]

			curr_reward = 0
			finished = False
			
			while not finished:
				action = int(policy[state])
				observation, reward, finished, truncated, info = self.env.step(action)
				if isinstance(observation, int): state = observation
				else: state = observation[0]
				curr_reward += reward
				if render_mode == 'human': self.env.render()
				if finished or truncated: break
			
			rewards.append(curr_reward)
		
		self.env.close()
		return rewards


if __name__ == "__main__":
	# initialize the environment
	env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True, render_mode='rgb_array')
	# matrices for transitions and rewards
	actions = np.zeros((env.observation_space.n, env.action_space.n))
	transitions = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
	rewards = np.zeros((env.observation_space.n, env.action_space.n, env.observation_space.n))
	
	# Q2.2
	random_policy = RandomPolicy(env, actions, transitions, rewards)
	env, transition_probs, expected_rewards = random_policy.run_RP()

	# Q2.3
	value_iteration = ValueIteration(env, transition_probs, expected_rewards)
	# commented out because execute_policy runs it
	value_iteration.run_VI()

	# Q2.4
	# commented out because execute_policy runs it
	policy = value_iteration.run_policy_extraction()

	# 2.5
	value_iteration.execute_policy(policy, 1000, 'human')
