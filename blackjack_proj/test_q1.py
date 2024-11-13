import gymnasium as gym

# initializing environments
env = gym.make("Blackjack-v1", render_mode="human")
observation, info = env.reset()

for _ in range(50):
	# environment policy that uses the observation and info
	action = env.action_space.sample()
	observation, reward, terminated, truncated, info = env.step(action)
	if terminated or truncated:
		observation, info = env.reset()

env.close()
