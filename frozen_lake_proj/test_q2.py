import gymnasium as gym

# initialization
env = gym.make("FrozenLake-v1", desc = None, map_name = "4x4", render_mode = "human", is_slippery=True)

observation, info = env.reset()

for _ in range(50):
	# agent policy that uses the observation and info
	action = env.action_space.sample()
	ovservation, reward, terminated, truncated, info = env.step(action)
	
	if terminated or truncated:
		observation, info = env.reset()

env.close()
