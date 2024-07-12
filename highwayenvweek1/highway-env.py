# Write a program to compare the human performance with random agent in Highway Env

# Metrics:

# Collisions per 1000 meters traveled

# Average speed

# Average collision-free speed

# Action distribution (histogram of actions)

import gymnasium as gym
from matplotlib import pyplot as plt
import random


env = gym.make('highway-v0', render_mode='rgb_array')
env.configure({
    "lanes_count": 6,
    "duration": 30,
})

env.seed(42)


def random_action():
    return random.choice([0, 1, 2, 3, 4])


env.reset()
for _ in range(100):
    action = random_action()
    obs, reward, done, truncated, info = env.step(action)
    env.render()

plt.imshow(env.render())
plt.show()