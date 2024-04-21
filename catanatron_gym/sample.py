import random
import gymnasium as gym

from catanatron import Color
from catanatron.players.minimax import AlphaBetaPlayer

def my_reward_function(game, p0_color):
    winning_color = game.winning_color()
    if p0_color == winning_color:
        return 100
    elif winning_color is None:
        return 0
    else:
        return -100

# 2-player catan until 6 points.
env = gym.make(
    "catanatron_gym:catanatron-v1",
    config={
        "map_type": "BASE",
        "vps_to_win": 6,
        "enemies": [AlphaBetaPlayer(Color.RED)],
        "reward_function": my_reward_function,
        "representation": "mixed",
    },
)
observation, info = env.reset()
for _ in range(1000):
    # your agent here (this takes random actions)
    action = random.choice(env.unwrapped.get_valid_actions())
    observation, reward, terminated, truncated, info = env.step(action)
    print("observation")
    print(observation['board'].shape)
    print(type(observation))
    print("valid actions")
    print(env.unwrapped.get_valid_actions())
    print("reward")
    print(reward)
    print("info")
    print(info)
    break
    done = terminated or truncated
    if done:
        observation, info = env.reset()
env.close()



