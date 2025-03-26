import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
print(SIMPLE_MOVEMENT)
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0', render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

obs, _ = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    if done:
        obs, info = env.reset()
    env.render()
env.close()