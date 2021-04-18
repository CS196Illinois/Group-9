import gym
from mujoco_worldgen.util.envs.flexible_load import load_env
import numpy as np
env, args_remaining = load_env(
    "/home/weustis/Group-9/multi-agent-emergence-environments/examples/tag.jsonnet",
    core_dir='/home/weustis/Group-9/multi-agent-emergence-environments',
    envs_dir= 'mae_envs/envs',
    xmls_dir= 'xmls',
    return_args_remaining=True
    )

env.reset()

def random_action(action_space):
    if isinstance(action_space, gym.spaces.Box):
        return np.zeros(action_space.shape[0])
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return action_space.nvec // 2  

print(env.action_space)
for _ in range(1000):
    env.render()
    action = random_action(env.action_space)
    print(env.action_space.sample())
    env.step(env.action_space.sample()) # take a random action
env.close()

