import gym
from mujoco_worldgen.util.envs.flexible_load import load_env
import numpy as np
env, args_remaining = load_env(
    "/home/weustis/Group-9/multi-agent-emergence-environments/examples/tag.jsonnet", # change this 
    core_dir='/home/weustis/Group-9/multi-agent-emergence-environments',            # change this
    envs_dir= 'mae_envs/envs', # don't change this
    xmls_dir= 'xmls', # don't change this
    return_args_remaining=True # don't change this
    )

env.reset()

print(env.action_space)
for _ in range(1000): # run for 15 substeps * n steps (15000 steps)
    env.render()
    action = random_action(env.action_space)
    env.step(env.action_space.sample()) # take a random action
env.close()

