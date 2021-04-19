import gym
from mujoco_worldgen.util.envs.flexible_load import load_env
import numpy as np
import copy 
from util import convert_action_space, convert_obs_space
# make sure to pip install stable_baselines
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common import make_vec_env

# https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

env, args_remaining = load_env(
    "/home/weustis/Group-9/multi-agent-emergence-environments/examples/tag.jsonnet",    # change this 
    core_dir='/home/weustis/Group-9/multi-agent-emergence-environments',                # change this
    envs_dir= 'mae_envs/envs',                                                          # don't change this
    xmls_dir= 'xmls',                                                                   # don't change this
    return_args_remaining=True                                                          # don't change this
    )

env_alt_space = copy.deepcopy(env)
convert_action_space(env_alt_space)
convert_obs_space(env_alt_space)

model = PPO2('MlpPolicy', env_alt_space, verbose=1)
# Train the agent
model.learn(total_timesteps=int(2e6))
# Save the agent
model.save("model_tag")
