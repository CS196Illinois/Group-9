import gym
from mujoco_worldgen.util.envs.flexible_load import load_env
import numpy as np
import copy 
from util import convert_action_space, convert_obs_space
# make sure to pip install stable_baselines
from stable_baselines import PPO2
from stable_baselines.common.evaluation import evaluate_policy

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
model.learn(total_timesteps=int(2e5))
# Save the agent
model.save("PPO2_tag")
del model  # delete trained model to demonstrate loading

# Load the trained agent
model = PPO2.load("PPO2_tag")

# Evaluate the agent
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()

