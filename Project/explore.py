from stable_baselines import PPO2
from util import convert_action_space, convert_obs_space
from mujoco_worldgen.util.envs.flexible_load import load_env
import copy 
import numpy as np 

def multi_agent_process_action(arr):
    
    comb_action = np.zeros((len(arr)*len(arr[0])))
    i = 0
    for agent_action in arr:
        for action in agent_action:
            comb_action[i] = int(action)
            i+=1
    return [list(comb_action)]

def mutli_agent_process_observations(obs, n_players, i):
    #print(obs.shape)
    #input(f"Obs! {n_players}, {i}")
    obs = obs[0]
    
    o1 = n_players*9*4
    o2 = n_players*30
    o3 = n_players*4
    o4 = n_players*9
    agent_qpos_qvel = obs[:o1]
    lidar = obs[o1:o1+o2]
    mask_aa_obs = obs[o1+o2:o1+o2+o3]
    observation_self = obs[o1+o2+o3:o1+o2+o3+o4]

    agent_i_qpos_qvel = agent_qpos_qvel[9*4*i:9*4*(i+1)]
    lidar_i = lidar[30*i:30*(i+1)]
    mask_aa_obs_i = mask_aa_obs[4*i:4*(i+1)]
    observation_self_i = observation_self[9*i:9*(i+1)]

    ret_arr = np.concatenate([agent_i_qpos_qvel,lidar_i, mask_aa_obs_i, observation_self_i])
    # ret_arr = np.pad(ret_arr, (0, 395-len(ret_arr)), 'constant')
    ret_arr = np.array([ret_arr])
    #print(ret_arr.shape) 
    #input("Waiting!")
    return ret_arr

model = PPO2.load("model_tag")

env, args_remaining = load_env(
    "/home/weustis/Group-9/multi-agent-emergence-environments/examples/tag.jsonnet",    # change this 
    core_dir='/home/weustis/Group-9/multi-agent-emergence-environments',                # change this
    envs_dir= 'mae_envs/envs',                                                          # don't change this
    xmls_dir= 'xmls',                                                                   # don't change this
    return_args_remaining=True                                                          # don't change this
    )

env_alt_space = copy.deepcopy(env)
convert_action_space(env)
convert_obs_space(env)


# Evaluate the agent
# mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
n_players = 5 
# Enjoy trained agent
obs = env.reset()
last_rew = 0 
while (True):
    agent_actions = []
    for n in range(n_players):
        obs_agent_i = mutli_agent_process_observations([obs], n_players, n)
        action_agent_i, _states = model.predict(obs_agent_i)
        agent_actions.append(action_agent_i[0])
       #  print(agent_actions)
    action = multi_agent_process_action(agent_actions)
    obs, rewards, dones, info = env.step(action[0])

    if rewards != last_rew:
        print(f"Player {np.argmin(last_rew)} tagged Player {np.argmin(rewards)}!")
    last_rew = rewards
    
    env.render()

