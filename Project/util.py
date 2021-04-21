from gym.spaces import MultiDiscrete, Box
import numpy as np 
def convert_action_space(env):
    # Dict(action_movement:Tuple(MultiDiscrete([11 11 11]), MultiDiscrete([11 11 11]), MultiDiscrete([11 11 11]), MultiDiscrete([11 11 11]), MultiDiscrete([11 11 11])))
    env.action_space = MultiDiscrete([11]*3*len(env.action_space['action_movement']))


def convert_obs_space(env):
    # Dict(
        #agent_qpos_qvel:Box(-inf, inf, (4, 9), float32), 
        #lidar:Box(-inf, inf, (30, 1), float32), 
        #mask_aa_obs:Box(-inf, inf, (4,), float32), 
        #observation_self:Box(-inf, inf, (9,), float32))
    n_players = 5
    env.observation_space = Box(np.NINF, np.inf, (n_players*9 + n_players*4*9 + n_players*4 + n_players*30 + n_players+ n_players*3,), np.float32)