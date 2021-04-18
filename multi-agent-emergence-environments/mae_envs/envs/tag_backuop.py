import gym
import numpy as np
from copy import deepcopy
from mae_envs.envs.base import Base
from mae_envs.wrappers.multi_agent import (SplitMultiAgentActions,
                                           SplitObservations, SelectKeysWrapper)
from mae_envs.wrappers.util import (DiscretizeActionWrapper, ConcatenateObsWrapper,
                                    MaskActionWrapper, SpoofEntityWrapper,
                                    DiscardMujocoExceptionEpisodes,
                                    AddConstantObservationsWrapper)
from mae_envs.wrappers.manipulation import (GrabObjWrapper, GrabClosestWrapper,
                                            LockObjWrapper, LockAllWrapper)
from mae_envs.wrappers.lidar import Lidar
from mae_envs.wrappers.line_of_sight import (AgentAgentObsMask2D, AgentGeomObsMask2D,
                                             AgentSiteObsMask2D)
from mae_envs.wrappers.prep_phase import (PreparationPhase, NoActionsInPrepPhase,
                                          MaskPrepPhaseAction)
from mae_envs.wrappers.limit_mvmnt import RestrictAgentsRect
from mae_envs.wrappers.team import TeamMembership
from mae_envs.wrappers.food import FoodHealthWrapper, AlwaysEatWrapper
from mae_envs.modules.agents import Agents, AgentManipulation
from mae_envs.modules.walls import RandomWalls, WallScenarios
from mae_envs.modules.objects import Boxes, Ramps, LidarSites
from mae_envs.modules.food import Food
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.util import (uniform_placement, close_to_other_object_placement,
                                   uniform_placement_middle)

def convert_action_space(env):
    # Dict(action_movement:Tuple(MultiDiscrete([11 11 11]), MultiDiscrete([11 11 11]), MultiDiscrete([11 11 11]), MultiDiscrete([11 11 11]), MultiDiscrete([11 11 11])))
    env.action_space = MultiDiscrete([11]*len(env.action_space['action_movement']))


def convert_obs_space(env):
    # Dict(
        #agent_qpos_qvel:Box(-inf, inf, (4, 9), float32), 
        #lidar:Box(-inf, inf, (30, 1), float32), 
        #mask_aa_obs:Box(-inf, inf, (4,), float32), 
        #observation_self:Box(-inf, inf, (9,), float32))

    print(env.observation_space)
    env.observation_space = Box(np.NINF, np.inf, (4*9 + 30 + 4 + 9, 1), np.float32)


class StableBaselineInputWrapper(gym.Wrapper):
    def __init__(self):
        super().__init__(env)
        self.sample_action = env.action_space.sample()
        self.sample_obs = env.observation_space.sample()

    def step(self, action):
        # convert input action to dict format
        # OrderedDict([('action_movement', (array([8, 0, 6]), array([10,  8,  3]), array([3, 7, 6]), array([6, 2, 3]), array([0, 9, 6])))])
        print(action)
        input("Correct?")
        obs, rew, done, info = self.env.step(action)
        return 

class StableBaselineOutputWrapper(gym.Wrapper):
    def __init__(self):
        super().__init__(env)

    def step(self):
        # convert input action, space from one format to other    
        return

class TrackStatWrapper(gym.Wrapper):
    '''
        Keeps track of important statistics that are indicative of hide and seek
        dynamics
    '''
    def __init__(self, env, n_boxes, n_ramps, n_food):
        super().__init__(env)
        self.n_boxes = n_boxes
        self.n_ramps = n_ramps
        self.n_food = n_food

    def reset(self):
        obs = self.env.reset()
        if self.n_boxes > 0:
            self.box_pos_start = obs['box_pos']
        if self.n_ramps > 0:
            self.ramp_pos_start = obs['ramp_pos']
        if self.n_food > 0:
            self.total_food_eaten = np.sum(obs['food_eat'])

        self.in_prep_phase = False

        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self.n_food > 0:
            self.total_food_eaten += np.sum(obs['food_eat'])

        if self.in_prep_phase and obs['prep_obs'][0, 0] == 1.0:
            # Track statistics at end of preparation phase
            self.in_prep_phase = False

            if self.n_boxes > 0:
                self.max_box_move_prep = np.max(np.linalg.norm(obs['box_pos'] - self.box_pos_start, axis=-1))
                self.num_box_lock_prep = np.sum(obs['obj_lock'])
            if self.n_ramps > 0:
                self.max_ramp_move_prep = np.max(np.linalg.norm(obs['ramp_pos'] - self.ramp_pos_start, axis=-1))
                if 'ramp_obj_lock' in obs:
                    self.num_ramp_lock_prep = np.sum(obs['ramp_obj_lock'])
            if self.n_food > 0:
                self.total_food_eaten_prep = self.total_food_eaten

        if done:
            # Track statistics at end of episode
            if self.n_boxes > 0:
                self.max_box_move = np.max(np.linalg.norm(obs['box_pos'] - self.box_pos_start, axis=-1))
                self.num_box_lock = np.sum(obs['obj_lock'])
                info.update({
                    'max_box_move_prep': self.max_box_move_prep,
                    'max_box_move': self.max_box_move,
                    'num_box_lock_prep': self.num_box_lock_prep,
                    'num_box_lock': self.num_box_lock})

            if self.n_ramps > 0:
                self.max_ramp_move = np.max(np.linalg.norm(obs['ramp_pos'] - self.ramp_pos_start, axis=-1))
                info.update({
                    'max_ramp_move_prep': self.max_ramp_move_prep,
                    'max_ramp_move': self.max_ramp_move})
                if 'ramp_obj_lock' in obs:
                    self.num_ramp_lock = np.sum(obs['ramp_obj_lock'])
                    info.update({
                        'num_ramp_lock_prep': self.num_ramp_lock_prep,
                        'num_ramp_lock': self.num_ramp_lock})

            if self.n_food > 0:
                info.update({
                    'food_eaten': self.total_food_eaten,
                    'food_eaten_prep': self.total_food_eaten_prep})

        return obs, rew, done, info


class TagPlayerWrapper(gym.Wrapper):
    '''
        Allows agents to tag the closest agent.
    '''
    def __init__(self, env, n_it, n_agents):
        super().__init__(env)
        self.it_status = [1]*n_it + [0]*(n_agents-n_it) 
        self.tag_timer = 0
        # self.last_it_stauts = self.it_status
        print("Initializing TagPlayerWrapper") 
        
    def observation(self, obs):
        # add relavent info to obs
        obs['it']=np.expand_dims(self.it_status, -1) # whether or not the player is it
        obs['it_loc'] = np.array([obs['agent_pos'][agent_idx] for agent_idx in range(len(obs['agent_pos'])) if self.it_status[agent_idx] for i in range(len(self.it_status))])

        return obs

    def step(self, action): 
        obs, rew, done, info = self.env.step(action)
        dist = lambda x,y : np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2 + (x[2]-y[2])**2)
        for it_pos in [obs['agent_pos'][i] for i in range(len(obs['agent_pos'])) if self.it_status[i]]: # for each it's position
            for position_idx in range(len(obs['agent_pos'])): 
                position = obs['agent_pos'][position_idx]
                distance = dist(it_pos, position)
                if distance <= .5 and distance>0 and self.tag_timer  == 0:
                    self.it_status = [0]*len(self.it_status)
                    self.it_status[position_idx] = 1
                    self.tag_timer = 15*5
        
        self.tag_timer -= 1 # don't let them tag until 65 timesteps have passed
        self.tag_timer = max(self.tag_timer, 0)
        #print(obs, rew, self.it_status)
        rew = [-1*x for x in self.it_status]
        print(rew)
    
        return self.observation(obs), rew, done, info


class MaskUnseenAction(gym.Wrapper):
    '''
        Masks a (binary) action with some probability if agent or any of its teammates was being observed
        by opponents at any of the last n_latency time step

        Args:
            team_idx (int): Team index (e.g. 0 = hiders) of team whose actions are
                            masked
            action_key (string): key of action to be masked
    '''

    def __init__(self, env, team_idx, action_key):
        super().__init__(env)
        self.team_idx = team_idx
        self.action_key = action_key
        self.n_agents = self.unwrapped.n_agents
        self.n_hiders = self.metadata['n_hiders']

    def reset(self):
        self.prev_obs = self.env.reset()
        self.this_team = self.metadata['team_index'] == self.team_idx

        return deepcopy(self.prev_obs)

    def step(self, action):
        is_caught = np.any(self.prev_obs['mask_aa_obs'][self.n_hiders:, :self.n_hiders])
        if is_caught:
            action[self.action_key][self.this_team] = 0

        self.prev_obs, rew, done, info = self.env.step(action)
        return deepcopy(self.prev_obs), rew, done, info


def quadrant_placement(grid, obj_size, metadata, random_state):
    '''
        Places object within the bottom right quadrant of the playing field
    '''
    grid_size = len(grid)
    qsize = metadata['quadrant_size']
    pos = np.array([random_state.randint(grid_size - qsize, grid_size - obj_size[0] - 1),
                    random_state.randint(1, qsize - obj_size[1] - 1)])
    return pos


def outside_quadrant_placement(grid, obj_size, metadata, random_state):
    '''
        Places object outside of the bottom right quadrant of the playing field
    '''
    grid_size = len(grid)
    qsize = metadata['quadrant_size']
    poses = [
        np.array([random_state.randint(1, grid_size - qsize - obj_size[0] - 1),
                  random_state.randint(1, qsize - obj_size[1] - 1)]),
        np.array([random_state.randint(1, grid_size - qsize - obj_size[0] - 1),
                  random_state.randint(qsize, grid_size - obj_size[1] - 1)]),
        np.array([random_state.randint(grid_size - qsize, grid_size - obj_size[0] - 1),
                  random_state.randint(qsize, grid_size - obj_size[1] - 1)]),
    ]
    return poses[random_state.randint(0, 3)]


def make_env(n_substeps=15, horizon=80, deterministic_mode=False,
             floor_size=6.0, grid_size=20, door_size=1,
             n_hiders=1, n_seekers=1, max_n_agents=None,
             n_boxes=2, n_ramps=1, n_elongated_boxes=0,
             rand_num_elongated_boxes=False, n_min_boxes=None,
             box_size=0.5, boxid_obs=False, box_only_z_rot=True,
             rew_type='joint_zero_sum',
             lock_box=False, grab_box=False, lock_ramp=False,
             lock_type='any_lock_specific',
             lock_grab_radius=0.25, lock_out_of_vision=True, grab_exclusive=False,
             grab_out_of_vision=False, grab_selective=False,
             box_floor_friction=0.2, other_friction=0.01, gravity=[0, 0, -50],
             action_lims=(-0.9, 0.9), polar_obs=True,
             scenario='quadrant', quadrant_game_hider_uniform_placement=False,
             p_door_dropout=0.0,
             n_rooms=1, random_room_number=False, prob_outside_walls=1.0,
             n_lidar_per_agent=0, visualize_lidar=False, compress_lidar_scale=None,
             hiders_together_radius=None, seekers_together_radius=None,
             prep_fraction=0.4, prep_obs=False,
             team_size_obs=False,
             restrict_rect=None, penalize_objects_out=False,
             n_food=0, food_radius=None, food_respawn_time=None, max_food_health=1,
             food_together_radius=None, food_rew_type='selfish', eat_when_caught=False,
             food_reward_scale=1.0, food_normal_centered=False, food_box_centered=False,
             n_food_cluster=1, n_it=1, n_players=5):
    n_agents = n_players
    grab_radius_multiplier = lock_grab_radius / box_size
    lock_radius_multiplier = lock_grab_radius / box_size

    env = Base(n_agents=n_players, n_substeps=n_substeps, horizon=horizon,
               floor_size=floor_size, grid_size=grid_size,
               action_lims=action_lims,
               deterministic_mode=deterministic_mode)

    env.add_module(RandomWalls(
        grid_size=grid_size, num_rooms=7,
        random_room_number=random_room_number, min_room_size=2,
        door_size=door_size,
        prob_outside_walls=prob_outside_walls, gen_door_obs=False))
    
    box_placement_fn = uniform_placement
    ramp_placement_fn = uniform_placement
    cell_size = floor_size / grid_size

    first_hider_placement = uniform_placement

    agent_placement_fn = [first_hider_placement] * (n_players-n_it)

    first_seeker_placement = uniform_placement

    
    agent_placement_fn += [first_seeker_placement] * (n_it)



    env.add_module(Agents(n_agents,
                          placement_fn=agent_placement_fn,
                          friction=other_friction,
                          polar_obs=polar_obs))
    if np.max(n_boxes) > 0:
        env.add_module(Boxes(n_boxes=n_boxes, placement_fn=box_placement_fn,
                             friction=box_floor_friction, polar_obs=polar_obs,
                             n_elongated_boxes=n_elongated_boxes,
                             boxid_obs=boxid_obs, box_only_z_rot=box_only_z_rot))
    if n_ramps > 0:
        env.add_module(Ramps(n_ramps=n_ramps, placement_fn=ramp_placement_fn, friction=other_friction, polar_obs=polar_obs,
                             pad_ramp_size=(np.max(n_elongated_boxes) > 0)))
    if n_lidar_per_agent > 0 and visualize_lidar:
        env.add_module(LidarSites(n_agents=n_agents, n_lidar_per_agent=n_lidar_per_agent))
    if n_food > 0:
        if scenario == 'quadrant':
            first_food_placement = quadrant_placement
        elif food_box_centered:
            first_food_placement = uniform_placement_middle(0.25)
        else:
            first_food_placement = uniform_placement
        if food_together_radius is not None:
            cell_size = floor_size / grid_size
            ftr_in_cells = np.ceil(food_together_radius / cell_size).astype(int)

            env.metadata['food_together_radius'] = ftr_in_cells

            assert n_food % n_food_cluster == 0
            cluster_assignments = np.repeat(np.arange(0, n_food, n_food // n_food_cluster), n_food // n_food_cluster)
            food_placement = [close_to_other_object_placement(
                "food", i, "food_together_radius") for i in cluster_assignments]
            food_placement[::n_food // n_food_cluster] = [first_food_placement] * n_food_cluster
        else:
            food_placement = first_food_placement
        env.add_module(Food(n_food, placement_fn=food_placement))

    env.add_module(AgentManipulation())
    if box_floor_friction is not None:
        env.add_module(FloorAttributes(friction=box_floor_friction))
    env.add_module(WorldConstants(gravity=gravity))
    env.reset()
    keys_self = ['agent_qpos_qvel', 'hider']
    keys_mask_self = ['mask_aa_obs']
    keys_external = ['agent_qpos_qvel']
    keys_copy = ['you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock']
    keys_mask_external = []
    env = SplitMultiAgentActions(env)
    if team_size_obs:
        keys_self += ['team_size']
    team_index = list(range(n_agents))
    env = TeamMembership(env, team_index, n_teams = len(np.unique(team_index)))
    env = AgentAgentObsMask2D(env)
    hider_obs = np.array([[1]] * n_hiders + [[0]] * n_seekers)
    env = AddConstantObservationsWrapper(env, new_obs={'hider': hider_obs})
    env = TagPlayerWrapper(env, n_it, n_players)
    if restrict_rect is not None:
        env = RestrictAgentsRect(env, restrict_rect=restrict_rect, penalize_objects_out=penalize_objects_out)
   #  env = PreparationPhase(env, prep_fraction=prep_fraction)
    env = DiscretizeActionWrapper(env, 'action_movement')
    if np.max(n_boxes) > 0:
        env = AgentGeomObsMask2D(env, pos_obs_key='box_pos', mask_obs_key='mask_ab_obs',
                                 geom_idxs_obs_key='box_geom_idxs')
        keys_external += ['mask_ab_obs', 'box_obs']
        keys_mask_external.append('mask_ab_obs')
    if n_food:
        env = AgentSiteObsMask2D(env, pos_obs_key='food_pos', mask_obs_key='mask_af_obs')
        env = FoodHealthWrapper(env, respawn_time=(np.inf if food_respawn_time is None else food_respawn_time),
                                eat_thresh=(np.inf if food_radius is None else food_radius),
                                max_food_health=max_food_health, food_rew_type=food_rew_type,
                                reward_scale=food_reward_scale)
        env = MaskActionWrapper(env, 'action_eat_food', ['mask_af_obs'])  # Can only eat if in vision
        if prep_obs:
            env = MaskPrepPhaseAction(env, 'action_eat_food')
        if not eat_when_caught:
            env = MaskUnseenAction(env, 0, 'action_eat_food')
        eat_agents = np.arange(n_hiders)
        env = AlwaysEatWrapper(env, agent_idx_allowed=eat_agents)
        keys_external += ['mask_af_obs', 'food_obs']
        keys_mask_external.append('mask_af_obs')

    if lock_box and np.max(n_boxes) > 0:
        env = LockObjWrapper(env, body_names=[f'moveable_box{i}' for i in range(np.max(n_boxes))],
                             agent_idx_allowed_to_lock=np.arange(n_hiders+n_seekers),
                             lock_type=lock_type, radius_multiplier=lock_radius_multiplier,
                             obj_in_game_metadata_keys=["curr_n_boxes"],
                             agent_allowed_to_lock_keys=None if lock_out_of_vision else ["mask_ab_obs"])
    if n_ramps > 0:
        env = AgentGeomObsMask2D(env, pos_obs_key='ramp_pos', mask_obs_key='mask_ar_obs',
                                 geom_idxs_obs_key='ramp_geom_idxs')
        if lock_ramp:
            env = LockObjWrapper(env, body_names=[f'ramp{i}:ramp' for i in range(n_ramps)],
                                 agent_idx_allowed_to_lock=np.arange(n_hiders+n_seekers),
                                 lock_type=lock_type, ac_obs_prefix='ramp_',
                                 radius_multiplier=lock_radius_multiplier,
                                 obj_in_game_metadata_keys=['curr_n_ramps'],
                                 agent_allowed_to_lock_keys=None if lock_out_of_vision else ["mask_ar_obs"])
        keys_external += ['ramp_obs']
        keys_mask_external.append('mask_ar_obs')
    if grab_box and (np.max(n_boxes) > 0 or n_ramps > 0):
        env = GrabObjWrapper(env, [f'moveable_box{i}' for i in range(np.max(n_boxes))] + ([f"ramp{i}:ramp" for i in range(n_ramps)]),
                             radius_multiplier=grab_radius_multiplier,
                             grab_exclusive=grab_exclusive,
                             obj_in_game_metadata_keys=['curr_n_boxes', 'curr_n_ramps'])
    
    if n_lidar_per_agent > 0:
        env = Lidar(env, n_lidar_per_agent=n_lidar_per_agent, visualize_lidar=visualize_lidar,
                    compress_lidar_scale=compress_lidar_scale)
        keys_copy += ['lidar']
        keys_external += ['lidar']
    
    if prep_obs:
        env = TrackStatWrapper(env, np.max(n_boxes), n_ramps, n_food)
    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy, keys_self_matrices=keys_mask_self)
    #l = ["box_obs", 'you_lock', 'team_lock', 'obj_lock']
    
    #env = SpoofEntityWrapper(env, np.max(n_boxes), l, ['mask_ab_obs'])
    if n_food:
        env = SpoofEntityWrapper(env, n_food, ['food_obs'], ['mask_af_obs'])
    keys_mask_external += ['mask_ab_obs_spoof', 'mask_af_obs_spoof']
    if max_n_agents is not None:
        env = SpoofEntityWrapper(env, max_n_agents, ['agent_qpos_qvel', 'hider', 'prep_obs'], ['mask_aa_obs'])
    # env = LockAllWrapper(env, remove_object_specific_lock=True)
    if not grab_out_of_vision and grab_box:
        env = MaskActionWrapper(env, 'action_pull',
                                ['mask_ab_obs'] + (['mask_ar_obs'] if n_ramps > 0 else []))
    if not grab_selective and grab_box:
        env = GrabClosestWrapper(env)
   #  env = NoActionsInPrepPhase(env, np.arange(n_hiders, n_hiders + n_seekers))
    env = DiscardMujocoExceptionEpisodes(env)
    env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel', 'hider'],
                                     # 'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                     # 'ramp_obs': ['ramp_obs'] + (['ramp_you_lock', 'ramp_team_lock', 'ramp_obj_lock'] if lock_ramp else [])
                                      })
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_other=keys_external + keys_mask_self + keys_mask_external)
                            
    return env
