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
from mae_envs.wrappers.manipulation import (GrabClosestWrapper, LockAllWrapper)
from mae_envs.wrappers.lidar import Lidar
from mae_envs.wrappers.line_of_sight import AgentAgentObsMask2D

from mae_envs.wrappers.team import TeamMembership

from mae_envs.modules.agents import Agents, AgentManipulation
from mae_envs.modules.walls import RandomWalls
from mae_envs.modules.objects import LidarSites
from mae_envs.wrappers.tag import TagPlayerWrapper
from mae_envs.modules.world import FloorAttributes, WorldConstants
from mae_envs.modules.util import (uniform_placement, close_to_other_object_placement,
                                   uniform_placement_middle)


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

        self.in_prep_phase = True

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


class TagRewardWrapper(gym.Wrapper):
    '''
        Establishes hide and seek dynamics (see different reward types below). Defaults to first half
            of agents being hiders and second half seekers unless underlying environment specifies
            'n_hiders' and 'n_seekers'.
        Args:
            rew_type (string): can be
                'selfish': hiders and seekers play selfishly. Seekers recieve 1.0 if they can
                    see any hider and -1.0 otherwise. Hiders recieve 1.0 if they are seen by no
                    seekers and -1.0 otherwise.
                'joint_mean': hiders and seekers recieve the mean reward of their team
                'joint_zero_sum': hiders recieve 1.0 only if all hiders are hidden and -1.0 otherwise.
                    Seekers recieve 1.0 if any seeker sees a hider.
            reward_scale (float): scales the reward by this factor
    '''
    def __init__(self, env, n_players, n_it, tag_radius, reward_scale=1.0):
        super().__init__(env)
        self.n_agents = self.unwrapped.n_agents
        self.n_players = n_players
        self.n_it = n_it

        self.reward_scale = reward_scale
        assert n_it < n_players, "must be less players it than there are players"

        self.metadata['n_players'] = n_players
        self.metadata['n_it'] = n_it

        # Agent names are used to plot agent-specific rewards on tensorboard
        self.unwrapped.agent_names = [f'player{i}' for i in range(self.n_players)]


    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        it_mask = obs['it_mask']

        this_rew = np.ones((self.n_players,))

        this_rew = [-1 if it_mask[i] else this_rew[i] for i in range(this_rew)]

        this_rew *= self.reward_scale
        rew += this_rew

        return obs, rew, done, info


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
             floor_size=6.0, grid_size=30, door_size=2,
             n_hiders=1, n_seekers=1, max_n_agents=None,
             n_boxes=2, n_ramps=1, n_elongated_boxes=0,
             rand_num_elongated_boxes=False, n_min_boxes=None,
             box_size=0.5, boxid_obs=False, box_only_z_rot=True,
             rew_type='joint_zero_sum',
             lock_box=True, grab_box=True, lock_ramp=True,
             lock_type='any_lock_specific',
             lock_grab_radius=0.25, lock_out_of_vision=True, grab_exclusive=False,
             grab_out_of_vision=False, grab_selective=False,
             box_floor_friction=0.2, other_friction=0.01, gravity=[0, 0, -50],
             action_lims=(-0.9, 0.9), polar_obs=True,
             scenario='quadrant', quadrant_game_hider_uniform_placement=False,
             p_door_dropout=0.0,
             n_rooms=4, random_room_number=True, prob_outside_walls=1.0,
             n_lidar_per_agent=0, visualize_lidar=False, compress_lidar_scale=None,
             hiders_together_radius=None, seekers_together_radius=None,
             prep_fraction=0.4, prep_obs=False,
             team_size_obs=False,
             restrict_rect=None, penalize_objects_out=False,
             n_food=0, food_radius=None, food_respawn_time=None, max_food_health=1,
             food_together_radius=None, food_rew_type='selfish', eat_when_caught=False,
             food_reward_scale=1.0, food_normal_centered=False, food_box_centered=False,
             n_food_cluster=1, tag_radius=.25, n_players=6, n_it=2):

    grab_radius_multiplier = lock_grab_radius / box_size

    env = Base(n_agents=n_hiders + n_seekers, n_substeps=n_substeps, horizon=horizon,
               floor_size=floor_size, grid_size=grid_size,
               action_lims=action_lims,
               deterministic_mode=deterministic_mode)

    # only support random walls

    env.add_module(RandomWalls(
        grid_size=grid_size, num_rooms=n_rooms,
        random_room_number=random_room_number, min_room_size=6,
        door_size=door_size,
        prob_outside_walls=prob_outside_walls, gen_door_obs=False))

    cell_size = floor_size / grid_size


    # BEGIN Agent placement and creation
    first_hider_placement = uniform_placement

    if hiders_together_radius is not None:
        htr_in_cells = np.ceil(hiders_together_radius / cell_size).astype(int)

        env.metadata['hiders_together_radius'] = htr_in_cells

        close_to_first_hider_placement = close_to_other_object_placement(
                                            "agent", 0, "hiders_together_radius")

        agent_placement_fn = [first_hider_placement] + \
                             [close_to_first_hider_placement] * (n_hiders - 1)
    else:
        agent_placement_fn = [first_hider_placement] * n_hiders

    first_seeker_placement = uniform_placement

    if seekers_together_radius is not None:
        str_in_cells = np.ceil(seekers_together_radius / cell_size).astype(int)

        env.metadata['seekers_together_radius'] = str_in_cells

        close_to_first_seeker_placement = close_to_other_object_placement(
                                            "agent", n_hiders, "seekers_together_radius")

        agent_placement_fn += [first_seeker_placement] + \
                              [close_to_first_seeker_placement] * (n_seekers - 1)
    else:
        agent_placement_fn += [first_seeker_placement] * (n_seekers)

    env.add_module(Agents(n_hiders + n_seekers,
                          placement_fn=agent_placement_fn,
                          color=[np.array((66., 235., 244., 255.)) / 255] * n_hiders + [(1., 0., 0., 1.)] * n_seekers,
                          friction=other_friction,
                          polar_obs=polar_obs))

    # END Agent placement and creation

    if n_lidar_per_agent > 0 and visualize_lidar:
        env.add_module(LidarSites(n_agents=n_hiders + n_seekers, n_lidar_per_agent=n_lidar_per_agent))

    # no food- keep code to reference
    '''
    if False and n_food > 0:
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
    '''

    env.add_module(AgentManipulation())
    if box_floor_friction is not None:
        env.add_module(FloorAttributes(friction=box_floor_friction))

    env.add_module(WorldConstants(gravity=gravity))

    env.reset()


    keys_self = ['agent_qpos_qvel', 'hider', 'prep_obs']
    keys_mask_self = ['mask_aa_obs']
    keys_external = ['agent_qpos_qvel']
    keys_copy = ['you_lock', 'team_lock', 'ramp_you_lock', 'ramp_team_lock']
    keys_mask_external = []
    env = SplitMultiAgentActions(env)
    if team_size_obs:
        keys_self += ['team_size']

    env = TeamMembership(env, np.append(np.zeros((n_hiders,)), np.ones((n_seekers,))))
    env = AgentAgentObsMask2D(env)

    hider_obs = np.array([[1]] * n_hiders + [[0]] * n_seekers)
    env = AddConstantObservationsWrapper(env, new_obs={'hider': hider_obs})

    env = TagPlayerWrapper(env)
    env = TagRewardWrapper(env, n_players=n_players, n_it=n_it, tag_radius=tag_radius)

    # restricts initial spawn to center
    env = DiscretizeActionWrapper(env, 'action_movement')


    if n_lidar_per_agent > 0:
        env = Lidar(env, n_lidar_per_agent=n_lidar_per_agent, visualize_lidar=visualize_lidar,
                    compress_lidar_scale=compress_lidar_scale)
        keys_copy += ['lidar']
        keys_external += ['lidar']

    if prep_obs:
        env = TrackStatWrapper(env, np.max(n_boxes), n_ramps, n_food)

    env = SplitObservations(env, keys_self + keys_mask_self, keys_copy=keys_copy, keys_self_matrices=keys_mask_self)
    env = SpoofEntityWrapper(env, np.max(n_boxes), ['box_obs', 'you_lock', 'team_lock', 'obj_lock'], ['mask_ab_obs'])


    keys_mask_external += ['mask_ab_obs_spoof', 'mask_af_obs_spoof']
    if max_n_agents is not None:
        env = SpoofEntityWrapper(env, max_n_agents, ['agent_qpos_qvel', 'hider', 'prep_obs'], ['mask_aa_obs'])

    env = LockAllWrapper(env, remove_object_specific_lock=True)

    # if trying to grab a box
    if not grab_out_of_vision and grab_box:
        env = MaskActionWrapper(env, 'action_pull',
                                ['mask_ab_obs'] + (['mask_ar_obs'] if n_ramps > 0 else []))

    if not grab_selective and grab_box:
        env = GrabClosestWrapper(env)

    env = DiscardMujocoExceptionEpisodes(env)
    env = ConcatenateObsWrapper(env, {'agent_qpos_qvel': ['agent_qpos_qvel', 'hider', 'prep_obs'],
                                      'box_obs': ['box_obs', 'you_lock', 'team_lock', 'obj_lock'],
                                      'ramp_obs': ['ramp_obs'] + (['ramp_you_lock', 'ramp_team_lock', 'ramp_obj_lock'] if lock_ramp else [])})
    env = SelectKeysWrapper(env, keys_self=keys_self,
                            keys_other=keys_external + keys_mask_self + keys_mask_external)
    return env
