{
    make_env: {
        "function": "mae_envs.envs.tag:make_env",
        args: {
            # Agents
            n_hiders: 4,
            n_seekers: 1,
            # Agent Actions
            grab_box: false,
            grab_out_of_vision: false,
            grab_selective: false,
            grab_exclusive: false,

            lock_box: false,
            lock_type: "all_lock_team_specific",
            lock_out_of_vision: false,

            # Scenario
            n_substeps: 15,
            horizon: 800,
            scenario: 'randomwalls',
            prep_fraction: 0.04,
            rew_type: "joint_zero_sum",
            restrict_rect: [0.1, 0.1, 5.9, 5.9],
            p_door_dropout: 0.0,
            quadrant_game_hider_uniform_placement: true,

            # Objects
            n_boxes: 0,
            box_only_z_rot: false,
            boxid_obs: false,

            n_ramps: 0,
            lock_ramp: false,
            penalize_objects_out: false,

            # Food
            n_food: 0,

            # Observations
            n_lidar_per_agent: 30,
            visualize_lidar: true,
            prep_obs: true,
        },
    },
}
