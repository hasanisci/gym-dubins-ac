import math


class Config:
    G = 9.8
    EPISODES = 1000

    # input dim
    window_width  = 800 # pixels
    window_height = 800 # pixels
    diagonal      = 800  # this one is used to normalize dist_to_intruder
    tick = 30
    scale = 30

    # distance param
    minimum_separation = 555 / scale
    NMAC_dist = 150 / scale
    horizon_dist = 4000 / scale
    initial_min_dist = 3000 / scale
    goal_radius = 600 / scale

    # speed
    min_speed = 50 / scale
    max_speed = 80 / scale
    d_speed = 5 / scale
    speed_sigma = 2 / scale
    position_sigma = 10 / scale

    # maximum steps of one episode
    max_steps = 1000

    # reward setting
    position_reward = 10. / 10.
    heading_reward  = 10 / 10.

    collision_penalty = -5. / 10
    outside_penalty   = -1. / 10


    step_penalty      = -0.01 / 10