import math
import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
from scipy.spatial.transform import Rotation as R

from .config import Config
from .ACEnvironment import ACEnvironment2D

class DubinsAC3Denv(gym.Env):
    metadata = {'render.modes': ['human']}

    _redAC = None
    _blueAC = None

    _vel_mps = None
    _action_time_s = None

    def __init__(self, actions='cont'):
        self._load_config()
        self.viewer = None


        self._vel_mps = 20
        self._action_time_s = 0.2
        self.actionIntegral = 0

        # 'err_x': spaces.Box(low=-self.area_width, high=self.area_width, shape=(1,), dtype=np.float32),
        # 'err_y': spaces.Box(low=-self.area_height, high=self.area_height, shape=(1,), dtype=np.float32),
        # 'err_z': spaces.Box(low=-self.area_height, high=self.area_height, shape=(1,), dtype=np.float32),
        # 'LOSxy_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'LOSz_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'ATAxy_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'ATAz_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'AA_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'redATAxy_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'redATAz_deg': spaces.Box(low=-180, high=180, shape=(1,), dtype=np.float32),
        # 'blue_heading': spaces.Box(low=0, high=359, shape=(1,), dtype=np.float32),
        # 'blue_bank': spaces.Box(low=-90, high=90, shape=(1,), dtype=np.float32),
        # 'blue_theta': spaces.Box(low=-90, high=90, shape=(1,), dtype=np.float32),

        # build observation space and action space
        lowlim=np.array([-self.window_width,  # err_x
                         -self.window_height, # err_y
                         -self.window_z,      # err_z
                         -180,                # LOSxy
                         -180,                # LOSz
                         -180,                # ATAxy
                         -180,                # ATAz
                         -180,                # AA
                         -180,                # redATAxy
                         -180,                # redATAz
                         -90,                 # bank
                         -90,                 # elevation
                         0                    # heading
                         ])
        highlim=np.array([self.window_width,  # err_x
                          self.window_height, # err_y
                          self.window_z,      # err_z
                          180,                # LOSxy
                          180,                # LOSz
                          180,                # ATAxy
                          180,                # ATAz
                          180,                # AA
                          180,                # redATAxy
                          180,                # redATAz
                          90,                 # bank
                          90,                 # elevation
                          359                 # heading
                         ])

        self.observation_space = spaces.Box(low=lowlim, high=highlim, shape=(13,), dtype=np.float32)

        if actions == 'discrete':
            self.action_space = spaces.Discrete(13)  # 0 -> 4 bank angle command: -90 45 0 45 90
        else:
            self.action_space = spaces.Box( low=np.array([-1., -1.]), high=np.array([1., 1.]), shape=(2,), dtype=np.float32 )  # 0 -> 4 bank angle command: -90 45 0 45 90

        self.seed(2)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):

        cmd_bank_deg = action[0] * 70.
        cmd_bank_deg = np.clip( cmd_bank_deg, -90, 90)

        cmd_theta_deg = -action[1] * 70.
        cmd_theta_deg = np.clip( cmd_theta_deg, -90, 90)

        self._blueAC.takeaction(cmd_bank_deg, cmd_theta_deg, self._vel_mps, self._action_time_s)

        self._redAC.takeaction(0, 0, self._vel_mps/2, self._action_time_s)

        if self._redAC._pos_m[0] > self.window_width:
            self._redAC._heading_rad = np.mod(self._redAC._heading_rad - np.pi/2, 2*np.pi)
        elif self._redAC._pos_m[0] < 0:
            self._redAC._heading_rad = np.mod(self._redAC._heading_rad + np.pi / 2, 2 * np.pi)

        if self._redAC._pos_m[1] > self.window_height:
            self._redAC._heading_rad = np.mod(self._redAC._heading_rad - np.pi/2, 2*np.pi)
        elif self._redAC._pos_m[1] < 0:
            self._redAC._heading_rad = np.mod(self._redAC._heading_rad + np.pi / 2, 2 * np.pi)

        self.actionIntegral += (cmd_bank_deg*cmd_bank_deg * 0.2 * 0.0001)

        envSta = self._get_sta_env_v2()

        reward, terminal, info = self._terminal_reward_2()

        return envSta, reward, terminal, info


    def reset(self):
        pos , head = self._random_pos2()
        pos[0] += 200
        pos[1] += 200
        pos[2] = -pos[2] - 200
        self._redAC = ACEnvironment2D(position=np.array([pos[0], pos[1], pos[2]]),
                                      vel_mps=self._vel_mps/2,
                                      heading_deg=head)

        bpos, bhead = self._random_pos()
        bpos[2] = -bpos[2]

        i = 0
        while (self._distance(pos, bpos) < 200) and i<20:
            bpos, bhead = self._random_pos()
            bpos[2]=-bpos[2]
            i += 1

        _, _, hdgxy, hdgz = self._calc_posDiff_hdg3d_deg(bpos, pos)
        self._blueAC = ACEnvironment2D(position=np.array([bpos[0], bpos[1], bpos[2]]),
                                       vel_mps=self._vel_mps,
                                       heading_deg= hdgxy)
        # self._blueAC._flightpath_rad = np.rad2deg(hdgz)

        return self._get_sta_env_v2()

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.window_width, self.window_height)
            self.viewer.set_bounds(0, self.window_width, 0, self.window_height)

        # if self.drone is None:
        #     return None

        import os
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        # draw red aircraft
        pos, _, att, pos_hist = self._redAC.get_sta()
        red_ac_img = rendering.Image(os.path.join(__location__, 'images/f16_red.png'), 36, 36)
        jtransform = rendering.Transform(rotation= -att[2], translation= np.array([pos[1], pos[0]]))
        red_ac_img.add_attr(jtransform)
        self.viewer.onetime_geoms.append(red_ac_img)
        self.viewer.draw_polyline(pos_hist[ ::5 , [-2, -3]],).set_color(1,0,0)

        transform2 = rendering.Transform(translation=(self.goal_pos[1], self.goal_pos[0]))  # Relative offset
        self.viewer.draw_circle(5).add_attr(transform2)

        transform2 = rendering.Transform(translation=(self.goal_pos[1], self.goal_pos[0]))  # Relative offset
        self.viewer.draw_circle(50,filled=False).add_attr(transform2)

        transform3 = rendering.Transform(translation=(pos[1], pos[0]))  # red dangerous circle
        self.viewer.draw_circle(250,filled=False).add_attr(transform3)

        l, r, t, b = 0, 10, -pos[2]/self.window_z*self.window_height, 0
        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        cart.set_color(1, 0, 0)
        self.viewer.add_onetime(cart)

        # draw blue aircraft
        pos, _, att, pos_hist = self._blueAC.get_sta()
        blue_ac_img = rendering.Image(os.path.join(__location__, 'images/f16_blue.png'), 36, 36)
        jtransform = rendering.Transform(rotation= -att[2], translation=np.array([pos[1], pos[0]]))
        blue_ac_img.add_attr(jtransform)
        self.viewer.onetime_geoms.append(blue_ac_img)
        self.viewer.draw_polyline(pos_hist[ ::5 , [-2, -3]]).set_color(0,0,1)

        l, r, t, b = 10, 20, -pos[2]/self.window_z*self.window_height, 0
        cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
        cart.set_color(0, 0, 1)
        self.viewer.add_onetime(cart)

        return self.viewer.render(return_rgb_array= mode == "rgb_array" )

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


    def _load_config(self):
        # input dim
        self.window_width  = Config.window_width
        self.window_height = Config.window_height
        self.window_z = Config.window_z
        self.EPISODES      = Config.EPISODES
        self.G             = Config.G
        self.tick          = Config.tick
        self.min_speed          = Config.min_speed
        self.max_speed          = Config.max_speed

    def _random_pos(self):
        pos0 = np.array([self.window_width/4, self.window_height/4, self.window_z/4])

        return (pos0 + np.random.uniform(low=np.array([0, 0, 0]), high=np.array([self.window_width/2, self.window_height/2, self.window_z/2])),
                np.random.uniform(low=-180, high=180))

    def _random_pos2(self):
        return (np.random.uniform(low=np.array([0, 0, 0]), high=np.array([self.window_width*0.5, self.window_height*0.5, self.window_z*0.5])),
                np.random.uniform(low=-180, high=180))


    def _get_sta_env_v2(self):
        Rpos, Rvel, Ratt_rad, _ = self._redAC.get_sta()
        Bpos, Bvel, Batt_rad, _ = self._blueAC.get_sta()

        Rpos[2] = -Rpos[2]
        Bpos[2] = -Bpos[2]


        target_dist = np.array([70., 0., 0.])
        r = R.from_euler('zyx', [Ratt_rad[2], 0, 0])
        target_dist = np.matmul(r.as_matrix(), target_dist)
        self.goal_pos = Rpos - target_dist

        self.errPos, self.errDist, _, _ = self._calc_posDiff_hdg3d_rad(Bpos, self.goal_pos)
        _, _, self.LOSxy_deg, self.LOSz_deg = self._calc_posDiff_hdg3d_rad(Bpos, Rpos)

        self.ATAxy_deg = np.rad2deg(self._pi_bound(self.LOSxy_deg - Batt_rad[2]))
        self.ATAz_deg = np.rad2deg(self._pi_bound(self.LOSz_deg - Batt_rad[1]))
        self.AA_deg = np.rad2deg(self._pi_bound(Ratt_rad[2] - self.LOSxy_deg))

        self.LOSxy_deg = np.rad2deg(self._pi_bound(self.LOSxy_deg))
        self.LOSz_deg = np.rad2deg(self._pi_bound(self.LOSz_deg))

        _, self.targetDist, self.redATAxy_deg, self.redATAz_deg = self._calc_posDiff_hdg3d_rad(Rpos, Bpos)
        self.redATAxy_deg = np.rad2deg(self._pi_bound(self.redATAxy_deg - Ratt_rad[2]))
        self.redATAz_deg = np.rad2deg(self._pi_bound(self.redATAz_deg - Ratt_rad[1]))

        return np.array([self.errPos[0],
                         self.errPos[1],
                         self.errPos[2],
                         self.LOSxy_deg,
                         self.LOSz_deg,
                         self.ATAxy_deg,
                         self.ATAz_deg,
                         self.AA_deg,
                         self.redATAxy_deg,
                         self.redATAz_deg,
                         np.rad2deg(Batt_rad[0]),
                         np.rad2deg(Batt_rad[1]),
                         np.rad2deg(Batt_rad[2]),
                         ], dtype=np.float32)

    def _get_sta_env_v2_redAC(self):
        Rpos, Rvel, Ratt_rad, _ = self._redAC.get_sta()
        Bpos, Bvel, Batt_rad, _ = self._blueAC.get_sta()
        Rpos[2] = -Rpos[2]
        Bpos[2] = -Bpos[2]


        target_dist = np.array([70., 0., 0.])
        r = R.from_euler('zyx', [Batt_rad[2], 0, 0])
        target_dist = np.matmul(r.as_matrix(), target_dist)
        goal_pos = Bpos - target_dist

        errPos, errDist, _, _ = self._calc_posDiff_hdg3d_rad(Rpos, goal_pos)
        _, _, LOSxy_deg, LOSz_deg = self._calc_posDiff_hdg3d_rad(Rpos, Bpos)

        ATAxy_deg = np.rad2deg(self._pi_bound(LOSxy_deg - Ratt_rad[2]))
        ATAz_deg = np.rad2deg(self._pi_bound(LOSz_deg - Ratt_rad[1]))
        AA_deg = np.rad2deg(self._pi_bound(Batt_rad[2] - LOSxy_deg))

        LOSxy_deg = np.rad2deg(self._pi_bound(LOSxy_deg))
        LOSz_deg = np.rad2deg(self._pi_bound(LOSz_deg))

        _, targetDist, redATAxy_deg, redATAz_deg = self._calc_posDiff_hdg3d_rad(Bpos, Rpos)
        redATAxy_deg = np.rad2deg(self._pi_bound(redATAxy_deg - Batt_rad[2]))
        redATAz_deg = np.rad2deg(self._pi_bound(redATAz_deg - Batt_rad[1]))

        return np.array([errPos[0],
                         errPos[1],
                         errPos[2],
                         LOSxy_deg,
                         LOSz_deg,
                         ATAxy_deg,
                         ATAz_deg,
                         AA_deg,
                         redATAxy_deg,
                         redATAz_deg,
                         np.rad2deg(Ratt_rad[0]),
                         np.rad2deg(Ratt_rad[1]),
                         np.rad2deg(Ratt_rad[2])
                         ], dtype=np.float32)


    def _terminal_reward_2(self):

        info = ''
        terminalState = False

        reward = 0
        result = "-"

        return reward, terminalState, {'info': info,
                                       'redObs': self._get_sta_env_v2_redAC(),
                                       'result':result}

    def _calc_posDiff_hdg3d_rad(self, start: np.array, dest: np.array):

        posDiff = dest - start
        angleDiffxy = np.arctan2(posDiff[1], posDiff[0])
        posxy = np.linalg.norm(posDiff[0:2])
        angleDiffz  = np.arctan2(  posDiff[2], posxy)

        distance = np.linalg.norm( posDiff )

        return posDiff, distance, angleDiffxy, angleDiffz

    def _calc_posDiff_hdg3d_deg(self, start: np.array, dest: np.array):

        pos, dist, anglexy, anglez = self._calc_posDiff_hdg3d_rad(start, dest)

        return pos, dist, np.rad2deg(anglexy), anglez

    def _distance(red, start: np.array, target: np.array):

        return np.linalg.norm(target - start)

    def _pi_bound(self, u):

        if u > np.pi:
            y = u - 2*np.pi
        elif u < -np.pi:
            y = u + 2 * np.pi
        else:
            y=u

        return y

    def _pi_bound_deg(self, u):

        if u > 180.:
            y = u - 360.
        elif u < -180.:
            y = u + 360.
        else:
            y=u

        return y
