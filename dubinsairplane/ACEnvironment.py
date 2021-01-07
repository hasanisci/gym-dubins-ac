import numpy as np
import matplotlib.pyplot as plt


def _pi_bound_deg(u):
    u_sat = np.mod(u, 360)

    if u_sat > 180:
        y = u_sat - 360
    else:
        y = u

    return y


class ACEnvironment2D:
    _pos_m = None
    _pos_history = None
    _vel_mps = None  # Velocity

    _bank_rad = None  # Bank angle , radians
    _flightpath_rad = None  # flight path angle , radians
    _heading_rad = None  # heading angle , radians

    _cmd_bank_rad = None
    _cmd_flightpath_rad = None
    _cmd_vel_mps = None

    _tau_bank_s = None
    _tau_flightpath_s = None
    _tau_vel_s = None

    _dt = 0.05  # seconds
    _t = 0.

    def __init__(self, position=np.array([0., 0., 0]),
                 heading_deg=0.,
                 vel_mps=0):

        self.reset()
        self._pos_m = position
        self._pos_history[0, :] = position
        self._heading_rad = np.deg2rad(heading_deg)
        self._vel_mps = vel_mps

    def reset(self):
        self._pos_m = np.zeros((1, 3), dtype=float)  # pos X,Y,Z
        self._pos_history = np.zeros((1, 3), dtype=float)  # pos X,Y,Z
        self._vel_mps = 0.

        self._bank_rad = 0.
        self._flightpath_rad = 0.
        self._heading_rad = 0.

        self._cmd_vel_mps = 0.
        self._cmd_flightpath_rad = 0.
        self._cmd_bank_rad = 0.

        self._tau_bank_s = 0.05
        self._tau_flightpath_s = 0.05
        self._tau_vel_s = 1.5

        self._t = 0.

    def get_sta(self):

        return (self._pos_m,
                self._vel_mps,
                np.array([self._bank_rad, self._flightpath_rad, self._heading_rad]),
                self._pos_history)

    def set_cmd_bank(self, cmd_deg):
        self._cmd_bank_rad = np.deg2rad(_pi_bound_deg(cmd_deg))

    def set_cmd_flightpath(self, flightpath_deg):
        self._cmd_flightpath_rad = np.deg2rad(_pi_bound_deg(flightpath_deg))

    def set_cmd_vel(self, cmd_mps):
        if cmd_mps < 0:
            cmd_mps = 0

        self._cmd_vel_mps = cmd_mps

    def simfor(self, Time_s):

        startTime = self._t
        while self._t < (startTime + Time_s):

            vx = self._vel_mps * np.cos(self._flightpath_rad) * np.cos(self._heading_rad)
            vy = self._vel_mps * np.cos(self._flightpath_rad) * np.sin(self._heading_rad)
            vz = -self._vel_mps * np.sin(self._flightpath_rad)

            self._pos_m[0] = self._pos_m[0] + self._dt * vx
            self._pos_m[1] = self._pos_m[1] + self._dt * vy
            self._pos_m[2] = self._pos_m[2] + self._dt * vz

            if self._vel_mps > 0:  # eger bir hiz varsa degisim olacak aksi durumda sifira bolme
                heading_dot_rad = 9.81 / self._vel_mps * np.tan(self._bank_rad) * np.cos(self._flightpath_rad)
            else:
                heading_dot_rad = 0
            bank_dot = (self._cmd_bank_rad - self._bank_rad) / self._tau_bank_s
            flightpath_dot = (self._cmd_flightpath_rad - self._flightpath_rad) / self._tau_flightpath_s
            vel_dot = (self._cmd_vel_mps - self._vel_mps) / self._tau_vel_s

            self._bank_rad += self._dt * bank_dot
            self._flightpath_rad += self._dt * flightpath_dot
            self._heading_rad += self._dt * heading_dot_rad
            self._vel_mps += self._dt * vel_dot

            self._pos_history = np.append( self._pos_history, [self._pos_m], axis=0 )
            #self._pos_history = np.concatenate((self._pos_history, self._pos_m), axis=0)

            self._t += self._dt

        return (self._pos_m,
                self._vel_mps,
                np.array([self._bank_rad, self._flightpath_rad, self._heading_rad]),
                self._pos_history)

    def sim(self, simTime_s):

        self.reset()
        return self.simfor(simTime_s)

    def takeaction(self, cmd_bank_deg, cmd_flightpath, cmd_vel_mps, actiontime=1):

        self.set_cmd_bank(cmd_bank_deg)
        self.set_cmd_flightpath(cmd_flightpath)
        self.set_cmd_vel(cmd_vel_mps)

        return self.simfor(actiontime)  # sim for 5 seconds to get the results of an action
