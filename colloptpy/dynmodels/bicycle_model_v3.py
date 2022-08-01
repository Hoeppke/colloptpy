from .dynamic_model import DynamicModel
from track_model.track import Track
import torch as th
import numpy as np


class BicycleModel3(DynamicModel):

    def __init__(self):
        super().__init__()
        self.max_steer = np.radians(15.0)  # [rad] max steering angle
        self.L = 3.5  # [m] Wheel base of vehicle
        self.Lr = self.L / 2.0  # [m]
        self.Lf = self.L - self.Lr
        self.Cf = 1600.0 * 15.0  # N/rad
        self.Cr = self.Cf*1.1
        self.Iz = 2250.0  # kg/m2
        self.m = 700.0  # kg
        # Aerodynamic and friction coefficients
        self.c_a = 1.3
        self.c_r = 0.25
        #
        self.x = self.add_state('x')
        self.y = self.add_state('y')
        self.yaw = self.add_state('yaw')
        self.vx = self.add_state('vx')
        self.vy = self.add_state('vy')
        self.omega = self.add_state('omega')
        # track tracing variables
        # controls
        self.throttle = self.add_control('throttle')
        tmax = 25.0
        self.throttle.set_bounds(-tmax, tmax)
        self.steer_angle = self.add_control('steer')
        self.steer_angle.set_bounds(-self.max_steer, +self.max_steer)
        # parameters:
        self.name = 'linear dynamic bicycle model v3'

    def forward(self, states: th.Tensor, ctrls: th.Tensor, device=None):

        # Compute the local velocity in the x-axis
        throttle = ctrls[:, self.throttle.get_idx()]
        delta = ctrls[:, self.steer_angle.get_idx()]
        #
        yaw = states[:, self.yaw.get_idx()]
        omega = states[:, self.omega.get_idx()]
        vx, vy = states[:, self.vx.get_idx()], states[:, self.vy.get_idx()]
        velocity = th.sqrt(vx*vx + vy*vy)
        delta_x = vx * th.cos(yaw) - vy * th.sin(yaw)
        delta_y = vx * th.sin(yaw) + vy * th.cos(yaw)
        delta_yaw = omega
        vfac = velocity / 20.0
        Ffy = -vfac*self.Cf * th.atan2(((vy + self.Lf *omega) /vx - delta), th.ones_like(vx))
        Fry = -vfac*self.Cr * th.atan2((vy - self.Lr *omega) /vx, th.ones_like(vx))
        R_x = self.c_r * vx
        F_aero = self.c_a * vx ** 2
        F_load = F_aero + R_x
        F_throttle = self.m * throttle / (1.0 + 0.01*velocity)
        delta_vx = (F_throttle/self.m - Ffy * th.sin(delta) / self.m - F_load/self.m + vy * omega)
        delta_vy = (Fry / self.m + Ffy * th.cos(delta) / self.m - vx * omega)
        delta_omega = (Ffy * self.Lf * th.cos(delta) - Fry * self.Lr) / self.Iz
        out_th = th.stack([delta_x, delta_y, delta_yaw, delta_vx, delta_vy, delta_omega]).T
        return out_th

    def forward_noparam(self, states: th.Tensor, controls: th.Tensor, device='cpu') -> th.Tensor:
        '''
        Compute the state derivative.
        '''
        return self.forward(states, controls, device=device)
