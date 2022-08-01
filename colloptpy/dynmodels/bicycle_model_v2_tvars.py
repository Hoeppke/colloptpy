from .dynamic_model import DynamicModel
from track_model.track import Track
import torch as th
import numpy as np


class BicycleModel2Tvars(DynamicModel):

    def __init__(self):
        super().__init__()
        self.max_steer = np.radians(15.0)  # [rad] max steering angle
        self.L = 3.5  # [m] Wheel base of vehicle
        self.Lr = self.L / 2.0  # [m]
        self.Lf = self.L - self.Lr
        self.Cf = 4.0 # N/rad
        self.Cr = self.Cf*1.1
        self.Iz = 2250.0  # kg/m2
        self.m = 700.0  # kg
        self.track: Track = None
        # Aerodynamic and friction coefficients
        self.c_a = 1.7 # 1.3
        self.c_r = 0.1
        #
        self.x = self.add_state('x')
        self.y = self.add_state('y')
        self.yaw = self.add_state('yaw')
        self.vx = self.add_state('vx')
        self.vy = self.add_state('vy')
        self.omega = self.add_state('omega')
        # track tracing variables
        self.s_dist = self.add_state('s')
        self.nu_dist = self.add_state('nu')
        self.xi_angle = self.add_state('xi')
        # controls
        self.throttle = self.add_control('throttle')
        throttle_max = 20.0
        self.throttle.set_bounds(-throttle_max, throttle_max)
        self.steer_angle = self.add_control('steer')
        self.steer_angle.set_bounds(-self.max_steer, +self.max_steer)
        # parameters:
        self.name = 'linear dynamic bicycle model v2 with track vars'

    def forward(self, states: th.Tensor, ctrls: th.Tensor, device=None):
        # Compute the local velocity in the x-axis
        device = str(states.device)
        throttle = ctrls[:, self.throttle.get_idx()]
        delta = ctrls[:, self.steer_angle.get_idx()]
        #
        yaw = states[:, self.yaw.get_idx()]
        omega = states[:, self.omega.get_idx()]
        vx, vy = states[:, self.vx.get_idx()], states[:, self.vy.get_idx()]
        xi_angle = states[:, self.xi_angle.get_idx()]
        nu_dist = states[:, self.nu_dist.get_idx()]
        s_dist = states[:, self.s_dist.get_idx()]
        velocity = th.sqrt(vx*vx + vy*vy)
        delta_x = vx * th.cos(yaw) - vy * th.sin(yaw)
        delta_y = vx * th.sin(yaw) + vy * th.cos(yaw)
        delta_yaw = omega
        vfac = velocity / 20.0
        vfac = 9.81 * self.m + 2.0*vx**2.0
        Ffy = -vfac*self.Cf*th.atan(((vy + self.Lf *omega) /vx - delta))
        Fry = -vfac*self.Cr*th.atan((vy - self.Lr *omega) /vx)
        # Update the track variables
        track_curve = self.track.get_curvatures_th(s_dist).to(device)
        delta_s = (vx*th.cos(xi_angle) - vy*th.sin(xi_angle)) / (1.0 - nu_dist * track_curve)
        delta_nu = (vx*th.sin(xi_angle) + vy*th.cos(xi_angle))
        delta_xi = (omega - delta_s * track_curve)
        #
        R_x = self.c_r * vx
        F_aero = self.c_a * vx ** 2
        F_load = F_aero + R_x
        delta_vx = (throttle - Ffy * th.sin(delta) / self.m - F_load/self.m + vy * omega)
        delta_vy = (Fry / self.m + Ffy * th.cos(delta) / self.m - vx * omega)
        delta_omega = (Ffy * self.Lf * th.cos(delta) - Fry * self.Lr) / self.Iz
        out_th = th.stack([delta_x, delta_y, delta_yaw, delta_vx, delta_vy, delta_omega, delta_s, delta_nu, delta_xi]).T
        return out_th

    def set_track(self, track: Track):
        self.track = track

    def forward_noparam(self, states: th.Tensor, controls: th.Tensor, device='cpu') -> th.Tensor:
        '''
        Compute the state derivative.
        '''
        return self.forward(states, controls, device=device)
