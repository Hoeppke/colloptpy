from .dynamic_model import DynamicModel
from track_model.track import Track
import torch as th
import numpy as np


class BicycleModel(DynamicModel):

    def __init__(self):
        super(BicycleModel, self).__init__()
        self.track: Track = None
        self.max_steer = np.radians(30.0)  # [rad] max steering angle
        self.L = 2.9  # [m] Wheel base of vehicle
        self.Lr = self.L / 2.0  # [m]
        self.Lf = self.L - self.Lr
        self.Cf = 1600.0 * 2.0 # N/rad
        self.Cr = 1700.0 * 2.0 # N/rad
        self.Iz = 2250.0  # kg/m2
        self.mass = 750.0  # kg
        # Aerodynamic and friction coefficients
        self.c_a = 1.36
        self.c_r1 = 0.10
        self.num_static_params = 9
        # position variables
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
        self.throttle.set_bounds(-5.0, +5.0)
        self.steer_angle = self.add_control('steer')
        self.steer_angle.set_bounds(-self.max_steer, +self.max_steer)
        # parameters:
        self.track_angle = self.add_param('track_angle')
        self.track_curvature = self.add_param('track_curve')
        self.name = 'Dynamic Bicycle model'

    def set_track(self, track: Track):
        self.track = track

    def get_main_variable(self) -> str:
        return self.s_dist.get_name()

    def forward(self, states: th.Tensor, controls: th.Tensor, params: th.Tensor) -> th.Tensor:
        '''
        Compute the state derivative.
        '''
        device: str = str(states.device)
        if isinstance(params, np.ndarray):
            params = th.tensor(params)
        #delta = np.clip(delta, -self.max_steer, self.max_steer)
        #self.load(states, controls, params)
        #
        # one_th = th.ones(1)
        #
        num_nodes = states.shape[0]
        num_states = states.shape[1]
        delta = controls[:, self.steer_angle.get_idx()]
        throttle = controls[:, self.throttle.get_idx()]
        #delta = delta, -self.max_steer, self.max_steer)
        vx = states[:, self.vx.get_idx()]
        vy = states[:, self.vy.get_idx()]
        yaw = states[:, self.yaw.get_idx()]
        omega = states[:, self.omega.get_idx()]
        s_dist = states[:, self.s_dist.get_idx()]
        nu_dist = states[:, self.nu_dist.get_idx()]
        xi_angle = states[:, self.xi_angle.get_idx()]
        # xi = yaw - track_angle.
        track_angle = params[:, self.track_angle.get_idx()]
        track_curve = params[:, self.track_curvature.get_idx()]
        #
        delta_x = vx*th.cos(yaw) - vy *th.sin(yaw)
        delta_y = vx*th.sin(yaw) + vy *th.cos(yaw)
        delta_yaw = omega
        alpha_r = th.atan((vy + self.Lr * omega) / vx)
        alpha_f = th.atan((vy + self.Lf * omega) / vx - delta)
        Ffy = -self.Cf*alpha_f
        Fry = -self.Cr*alpha_r
        R_x = self.c_r1 * vx
        F_aero = self.c_a * vx ** 2
        F_load = F_aero + R_x
        delta_vx = throttle - Ffy *th.sin(delta) / self.mass - F_load/self.mass #+ (vy * omega)
        delta_vy = (Fry / self.mass) + (Ffy*th.cos(delta) / self.mass) #- (vx * omega)
        delta_omega = (Ffy * self.Lf *th.cos(delta) - Fry * self.Lr) / self.Iz
        # Update the track variables
        delta_s = (vx*th.cos(xi_angle) - vy*th.sin(xi_angle)) / (1.0 - nu_dist * track_curve)
        delta_nu = (vx*th.sin(xi_angle) + vy*th.cos(xi_angle))
        delta_xi = (omega - delta_s * track_curve)
        #print('Delta nu: {}, yaw: {}, xi: {}'.format(delta_nu, yaw, xi_angle))
        out_vector = th.zeros((num_nodes, num_states), dtype=th.float64).to(device)
        out_vector[:, self.x.get_idx()] += delta_x
        out_vector[:, self.y.get_idx()] = delta_y
        out_vector[:, self.yaw.get_idx()] = delta_yaw
        out_vector[:, self.vx.get_idx()] = delta_vx
        out_vector[:, self.vy.get_idx()] = delta_vy
        out_vector[:, self.omega.get_idx()] = delta_omega
        out_vector[:, self.s_dist.get_idx()] = delta_s
        out_vector[:, self.nu_dist.get_idx()] = delta_nu
        out_vector[:, self.xi_angle.get_idx()] = delta_xi
        #out_th = th.cat(out_vector)
        return out_vector

    def forward_noparam(self, states: th.Tensor, controls: th.Tensor) -> th.Tensor:
        '''
        Compute the state derivative.
        '''
        device = str(states.device)
        s_dist = states[:, self.s_dist.get_idx()]
        track_angle = self.track.get_orientations_th(s_dist).to(device)
        track_curve = self.track.get_curvatures_th(s_dist).to(device)
        params = th.stack([track_angle, track_curve], dim=1).to(device)
        return self.forward(states, controls, params)
