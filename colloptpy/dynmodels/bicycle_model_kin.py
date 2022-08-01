from .dynamic_model import DynamicModel
import torch as th
import numpy as np


class KinematicBicycleModel(DynamicModel):

    def __init__(self):
        """
        2D Kinematic Bicycle Model
        """
        super().__init__()
        self.max_steer = np.radians(10.0)  # [rad] max steering angle
        self.L = 5.0  # [m] Wheel base of vehicle
        #self.Lr = self.L / 2.0  # [m]
        #self.Lf = self.L - self.Lr
        #self.Cf = 1600.0 * 2.0# N/rad
        #self.Cr = 1700.0 * 2.0 # N/rad
        #self.Iz = 2250.0  # kg/m2
        self.mass = 750.0  # kg
        # Aerodynamic and friction coefficients
        self.c_a = 0.0
        self.c_r = 0.01
        self.num_static_params = 9
        #
        self.x = self.add_state('x')
        self.y = self.add_state('y')
        self.yaw = self.add_state('yaw')
        self.vx = self.add_state('vx')
        self.vy = self.add_state('vy')
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
        self.name = 'Kinematic bicycle model'

    def forward(self, states: th.Tensor, ctrls: th.Tensor, params: th.Tensor):

        # Compute the local velocity in the x-axis
        yaw = states[self.yaw.get_idx()]
        vx, vy = states[self.vx.get_idx()], states[self.vy.get_idx()]
        velocity = th.sqrt(vx*vx + vy*vy)
        throttle = ctrls[self.throttle.get_idx()]
        delta = ctrls[self.steer_angle.get_idx()]
        track_angle = params[self.track_angle.get_idx()]
        track_curve = params[self.track_curvature.get_idx()]
        xi_angle = states[self.xi_angle.get_idx()]
        nu_dist = states[self.nu_dist.get_idx()]
        #
        f_load = velocity * (self.c_r + self.c_a * velocity)
        d_velocity = throttle - f_load

        # Compute the state change rate
        x_dot = velocity * th.cos(yaw)
        y_dot = velocity * th.sin(yaw)
        omega = velocity * th.tan(delta) / self.L

        # Update the track variables
        delta_s = (vx*th.cos(yaw-track_angle) - vy*th.sin(yaw-track_angle)) / (1.0 - nu_dist * track_curve)
        delta_nu = (vx*th.sin(yaw-track_angle) + vy*th.cos(yaw-track_angle))
        delta_xi = (omega - delta_s * track_curve)
        #print('Delta nu: {}, yaw: {}, xi: {}'.format(delta_nu, yaw, xi_angle))

        delta_s = (vx*th.cos(xi_angle) - vy*th.sin(xi_angle)) / (1.0 - nu_dist * track_curve)
        delta_nu = (vx*th.sin(xi_angle) + vy*th.cos(xi_angle))
        delta_xi = (omega - delta_s * track_curve)
        # Compute the final state using the discrete time model
        out_vector = [None for var in self.states]
        out_vector[self.x.get_idx()] = th.ravel(x_dot)
        out_vector[self.y.get_idx()] = th.ravel(y_dot)
        out_vector[self.vx.get_idx()] = th.ravel(d_velocity)
        out_vector[self.vy.get_idx()] = th.Tensor([0.0])
        out_vector[self.yaw.get_idx()] = th.ravel(omega)
        out_vector[self.s_dist.get_idx()] = th.ravel(delta_s)
        out_vector[self.nu_dist.get_idx()] = th.ravel(delta_nu)
        out_vector[self.xi_angle.get_idx()] = th.ravel(delta_xi)
        out_th = th.cat(out_vector)
        return out_th
