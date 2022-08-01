from .dynamic_model import DynamicModel
from track_model.track import Track
import torch as th
import numpy as np


class Vehicle4W(DynamicModel):

    def __init__(self):
        super().__init__()
        self.max_steer = np.radians(5.0)  # [rad] max steering angle
        self.L = 3.4  # [m] Wheel base of vehicle
        self.Lf = 1.8
        self.Lr = self.L - self.Lf
        self.kappa = 0.1
        self.mu_x = 1.6
        self.mu_y = 1.7
        self.Q = 1.9
        self.Iz = 450.0
        self.Iz_wheel = 4.22
        self.wf = 0.73
        self.wr = 0.73
        self.wheel_radius = 0.33
        self.k_d = 10.47
        self.c_drag = 0.9
        self.c_down = 3.0
        self.mass = 660.0  # kg
        self.h = 0.3 # meters
        self.air_density = 1.2
        self.frontal_area = 1.5
        # Aerodynamic and friction coefficients
        # position variables
        self.x = self.add_state('x')
        self.y = self.add_state('y')
        self.yaw = self.add_state('yaw')
        self.vx = self.add_state('vx')
        self.vy = self.add_state('vy')
        self.omega = self.add_state('omega')
        # track tracing variables, don't do track tracing
        #self.s_dist = self.add_state('s')
        #self.nu_dist = self.add_state('nu')
        #self.xi_angle = self.add_state('xi')
        # controls
        self.throttle = self.add_control('throttle')
        self.throttle_max = +750.0
        self.throttle.set_bounds(-self.throttle_max, self.throttle_max)
        self.steer_angle = self.add_control('steer')
        self.steer_angle.set_bounds(-self.max_steer, +self.max_steer)
        # parameters:
        self.name = 'Dynamic Bicycle model v2'

    def _compute_slip_angles(self, states: th.Tensor, controls: th.Tensor) -> th.Tensor:
        '''
        Compute, for each row, the four slip angles
        1) alpha_rr
        2) alpha_rl
        3) alpha_fr
        4) alpha_fl
        '''
        vx = states[:, self.vx.get_idx()]
        vy = states[:, self.vy.get_idx()]
        psi_d = states[:, self.omega.get_idx()]
        delta = controls[:, self.steer_angle.get_idx()]
        sind, cosd = th.sin(delta), th.cos(delta)
        a_rr = th.arctan((vy-psi_d*self.Lr) / (vx-psi_d*self.wr))
        a_rl = th.arctan((vy-psi_d*self.Lr) / (vx+psi_d*self.wr))
        a1 = sind*(psi_d*self.wf - vx)+cosd*(psi_d*self.Lf+vy)
        b1 = cosd*(vx - psi_d*self.wf)+sind*(psi_d*self.Lf+vy)
        a2 = cosd*(psi_d*self.Lf + vy)-sind*(psi_d*self.wf+vx)
        b2 = cosd*(psi_d*self.wf+vx)+sind*(psi_d*self.Lf+vy)
        a_fr = th.arctan(a1 / b1)
        a_fl = th.arctan(a2 / b2)
        return th.stack([a_rr, a_rl, a_fr, a_fl]).T

    def forward(self, states: th.Tensor, controls: th.Tensor, device='cpu') -> th.Tensor:
        '''
        Compute the state derivative.
        '''
        num_nodes = states.shape[0]
        #num_states = states.shape[1]
        delta = controls[:, self.steer_angle.get_idx()]
        throttle = controls[:, self.throttle.get_idx()]
        #delta = delta, -self.max_steer, self.max_steer)
        vx = states[:, self.vx.get_idx()]
        vy = states[:, self.vy.get_idx()]
        yaw = states[:, self.yaw.get_idx()]
        omega = states[:, self.omega.get_idx()]
        #
        delta_x = vx*th.cos(yaw) - vy *th.sin(yaw)
        delta_y = vx*th.sin(yaw) + vy *th.cos(yaw)
        delta_yaw = omega
        sin_d, cos_d = th.sin(delta), th.cos(delta)
        #
        speed = th.sqrt(vx*vx + vy*vy)
        f_mass = self.mass*9.81
        f_down = 0.5*self.c_down*self.air_density*self.frontal_area*(speed**2)
        f_drag = -0.5*self.c_drag*self.air_density*self.frontal_area*(speed**2)
        f_zwheel = 0.25*(f_mass + f_down)
        alpha = self._compute_slip_angles(states, controls)
        print('\nSlip angles:', alpha)
        kappa = self.kappa*th.ones_like(alpha) * (throttle / self.throttle_max)
        rho = th.sqrt(alpha**2+kappa**2)+1e-6
        #
        f_x = self.mu_x*f_zwheel*(kappa/rho)
        f_y = -self.mu_y*f_zwheel*(alpha/rho)
        fx_cmass = cos_d*(f_x[:, 2] + f_x[:, 3]) - sin_d*(f_y[:, 2] + f_y[:, 3]) + (f_x[:, 0] + f_x[:, 1]) + f_drag
        fy_cmass = cos_d*(f_y[:, 2] + f_y[:, 3]) + sin_d*(f_x[:, 2] + f_x[:, 3]) + (f_y[:, 0] + f_y[:, 1])
        f_moment0 = self.Lf*(cos_d*(f_y[:, 2]+f_y[:, 3]) + sin_d*(f_x[:, 2]+f_x[:, 3]))
        f_moment1 = self.wf*(sin_d*f_y[:, 2]-cos_d*f_x[:, 2]) - self.wr*f_x[:, 0]
        f_moment2 = self.wf*(cos_d*f_x[:, 3]-sin_d*f_y[:, 3]) + self.wr*f_x[:, 1]
        f_moment3 = -self.Lr * (f_y[:, 0]+f_y[:, 1])
        f_moment = f_moment0 + f_moment1 + f_moment2 + f_moment3
        #
        print('V={}, F_x={}, F_y={}, M={}'.format(speed, fx_cmass, fy_cmass, f_moment))
        delta_vx = (+self.mass*omega*vy + fx_cmass) / self.mass
        delta_vy = (-self.mass*omega*vx + fy_cmass) / self.mass
        delta_omega = f_moment / self.Iz
        # Update the track variables
        out_tensors = [delta_x, delta_y, delta_yaw, delta_vx, delta_vy, delta_omega]
        out_vector = th.stack(out_tensors).T
        return out_vector

    def forward_noparam(self, states: th.Tensor, controls: th.Tensor, device='cpu') -> th.Tensor:
        '''
        Compute the state derivative.
        '''
        return self.forward(states, controls, device=device)
