import numpy as np
import torch as th
import matplotlib.pyplot as plt
from track_model.track import Track
from matplotlib.patches import Rectangle, Circle
from colloptpy.copt.collocation_problem import CollocationProblem
from colloptpy.layout.domain_layout import DomainLayout
from colloptpy.layout.domain import Domain
from colloptpy.dynmodels.dynamic_model import DynamicModel


class VehicleDyanmics(DynamicModel):
    """
    Dynamics bicycle model vehicle dynamics with aerodynamic down-force model.
    """

    def __init__(self, track: Track):
        """
        Dynamic bicycle vehicle model in a track oriented coordinate system
        """
        super().__init__()
        self.L = 3.5  # [m] Wheel base of vehicle
        self.Lr = self.L / 2.0  # [m]
        self.Lf = self.L - self.Lr
        self.Cf = 4.0 # N/rad
        self.Cr = self.Cf*1.1
        self.Iz = 2250.0  # kg/m2
        self.m = 700.0  # kg
        self.track: Track = track
        # Aerodynamic and friction coefficients
        self.c_a = 1.7
        self.c_r = 0.1
        # States
        self.px = self.add_state('x')
        self.py = self.add_state('y')
        self.vx = self.add_state('vx')
        self.vy = self.add_state('vy')
        self.yaw = self.add_state('yaw')
        self.omega = self.add_state('omega')
        self.nu_dist = self.add_state('nu')
        self.xi_angle = self.add_state('xi')
        # Controls
        self._throttle_max = 20.0  # maximal acceleration of +/- 20.0 m/s^2
        self._steer_max = np.radians(15.0)  # maximal steering angle of +/- 15 degrees
        self.throttle = self.add_control('throttle')
        self.throttle.set_bounds(-self._throttle_max, +self._throttle_max)
        self.steer_angle = self.add_control('steer')
        self.steer_angle.set_bounds(-self._steer_max, +self._steer_max)
        self.name = 'dynamic bicycle model with linear tyre forces in a track centric coordinate system'

    def forward(self, states: th.Tensor, ctrls: th.Tensor, s_dist: th.Tensor):
        # Compute the local velocity in the x-axis
        dev, dtype = states.device, states.dtype
        throttle = ctrls[:, self.throttle.get_idx()]
        delta = ctrls[:, self.steer_angle.get_idx()]
        #
        yaw = states[:, self.yaw.get_idx()]
        omega = states[:, self.omega.get_idx()]
        vx, vy = states[:, self.vx.get_idx()], states[:, self.vy.get_idx()]
        xi_angle = states[:, self.xi_angle.get_idx()]
        nu_dist = states[:, self.nu_dist.get_idx()]
        # omega is the yaw rate
        delta_yaw = omega
        # Compute aero forces
        velocity = th.sqrt(vx*vx + vy*vy)
        vfac = velocity / 20.0
        vfac = 9.81 * self.m + 2.0*vx**2.0
        Ffy = -vfac*self.Cf*(th.atan((vy + self.Lf *omega) /vx) - delta)
        Fry = -vfac*self.Cr*th.atan((vy - self.Lr *omega) /vx)
        # Compute track derivatives
        track_curve = self.track.get_curvatures_th(s_dist).to(states.device)
        delta_s = (vx*th.cos(xi_angle) - vy*th.sin(xi_angle)) / (1.0 - nu_dist * track_curve)
        delta_nu = vx*th.sin(xi_angle) + vy*th.cos(xi_angle)
        delta_xi = omega - delta_s * track_curve
        #
        R_x = self.c_r * vx
        F_aero = self.c_a * vx ** 2
        F_load = F_aero + R_x
        delta_x = vx * th.cos(yaw) - vy * th.sin(yaw)
        delta_y = vx * th.sin(yaw) + vy * th.cos(yaw)
        delta_vx = (throttle - Ffy * th.sin(delta) / self.m - F_load/self.m + vy * omega)
        delta_vy = (Fry / self.m + Ffy * th.cos(delta) / self.m - vx * omega)
        delta_omega = (Ffy * self.Lf * th.cos(delta) - Fry * self.Lr) / self.Iz
        # Compute derivatives with respect to s_dist
        out_vectors = [delta_x, delta_y, delta_vx, delta_vy, delta_yaw, delta_omega, delta_nu, delta_xi]
        delta_s_mtx = delta_s.repeat((self.num_states, 1)).T
        out_th = th.stack(out_vectors, axis=1) / delta_s_mtx
        return out_th

class VehicleDomain(Domain):

    def __init__(self, model: VehicleDyanmics, layout: DomainLayout, track: Track):
        """
        Initialise a cartpole domain 
        """
        self.track: Track = track
        super().__init__(model, layout)

    def get_state_bounds(self, s_pos: float):
        """
        Get the state bounds
        """
        eps = 1e-8
        twidth = self.track.get_width()
        tpos = self.track.get_point(s_pos)
        x_bound = [-10.0**6, +10.0**6]
        y_bound = [-10.0**6, +10.0**6]
        vx_bound = [0.1, +100.0]
        vy_bound = [-20.0, +20.0]
        yaw_bound = [tpos.get_orientation()-np.pi, tpos.get_orientation()+np.pi]
        omega_bound = [-10.0*np.pi, +10.0*np.pi]
        xi_bound = [-0.5*np.pi, +0.5*np.pi]
        nu_bound = [-0.5*twidth, +0.5*twidth]
        if s_pos <= 0.0 + eps:
            # Set starting velocity to 10.0
            px, py = tpos.get_position().as_tuple()
            x_bound = [px, px]
            y_bound = [py, py]
            vx_bound = [10.0, 10.0]
            vy_bound = [0.0, 0.0]
            yaw_bound = [tpos.get_orientation(), tpos.get_orientation()]
            omega_bound = [0.0, 0.0]
            xi_bound = [0.0, 0.0]
            nu_bound = [0.0, 0.0]
        rows = [x_bound, y_bound, vx_bound, vy_bound, yaw_bound, omega_bound]
        rows.append(xi_bound)
        rows.append(nu_bound)
        bound_mtx = np.array(rows)
        return bound_mtx

    def get_init_values(self, s_dist: float):
        tpoint = self.track.get_point(s_dist)
        px_init, py_init = tpoint.get_position().as_tuple()
        vx_init = 10.0
        vy_init = 0.0
        yaw_init = tpoint.get_orientation()
        omega_init = 0.0
        xi_init = 0.0
        nu_init = 0.0
        state_vals = [px_init, py_init, vx_init, vy_init, yaw_init]
        state_vals.extend([omega_init, xi_init, nu_init])
        ctrl_vals = [0.0, 0.0] 
        state, ctrl = np.array(state_vals), np.array(ctrl_vals)
        return state, ctrl

class VehicleTrajectoryProblem(CollocationProblem):
    
    def __init__(self, domain: VehicleDomain, track: Track, save_folder):
        super().__init__(VehicleDyanmics(track), domain, save_folder) 
        self.track = track

    def objective_th(self, xvec_th: th.Tensor):
        """
        Evaluate the objective function
        """
        nodes_pos_list = [node.get_pos() for node in self.domain.nodes]
        dev, data_type = xvec_th.device, xvec_th.dtype
        state_imtx = self.domain.state_imtx
        state_mtx = xvec_th[state_imtx.ravel()].reshape(state_imtx.shape)
        vx = state_mtx[:, self.model.vx.get_idx()]
        vy = state_mtx[:, self.model.vy.get_idx()]
        xi = state_mtx[:, self.model.xi_angle.get_idx()]
        nu = state_mtx[:, self.model.nu_dist.get_idx()]
        s_dists = th.tensor(nodes_pos_list, device=dev, dtype=data_type)
        track_curves = self.track.get_curvatures_th(s_dists)
        delta_s = (vx*th.cos(xi) - vy*th.sin(xi)) / (1.0 - nu * track_curves)
        dt_ds = 1.0 / delta_s
        total_time = th.trapz(dt_ds, s_dists)
        return total_time

def example_track() -> Track:
    tdict = {}
    tdict["start position"] = [0.0, 0.0]
    tdict["angle"] = 0.0
    tdict["width"] = 25.0
    telems = []
    telems.append({"type": "straight", "length": 200})
    telems.append({"type": "curve", "radius": 200, "angle distance": 90})
    telems.append({"type": "straight", "length": 200})
    tdict["elements"] = telems
    new_track = Track(tdict)
    return new_track

def main():
    save_folder = './solutions/vehicle_trajectory/' 
    track = example_track()
    layout = DomainLayout()
    num = 20
    for k in range(num):
        width = track.get_length() / num
        layout.add_segment(width, 'trapz', 1)
    track_dict = {}
    # Create a track dictionary here!
    model = VehicleDyanmics(track)
    domain = VehicleDomain(model, layout, track)
    problem = VehicleTrajectoryProblem(domain, track, save_folder)
    # Enable cuda computation, if it is available.
    if th.cuda.is_available():
        problem.device = 'cuda:0'
    else:
        problem.device = 'cpu'
    x0 = problem.get_init_values()
    problem.save_freq = 5
    res = problem.solve_problem(x0, max_iter=1000)
    # Plot the solution
    problem.save_solution(res.x, 1)
    problem.plot_solution(res.x)

if __name__ == "__main__":
    main()
