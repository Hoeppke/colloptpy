import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from colloptpy.copt.collocation_problem import CollocationProblem
from colloptpy.layout.domain_layout import DomainLayout
from colloptpy.layout.domain import Domain
from colloptpy.dynmodels.dynamic_model import DynamicModel


class PendulumDynamics(DynamicModel):

    def __init__(self):
        super().__init__()
        # Add the required variables
        self.rod_length = 1.0
        self.cart_mass = 1.0
        self.ball_mass = 0.1
        self.u_max = 1.0
        self.const_g = 9.81
        self.pos = self.add_state('x')
        self.vel = self.add_state('v')
        self.theta = self.add_state('theta')
        self.omega = self.add_state('omega')
        self.force = self.add_control('force')
        self.force.set_bounds(-10.0, +10.0)

    def get_main_variable(self) -> str:
        """
        We override the main variable explicitly. 
        """
        return 'time'

    def forward(self, x_mtx: th.tensor, u_mtx: th.tensor, t_mtx: th.tensor):
        """
        Evaluate the cart-pole dynamics:
            http://www.matthewpeterkelly.com/tutorials/cartPole/index.html
        """
        num_rows: int = x_mtx.shape[0]
        mtx = th.zeros((num_rows, 2, 2), dtype=x_mtx.dtype, device=x_mtx.device)
        pos = x_mtx[:, self.pos.get_idx()]
        vel = x_mtx[:, self.vel.get_idx()]
        tet = x_mtx[:, self.theta.get_idx()]
        omg = x_mtx[:, self.omega.get_idx()]
        force = u_mtx[:, self.force.get_idx()]
        mtx[:, 0, 0] = th.cos(tet)
        mtx[:, 0, 1] = self.rod_length
        mtx[:, 1, 0] = self.ball_mass + self.cart_mass
        mtx[:, 1, 1] = self.ball_mass * self.rod_length * th.cos(tet)
        det = (mtx[:, 0, 0] * mtx[:, 1, 1]) - (mtx[:, 0, 1] * mtx[:, 1, 0])
        mtx_i = th.zeros((num_rows, 2, 2), dtype=x_mtx.dtype, device=x_mtx.device)
        mtx_i[:, 0, 0] = mtx[:, 1, 1] / det
        mtx_i[:, 1, 1] = mtx[:, 0, 0] / det
        mtx_i[:, 0, 1] = -1.0*mtx[:, 0, 1] / det
        mtx_i[:, 1, 0] = -1.0*mtx[:, 1, 0] / det
        # Compute the RHS
        rhs = th.zeros((num_rows, 2), dtype=x_mtx.dtype, device=x_mtx.device)
        rhs_parts = []
        rhs_parts.append(-self.const_g * th.sin(tet))
        rhs_parts.append(force + self.ball_mass * self.rod_length * omg * omg * th.sin(tet))
        rhs = th.stack(rhs_parts, axis=1)
        sol_vecs = []
        for j in range(mtx_i.shape[0]):
            new_sol = th.matmul(mtx_i[j, :, :], rhs[j, :]) 
            sol_vecs.append(new_sol)
        sol_vec = th.stack(sol_vecs, axis=0)
        x_accel = sol_vec[:, 0]
        tet_accel = sol_vec[:, 1]
        out_vals = [vel, x_accel, omg, tet_accel]
        res = th.stack(out_vals, axis=1)
        return res

class CartPoleDomain(Domain):

    def __init__(self, model: PendulumDynamics, layout: DomainLayout):
        """
        Initialise a cartpole domain 
        """
        super().__init__(model, layout)
        self.can_plot = True

    def get_state_bounds(self, time: float):
        """
        Get the state bounds
        """
        eps = 1e-8
        if time <= 0.0 + eps:
            x_bound = [0.0, 0.0]
            v_bound = [0.0, 0.0]
            tet_bound = [0.0, 0.0]
            omg_bound = [0.0, 0.0]
        elif time >= 1.0 - eps:
            x_bound = [0.0, 0.0]
            v_bound = [0.0, 0.0]
            tet_bound = [np.pi, np.pi]
            omg_bound = [0.0, 0.0]
        else: 
            x_bound = [-50.0, +50.0]
            v_bound = [-50.0, +50.0]
            tet_bound = [-50.0, +50.0]
            omg_bound = [-50.0, +50.0]
        rows = [x_bound, v_bound, tet_bound, omg_bound]
        bound_mtx = np.array(rows)
        return bound_mtx

    def get_init_values(self, time: float):
        eps = 1e-8
        x_val = 0.0
        v_val = 0.0
        tet_val = time*np.pi
        omg_val = 0.0
        state_vals = [x_val, v_val, tet_val, omg_val]
        ctrl_vals = [0.0] 
        state, ctrl = np.array(state_vals), np.array(ctrl_vals)
        return state, ctrl

    def plot_state(self, state: np.ndarray, save_path: str):
        """
        Make a plot of the cartpole state
        """
        pos = state[self.model.pos.get_idx()]
        tet = state[self.model.theta.get_idx()]
        width = 1.0
        height = 1.0
        px, py = pos-0.5*width, -0.5*height
        fig = plt.figure(figsize=(6, 6)) 
        ax = fig.add_subplot(1, 1, 1)
        ax.add_patch(Rectangle((px, py), width, height, fc='none', ec='black', lw=2.0))
        # Add the circle
        px_c = pos + self.model.rod_length*np.cos(tet)
        py_c = 0.0 + self.model.rod_length*np.sin(tet)
        ax.add_patch(Circle((px_c, py_c), 0.1))
        # Add the connection_line
        ax.plot([pos, px_c], [0.0, py_c])
        plt.xlim(-5.0, +5.0)
        plt.ylim(-5.0, +5.0)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

class CartPoleProblem(CollocationProblem):
    
    def __init__(self, layout: DomainLayout, save_folder):
        super().__init__(PendulumDynamics(), layout, save_folder) 

    def objective_th(self, xvec_th: th.Tensor):
        """
        Evaluate the objective function
        """
        ctrl_imtx = self.domain.ctrl_imtx
        ctrl_mtx = xvec_th[ctrl_imtx.ravel()].reshape(ctrl_imtx.shape)
        force_vec = ctrl_mtx[:, self.model.force.get_idx()]
        pos_vec = self.domain.get_node_pos_vec()
        pos_th = th.tensor(pos_vec, device=self.device)
        force_quad = th.trapz(force_vec*force_vec, pos_th)
        return force_quad

def main():
    save_folder = './solutions/cartpole/' 
    layout = DomainLayout()
    num = 10
    for _ in range(num):
        width = 1.0 / num
        layout.add_segment(width, 'trapz', 1)
    model = PendulumDynamics()
    domain = CartPoleDomain(model, layout)
    problem = CartPoleProblem(domain, save_folder)
    # Enable cuda computation, if it is available.
    if th.cuda.is_available():
        problem.device = 'cuda:0'
    else:
        problem.device = 'cpu'
    x0 = problem.get_init_values()
    problem.save_freq = 10
    res = problem.solve_problem(x0, max_iter=1000)
    # Plot the solution
    problem.save_solution(res.x, 1)
    problem.plot_solution(res.x)

if __name__ == "__main__":
    main()
