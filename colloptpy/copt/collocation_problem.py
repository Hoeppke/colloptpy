from ..storage.saved_solution import SavedSolution
from ..dynmodels.dynamic_model import DynamicModel
from ..layout.domain import Domain
import datetime as dt
from pathlib import Path
import pandas as pd
import numpy as np
import torch as th
from scipy import sparse
from scipy import optimize
from scipy import interpolate
from abc import ABC, abstractmethod
from typing import Union
import os


class CollocationProblem(ABC):
    '''
    More restrictive version of the CollocationDomain class.
    Only solve the full problem using the new torch based dynamics class
    '''

    def __init__(self, model: DynamicModel, domain: Domain, save_dir: str):
        self.model: DynamicModel = model
        self.domain: Domain = domain
        self.save_dir: str = save_dir
        self.save_freq: int = 1
        self._save_format: str = 'csv'
        self.device = 'cpu'
        #
        self.verbose: bool = False
        self.nit: int = 0
        # Setup paths
        self.save_path = Path(self.save_dir)
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.sol_times = []

    @abstractmethod
    def objective_th(self, xvec: th.tensor) -> th.tensor:
        return None

    def objective(self, xvec: np.ndarray) -> float:
        xvec_th = th.tensor(xvec, requires_grad=False, device=self.device)
        obj = self.objective_th(xvec_th)
        obj = obj.detach().cpu().numpy()
        return obj

    def objective_grad(self, xvec: np.ndarray) -> np.ndarray:
        xvec_th = th.tensor(xvec, requires_grad=True, device=self.device)
        obj = self.objective_th(xvec_th)
        obj.backward()
        obj_grad = xvec_th.grad.detach().cpu().numpy()
        return obj_grad

    def constraints(self, xvec: np.ndarray) -> np.ndarray:
        xvec_th = th.tensor(xvec, requires_grad=False, device=self.device)
        cvec = self.domain.eval_constraints(xvec_th)
        cvec = cvec.detach().cpu().numpy()
        return cvec

    def cviol_norm(self, cviol: np.ndarray):
        num_rows = int(len(cviol) / self.model.get_num_states())
        num_states = self.model.get_num_states()
        cviol_mtx = cviol.reshape(num_rows, num_states)
        seg_lengths = [seg.width for seg in self.domain.segments]
        # Integrate each variable
        var_norms = [np.dot(cviol_mtx[:, i]**2.0, seg_lengths)**0.5 for i in range(num_states)]
        cviol_norm = np.linalg.norm(var_norms)
        return cviol_norm

    def constraints_jac(self, xvec: np.ndarray) -> sparse.csr_matrix:
        xvec_th = th.tensor(xvec, requires_grad=True, device=self.device)
        cjac = self.domain.eval_constraint_jac(xvec_th)
        return cjac

    def get_num_constraints(self) -> int:
        return self.domain.get_num_constraints()

    def record_solution(self, iter: int, x_vec: np.ndarray):
        new_time = dt.datetime.now()
        tflt = new_time.timestamp()
        tstr = new_time.strftime('%Y-%m-%d %H:%M:%S.%f')
        obj = self.objective(x_vec)
        cviol_l1 = np.linalg.norm(self.constraints(x_vec), ord=1)
        cviol_l2 = np.linalg.norm(self.constraints(x_vec), ord=2)
        cviol_linf = np.linalg.norm(self.constraints(x_vec), ord=np.inf)
        res_dict = {}
        res_dict['iter'] = iter
        res_dict['datetime'] = tstr
        res_dict['timestamp'] = tflt
        res_dict['objective'] = obj
        res_dict['constraint_violation_l1'] = cviol_l1
        res_dict['constraint_violation_l2'] = cviol_l2
        res_dict['constraint_violation_linf'] = cviol_linf
        self.sol_times.append(res_dict)

    def solve_problem(self, x0, max_iter=50):
        """
        Solve the given problem
        """
        # build the bounds
        bounds = self.domain.get_bound_vectors()
        # Continue here to build, and then solve, a given problem.
        num_constr = self.domain.get_num_constraints()
        clb, cub = np.zeros(num_constr), np.zeros(num_constr)
        constr = optimize.NonlinearConstraint(self.constraints, clb, cub, jac=self.constraints_jac)
        optimiser_options = {'maxiter': max_iter, 'disp': True, 'verbose': 2, 'xtol': 1e-10}
        opts = {}
        opts['method'] = 'trust-constr'
        opts['bounds'] = bounds
        opts['constraints'] = [constr]
        opts['jac'] = self.objective_grad
        opts['options'] = optimiser_options
        opts['callback'] = self.save_solution_callback
        self.record_solution(0, x0)
        res = optimize.minimize(self.objective, x0, **opts)
        return res

    def get_init_values(self):
        """
        Collect all initial values from the domain and construct a heuristic guess for
        an initial vector.
        """
        num_vars = self.domain.get_num_problem_vars()
        init_vec = np.zeros(num_vars, dtype=np.float64)
        for node in self.domain.nodes:
            node_pos = node.get_pos()
            states, ctrls = self.domain.get_init_values(node_pos)
            init_vec[node.get_state_idxs()] = states
            if node.has_ctrl():
                init_vec[node.get_ctrl_idxs()] = ctrls
        return init_vec

    def plot_solution(self, x_vec: np.ndarray):
        """
        Visualise the current solution
        """
        if not self.domain.can_plot:
            raise ValueError('The domain does not supplot visualisation')
        # Plot solution
        x_mtx = x_vec[self.domain.state_imtx]
        fig_dir = os.path.join(self.save_dir, 'plots')
        fig_dir_path = Path(fig_dir)
        fig_dir_path.mkdir(parents=True, exist_ok=True)
        zmax = int(np.log10(x_mtx.shape[0]) + 1)
        for ridx in range(x_mtx.shape[0]):
            print('\rplotting frame: {}'.format(ridx), end='')
            fname = 'plot_{}.png'.format(str(ridx).zfill(zmax))
            fpath = os.path.join(fig_dir, fname)
            self.domain.plot_state(x_mtx[ridx, :], fpath) 
        print('\tPlotted solution')

    def save_solution_callback(self, xvec: np.ndarray, sol_state):
        """
        Callback function to be used for saving intermediary solutions
        """
        nit = sol_state.nit
        self.record_solution(nit, xvec)
        if nit % self.save_freq == 0:
            self.save_solution(xvec, nit)
        # Save the solution times
        times_fname = os.path.join(self.save_path, 'sol_times.csv')
        data_pd = pd.DataFrame(self.sol_times)
        data_pd.to_csv(times_fname, index=False)

    def save_solution(self, xvec: np.ndarray, nit: int):
        """
        Save the current solution
        """
        num_nodes = len(self.domain.nodes)
        vars = self.model.get_all_variables()
        var_names = [v.get_name() for v in vars]
        var_names.append(self.model.get_main_variable())
        var_names = ['node_idx'] + var_names
        sol = SavedSolution(var_names, num_nodes)
        pos_vec = self.domain.get_node_pos_vec()
        ctrl_pos_vec = pos_vec[self.domain.get_ctrl_node_idxs()]
        sol.set_values(self.model.get_main_variable(), pos_vec)
        sol.set_values('node_idx', np.arange(num_nodes))
        # Save the state variables
        for var in self.model.states:
            # State variables
            vals = xvec[self.domain.state_imtx[:, var.get_idx()]]
            sol.set_values(var.get_name(), vals) 
        # Need to potentially interpolate the control variables
        for var in self.model.ctrls:
            # Control values
            vals = xvec[self.domain.ctrl_imtx[:, var.get_idx()]]
            interp_func = interpolate.interp1d(ctrl_pos_vec, vals)
            vals_interp = interp_func(pos_vec) 
            sol.set_values(var.get_name(), vals_interp)
        fname = 'sol_{}.{}'.format(str(nit).zfill(5), self._save_format)
        fpath = os.path.join(self.save_path, fname)
        dframe: pd.DataFrame = sol.to_dataframe()
        dframe.set_index('node_idx')
        dframe.to_csv(fpath, index=False) 

    def load_solution(self, sol: Union[SavedSolution, str]): 
        """
        Load a saved solution
        """
        if isinstance(sol, str):
            my_sol: SavedSolution = SavedSolution.load_csv(sol)
        else:
            my_sol: SavedSolution = sol
        # Interpolate the solution onto the required grid
        node_pos_vec = self.domain.get_node_pos_vec()
        my_sol = my_sol.interpolate(self.model.get_main_variable(), node_pos_vec) 
        # Loaded saved solution
        xvec = np.zeros(self.domain.get_num_problem_vars(), dtype=np.float64)
        # Iterate over state variables
        for var in self.model.states:
            # State variables
            vname = var.get_name()
            vals = my_sol.get_values(vname)
            idxs = self.domain.state_imtx[:, var.get_idx()]
            xvec[idxs] = vals
        # Iterate over control variables
        ctrl_nodes = self.domain.get_ctrl_node_idxs()
        for var in self.model.ctrls:
            # Control values
            vals = my_sol.get_values(var.get_name())
            valid_vals = vals[ctrl_nodes]
            # select the valid control idxs
            idxs = self.domain.ctrl_imtx[:, var.get_idx()]
            xvec[idxs] = vals
        return xvec 
