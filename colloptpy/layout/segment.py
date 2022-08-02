from ..dynmodels.dynamic_model import DynamicModel
from .node import Node
from abc import ABC, abstractmethod
from functorch import jacrev
from typing import List
import numpy as np
import scipy.sparse as sparse
import torch as th

mtx = th.tensor

class Segment(ABC):
    
    def __init__(self, nodes: List[Node], model: DynamicModel):
        self.nodes = nodes
        self.model = model
        self.width = nodes[-1].get_pos() - nodes[0].get_pos()
        self.nidxs = [node.get_idx() for node in self.nodes]
        self.npos_vec = np.array([node.get_pos() for node in self.nodes])
        self.jac_x = jacrev(self.eval_constraints, argnums=0)
        self.jac_u = jacrev(self.eval_constraints, argnums=1)
        self._setup_indices()
        self.num_total_vars: int = 0

    def _setup_indices(self):
        state_idxs, ctrl_idxs = [], []
        for node in self.nodes:
            state_idxs.append(node.state_idxs)
            if node.has_ctrl:
                ctrl_idxs.append(node.ctrl_idxs)
        self.state_idxs = np.array(state_idxs, dtype=np.int64)
        self.ctrl_idxs = np.array(ctrl_idxs, dtype=np.int64)

    def set_num_total_vars(self, num_vars: int):
        self.num_total_vars = num_vars
        
    @abstractmethod
    def get_num_constraints(self) -> int:
        return 0

    @abstractmethod
    def eval_constraints(self, x_vec: mtx, u_vec: mtx) -> mtx:
        '''
        Note: All parameters have to be provided in vectorised form.
        '''
        return 0 

    def get_num_ctrl_nodes(self) -> int:
        res: int = 0
        for node in self.nodes:
            if node.has_ctrl():
                res += 1
        return res

    def get_ctrl_nodes(self) -> list[int]:
        """
        Get a list of all the nodes which have control variables
        """
        out_list = []
        for idx, node in enumerate(self.nodes):
            if node.has_ctrl():
                out_list.append(idx)
        return out_list

    def forward_model(self, x_vec: mtx, u_vec: mtx):
        """
        Compute the forward of the model while transforming the parameters
        into a more convenient matrix format.
        """
        time_vec = th.tensor(self.npos_vec, dtype=x_vec.dtype, device=x_vec.device)
        num_x_nodes = len(self.nodes)
        num_u_nodes = self.get_num_ctrl_nodes()
        num_states = self.model.get_num_states()
        num_ctrls = self.model.get_num_controls()
        ctrl_node_idxs = self.get_ctrl_nodes()
        x_mtx = th.reshape(x_vec, (num_x_nodes, num_states))
        u_mtx = th.reshape(u_vec, (num_u_nodes, num_ctrls))
        x_mtx_forward = x_mtx[ctrl_node_idxs, :]
        f_mtx = self.model.forward(x_mtx_forward, u_mtx, time_vec)
        return x_mtx, u_mtx, f_mtx

    def estimate_state_err(self, x_vec: mtx, u_vec: mtx) -> mtx:
        """
        estimate the state errors on this segment.
        """
        pass

    def eval_jacobian(self, x_vec: mtx, u_vec: mtx) -> sparse.csr_matrix:
        '''
        Evaluate the constraint jacobian, by variables on this segment
        '''
        my_jx = self.jac_x(x_vec, u_vec).detach().cpu().numpy()
        my_ju = self.jac_u(x_vec, u_vec).detach().cpu().numpy()
        # Assemble the Jacobian matrix:
        num_rows = my_jx.shape[0]
        state_idxs = self.state_idxs.ravel()
        ctrl_idxs = self.ctrl_idxs.ravel()
        # A little inefficient with double for loop
        data = []
        rows = []
        cols = []
        for ridx in range(num_rows):
            # State jacobian
            rows.extend([ridx]*len(state_idxs))
            cols.extend(state_idxs)
            data.extend(list(my_jx[ridx, :]))
            # Ctrl jacobian
            rows.extend([ridx]*len(ctrl_idxs))
            cols.extend(ctrl_idxs)
            data.extend(list(my_ju[ridx, :]))
        rows = np.array(rows, dtype=np.int64)
        cols = np.array(cols, dtype=np.int64)
        data = np.array(data, dtype=np.float64)
        jac_shape = (num_rows, self.num_total_vars)
        jac_slice = sparse.coo_matrix((data, (rows, cols)), jac_shape)
        jac_slice = jac_slice.tocsr()
        return jac_slice
