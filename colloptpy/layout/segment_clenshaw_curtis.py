from ..dynmodels.dynamic_model import DynamicModel
from .node import Node
from .segment import Segment
from typing import List
import numpy as np
import torch as th


mtx = np.ndarray


class SegmentClenshawCurtis(Segment):
    '''
    Clenshaw--Curtis quadrature segment.
    '''

    def __init__(self, nodes: List[Node], model: DynamicModel):
        super().__init__(nodes, model)
        # Check that the nodes are positioned at chebyshev points
        num = len(nodes)
        self.num_nodes = num
        pts = np.polynomial.chebyshev.chebpts2(num)
        pts2 = nodes[0].get_pos() + 0.5*self.width*(pts+1.0)
        pos = np.array([n.get_pos() for n in nodes])
        diff = np.linalg.norm(pts2 - pos)
        if diff > 1e-8:
            raise ValueError('Incorrect node positioning for Clenshaw--Curtis collocation')
        # Setup vandermonde matrix:
        vmtx = np.polynomial.chebyshev.chebvander(pts, num-1)
        self.vmtxi = np.linalg.inv(vmtx)
        # Setup quadrature matricies
        vmtx2 = np.polynomial.chebyshev.chebvander(pts, num)
        cols = []
        for k in range(num):
            coefs = np.zeros(num)
            coefs[k] = 1.0
            coefs_int = np.polynomial.chebyshev.chebint(coefs, 1)
            new_col = vmtx2.dot(coefs_int)
            cols.append(new_col)
        self.qmtx = np.array(cols).T

    def get_num_constraints(self) -> int:
        num_states = self.model.get_num_states()
        return num_states * (self.num_nodes-1)

    def eval_constraints(self, x_vec: mtx, u_vec: mtx):
        '''
        Eval the HS conditions
        '''
        x_mtx, _, f_mtx = self.forward_model(x_vec, u_vec)
        # Integrate the state derivatives
        vi_mtx = th.tensor(self.vmtxi, device=x_vec.device, dtype=x_vec.dtype)
        q_mtx = th.tensor(self.qmtx, device=x_vec.device, dtype=x_vec.dtype)
        c_facs = th.matmul(vi_mtx, f_mtx)
        x_quad = th.matmul(q_mtx, c_facs)
        # Compute the errors at nodes 1, ..., self.num_nodes-1
        parts = []
        for nidx in range(1, self.num_nodes):
            diff = x_mtx[nidx, :] - (x_mtx[0, :] + 0.5*self.width*x_quad[nidx, :])
            parts.append(diff)
        constr_err = th.cat(parts)
        return constr_err
