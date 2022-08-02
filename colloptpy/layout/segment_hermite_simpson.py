from ..dynmodels.dynamic_model import DynamicModel
from .node import Node
from .segment import Segment
from typing import List
import numpy as np
import torch as th


mtx = np.ndarray


class SegmentHermiteSimpson(Segment):
    '''
    Hermite simpson quadrature segment.
    There are two variants of the Hermite simpson rule, known as compressed
    and separated form. We implement the separated form here. For reference
    please consider
        Practical Methods for Optimal Control and Estimation Using Nonlinear Programming
    by J.T. Betts
    '''

    def __init__(self, nodes: List[Node], model: DynamicModel):
        super().__init__(nodes, model)
        # Check the number of nodes
        if len(nodes) != 3:
            raise ValueError('3 nodes required for hermite simpson collocation segment')
        # Check that nodes have the correct position
        mid_pos = 0.5*(self.nodes[2].get_pos() + self.nodes[0].get_pos())
        mid_diff = abs(self.nodes[1].get_pos() - mid_pos)
        if mid_diff > 1e-8:
            raise ValueError('Incorrect node positioning for HermiteSimpson collocation')

    def get_num_constraints(self) -> int:
        num_states = self.model.get_num_states()
        return num_states * 2

    def eval_constraints(self, x_vec: mtx, u_vec: mtx):
        '''
        Eval the HS conditions
        '''
        width = self.width
        x_mtx, u_mtx, f_mtx = self.forward_model(x_vec, u_vec)
        x0, x1, x2 = x_mtx[0, :], x_mtx[1, :], x_mtx[2, :]
        f0, f1, f2 = f_mtx[0, :], f_mtx[1, :], f_mtx[2, :]
        x_diff1 = x2 - x0 - (self.width/6.0)*(f0 + 4.0*f1 + f2)
        x_diff2 = x1 - 0.5*(x0+x2) - (self.width/8.0)*(f0 - f2)
        constr_err = th.cat([x_diff1, x_diff2])
        return constr_err
