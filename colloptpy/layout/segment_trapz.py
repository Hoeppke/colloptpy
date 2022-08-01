from ..dynmodels.dynamic_model import DynamicModel
from .node import Node
from .segment import Segment
from typing import List
import numpy as np


mtx = np.ndarray


class SegmentTrapz(Segment):
    '''
    Trapz quadrature segment, implemented in python
    '''

    def __init__(self, nodes: List[Node], model: DynamicModel):
        super().__init__(nodes, model)
        if len(nodes) != 2:
            raise ValueError('Wrong number of constraints for Trapz quadrature')

    def get_num_constraints(self) -> int:
        num_states = self.model.get_num_states()
        return num_states

    def eval_constraints(self, x_vec: mtx, u_vec: mtx):
        '''
        Eval the trapz constraint
        '''
        x_mtx, u_mtx, f_mtx = self.forward_model(x_vec, u_vec)
        x0, x1 = x_mtx[0, :], x_mtx[1, :]
        f0, f1 = f_mtx[0, :], f_mtx[1, :]
        x_diff = x1 - x0
        x_quad = 0.5*self.width*(f0 + f1)
        return x_diff - x_quad
