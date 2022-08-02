from .domain_layout import DomainLayout
from ..dynmodels.dynamic_model import DynamicModel
from ..storage.saved_solution import SavedSolution
from .segment_trapz import SegmentTrapz
from .segment_hermite_simpson import SegmentHermiteSimpson
from .segment_clenshaw_curtis import SegmentClenshawCurtis
from .node import Node
from .segment import Segment
from abc import ABC, abstractmethod
import numpy as np
import torch as th
from scipy import sparse


class Domain():

    def __init__(self, model: DynamicModel, layout: DomainLayout):
        self.model = model
        self.layout = layout
        # Setup the domain
        self._next_node_idx: int = 0
        self._next_variable_idx: int = 0
        self._node_pos_vec = None
        self.nodes: list[Node] = []
        self.segments: list[Segment] = []
        # initialise the segments
        self._init_segments()
        # Initialise the index matricies
        self._init_indices()
        # Initialise the variable bounds
        self._init_bounds()
        # Other functionality
        self.can_plot: bool = False

    def _init_segments(self):
        # construct the segments
        for seg in self.layout.segment_dicts:
            self._build_segment(seg)
        # Send number of variables to segments
        for seg in self.segments:
            seg.set_num_total_vars(self._next_variable_idx)

    def _init_indices(self):
        '''
        Initialise the index matricies
        '''
        state_idxs = []
        ctrl_idxs = []
        for node in self.nodes:
            state_idxs.append(node.state_idxs)
            if node.has_ctrl:
                ctrl_idxs.append(node.ctrl_idxs)
        self.state_imtx = np.array(state_idxs, dtype=np.int64)
        self.ctrl_imtx = np.array(ctrl_idxs, dtype=np.int64)

    def _init_bounds(self):
        all_bounds = np.ndarray((self._next_variable_idx, 2))
        for node in self.nodes:
            state_bounds = self.get_state_bounds(node.get_pos())
            all_bounds[node.state_idxs, :] = state_bounds
            if node.has_ctrl:
                ctrl_bounds = self.get_ctrl_bounds(node.get_pos())
                all_bounds[node.ctrl_idxs, :] = ctrl_bounds
        self.lb_vec = all_bounds[:, 0]
        self.ub_vec = all_bounds[:, 1]

    def get_bound_vectors(self) -> list[tuple[float, float]]:
        """
        Return the bound vectors.
        """
        bounds = list(zip(self.lb_vec, self.ub_vec))
        return bounds

    def get_node_pos_vec(self) -> np.ndarray:
        if self._node_pos_vec is None:
            pos_vec = np.zeros(len(self.nodes), dtype=np.float64)
            for i, node in enumerate(self.nodes):
                pos_vec[i] = node.get_pos()
            self._node_pos_vec = pos_vec
        return self._node_pos_vec

    def load_init_solution(self, sol: SavedSolution, main_var: str=None) -> np.ndarray:
        """
        Load the variables from a saved solution.
        """
        pvec = np.zeros(self._next_variable_idx)
        xarr = np.ndarray([node.get_pos() for node in self.nodes])
        if main_var is None:
            main_var: str = self.model.get_main_variable()
        for var in self.model.states:
            var_name = var.get_name()
            xvar_name = main_var
            values = sol.get_values_interp(var_name, xvar_name, xarr)
            idxs = self.state_imtx[:, var.get_idx()]
            pvec[idxs] = values
        for var in self.model.ctrls:
            var_name = var.get_name()
            xvar_name = main_var
            values = sol.get_values_interp(var_name, xvar_name, xarr)
            idxs = self.ctrl_idxs[:, var.get_idx()]
            pvec[idxs] = values
        return pvec

    def get_num_problem_vars(self) -> int:
        return self._next_variable_idx

    @abstractmethod
    def get_state_bounds(self, pos: float) -> np.ndarray:
        """
        Construct state bounds
        """
        return None

    @abstractmethod
    def get_init_values(self, pos: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct an initial value if not solution is provided.
        """
        return None

    def get_ctrl_bounds(self, pos: float) -> np.ndarray:
        '''
        Override this method if custom control bounds are required
        '''
        bounds = []
        for ctrl_var in self.model.ctrls:
            lb, ub = ctrl_var.get_bounds()
            bounds.append([lb, ub])
        bounds_mtx = np.array(bounds)
        return bounds_mtx

    def get_ctrl_node_idxs(self) -> np.ndarray:
        """
        Get a list of node indices where we have controls
        """
        out_list = []
        for node in self.nodes:
            if node.has_ctrl():
                out_list.append(node.get_idx())
        return np.array(out_list, dtype=np.int64)

    def _make_node(self, pos: float, has_ctrl: bool):
        new_idx = self._next_node_idx
        new_node = Node(new_idx, pos, has_ctrl)
        new_node.set_indices(self._next_variable_idx, self.model)
        self._next_node_idx += 1
        self._next_variable_idx = new_node.get_last_vidx() + 1
        self.nodes.append(new_node)
        self._node_pos_vec = None
        return new_node

    def get_num_constraints(self) -> int:
        seg_constr = [seg.get_num_constraints() for seg in self.segments]
        return sum(seg_constr)

    def eval_constraints(self, xvec: th.tensor) -> th.tensor:
        constr_parts = []
        for seg in self.segments: 
            xidxs, uidxs = seg.state_idxs, seg.ctrl_idxs
            xidxs = xidxs.ravel()
            uidxs = uidxs.ravel()
            state_vec = xvec[xidxs]
            ctrl_vec = xvec[uidxs]
            constr_vals = seg.eval_constraints(state_vec, ctrl_vec)
            constr_parts.append(constr_vals)
        constr_vec = th.cat(constr_parts)
        return constr_vec

    def eval_constraint_jac(self, xvec: th.tensor) -> sparse.csr_matrix:
        jac_parts = []
        for seg in self.segments:
            xidxs, uidxs = seg.state_idxs, seg.ctrl_idxs
            xidxs = xidxs.ravel()
            uidxs = uidxs.ravel()
            state_vec = xvec[xidxs]
            ctrl_vec = xvec[uidxs]
            jac_part = seg.eval_jacobian(state_vec, ctrl_vec)
            jac_parts.append(jac_part)
        jac_mtx = sparse.vstack(jac_parts)
        return jac_mtx
        
    def _build_segment(self, seg):
        width, method, order = seg['width'], seg['method'], seg['order']
        method: str = str(method)
        order: int = int(order)
        width: float = float(width)
        if method == 'trapz':
            if order == 1:
                self._build_segment_trapz(width)
            else:
                raise ValueError('Trapz segment has to be constructed with order 1')
        elif method == 'hermite-simpson':
            if order == 2:
                self._build_segment_hsimpson(width)
            else:
                raise ValueError('Hermite-Simpson requires quadratic control interpolation (order=2)')
        elif method == 'chebyshev' or method == 'clenshaw-curtis':
            self._build_segment_cheb(width, order)

    def _build_segment_trapz(self, width: float):
        '''
        Add a trapz quadrature segment
        '''
        if len(self.nodes) == 0:
            # No nodes here yet
            n0 = self._make_node(0.0, True)
        else:
            n0 = self.nodes[-1]
        last_pos = n0.get_pos()
        new_pos = last_pos + width
        n1 = self._make_node(new_pos, True)
        nodes = [n0, n1]
        new_seg = SegmentTrapz(nodes, self.model)
        self.segments.append(new_seg)

    def _build_segment_hsimpson(self, width: float):
        '''
        Add a Hermite-Simpson segment
        '''
        if len(self.nodes) == 0:
            # No nodes here yet
            n0 = self._make_node(0.0, True)
        else:
            n0 = self.nodes[-1]
        last_pos = n0.get_pos()
        n1 = self._make_node(last_pos+0.5*width, True)
        n2 = self._make_node(last_pos+1.0*width, True)
        nodes = [n0, n1, n2]
        new_seg = SegmentHermiteSimpson(nodes, self.model)
        self.segments.append(new_seg)

    def _build_segment_cheb(self, width: float, order: int):
        """
        Add a Hermite-Simpson segment
        """
        if len(self.nodes) == 0:
            # No nodes here yet
            n0 = self._make_node(0.0, True)
        else:
            n0 = self.nodes[-1]
        last_pos = n0.get_pos()
        cpts = np.polynomial.chebyshev.chebpts2(order+1)
        cpts = last_pos + 0.5*width*(cpts+1.0)
        coll_nodes = [n0]
        for new_pos in cpts[1:]:
            new_node = self._make_node(new_pos, True)
            coll_nodes.append(new_node)
        new_seg = SegmentClenshawCurtis(coll_nodes, self.model)
        self.segments.append(new_seg)

    def plot_state(self, state: np.ndarray, save_path: str):
        pass
