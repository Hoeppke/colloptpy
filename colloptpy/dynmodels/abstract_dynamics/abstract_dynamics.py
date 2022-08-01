from abc import ABC, abstractmethod
from ..dynamic_model import DynamicModel
from typing import List, Callable, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import numba 

Vec = np.ndarray


class AbstractDynamics(ABC):

    def __init__(self, model: DynamicModel):
        '''
        Updated abstract dynamics relying on pytorch tensors
        for automatic differentiation
        '''
        self.states: List[DynamicsVariable] = []
        self.ctrls: List[ControlVariable] = []
        self.params: List[ParameterVariable] = []
        self.state_idxs: Dict[str, int] = {}
        self.ctrls_idxs: Dict[str, int] = {}
        self.params_idxs: Dict[str, int] = {}
        self.num_states: int = 0
        self.num_ctrls: int = 0
        self.num_params: int = 0
        self._dynfunc: Callable = None
        self._dynfunc_jit: Callable = None
        self._is_initialized: bool = False
        self.init_dynamics()

    @abstractmethod
    def get_dynamics(self):
        pass 

    def is_ready(self) -> bool:
        return self._is_initialized

    def get_state_vars(self) -> List[DynamicsVariable]:
        return self.states

    def get_state_var(self, name: str) -> DynamicsVariable:
        idx_arr = self.state_idxs
        var_arr = self.states
        if name in idx_arr:
            var_idx = idx_arr[name]
            return var_arr[var_idx]
        return None

    def get_param_vars(self) -> List[ParameterVariable]:
        return self.params

    def get_param_var(self, name: str) -> ParameterVariable:
        idx_arr = self.params_idxs
        var_arr = self.params
        if name in idx_arr:
            var_idx = idx_arr[name]
            return var_arr[var_idx]
        return None

    def get_ctrls_vars(self) -> List[ControlVariable]:
        return self.ctrls
    
    def get_ctrl_var(self, name: str) -> DynamicsVariable:
        idx_arr = self.ctrls_idxs
        var_arr = self.ctrls
        if name in idx_arr:
            var_idx = idx_arr[name]
            return var_arr[var_idx]
        return None

    def init_dynamics(self, use_jax=True):
        self._dynfunc = self.get_dynamics()
        self._is_initialized = True
        self._dynfunc_numba = numba.jit(self.get_dynamics_numba())
        if not use_jax:
            self._dynfync_jit = self._dynfunc
        else:
            self._dynfunc_jit = jax.jit(self._dynfunc)

    def add_state(self, name: str) -> DynamicsVariable:
        new_idx = len(self.states)
        new_var = DynamicsVariable(name, new_idx)
        self.states.append(new_var)
        self.state_idxs[name] = new_idx
        return new_var

    def add_ctrl(self, name: str, lb:float=-1e16, ub:float=1e16) -> ControlVariable:
        new_idx = len(self.ctrls)
        new_var =ControlVariable(name, new_idx, lb=lb, ub=ub)
        self.ctrls.append(new_var)
        self.ctrls_idxs[name] = new_idx
        return new_var

    def add_param(self, name: str, value: float) -> ParameterVariable:
        new_idx = len(self.params)
        new_var = ParameterVariable(name, new_idx, value)
        self.params.append(new_var)
        self.params_idxs[name] = new_idx
        return new_var

    def get_state_idx(self, name: str) -> int:
        if name in self.state_idxs:
            return self.state_idxs[name]
        return -1

    def get_ctrl_idx(self, name: str) -> int:
        if name in self.ctrls_idxs:
            return self.ctrls_idxs[name]
        return -1

    def get_param_idx(self, name: str) -> int:
        if name in self.params_idxs:
            return self.params_idxs[name]
        return -1

    def get_states(self, x_arr: np.ndarray) -> jnp.ndarray:
        end_idx = self.num_states
        return jnp.array(x_arr[0:end_idx])

    def get_params(self, x_arr: np.ndarray) -> jnp.ndarray:
        end_idx = self.num_states + self.num_params
        return jnp.array(x_arr[self.num_states:end_idx])

    def get_ctrls(self, x_arr: np.ndarray) -> jnp.ndarray:
        start_idx = self.num_states + self.num_params
        return jnp.array(x_arr[start_idx:])

    def make_xarr(self, states, params, ctrls) -> np.ndarray:
        return np.concatenate([states, params, ctrls])

    def split_xarr(self, x_arr: np.ndarray) -> List[jnp.ndarray]:
        states = self.get_states(x_arr)
        params = self.get_params(x_arr)
        actions = self.get_actions(x_arr)
        return states, params, action

    def eval_jnp(self, x_arr: jnp.ndarray) -> jnp.ndarray:
        states, params, actions = self.split_xarr(x_arr)
        out_vec = self._dynfunc_jit(states, params, actions)
        return out_vec

    def eval_np(self, x_arr: np.ndarray) -> np.ndarray:
        '''
        Evaluate the dynamics function:
        '''
        in_vec = jnp.array(x_arr)
        out_vec = self.eval_jnp(in_vec)
        return np.array(out_vec)
