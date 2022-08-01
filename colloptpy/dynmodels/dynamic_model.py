import functorch
import numpy as np
import torch as th
from abc import ABC, abstractmethod
from .model_variable import ModelVariable
from typing import List


class DynamicModel(ABC):

    def __init__(self):
        self.num_states: int = 0
        self.num_controls: int = 0
        self.num_params: int = 0
        self.num_static_params: int = 0
        self.name = 'Generic Dynamic Model'
        self.states: List[ModelVariable] = []
        self.ctrls: List[ModelVariable] = []
        self.params: List[ModelVariable] = []
        self._state_th = th.zeros(1, requires_grad=True)
        self._control_th = th.zeros(1, requires_grad=True)
        self._params_th = th.zeros(1, requires_grad=True)

    def get_model_name(self) -> str:
        return self.name

    def get_main_variable(self) -> str:
        return 'time'

    def get_state_names(self) -> List[str]:
        out_list = [var.get_name() for var in self.states]
        return out_list

    def get_control_names(self) -> List[str]:
        out_list = [var.get_name() for var in self.ctrls]
        return out_list

    def get_parameter_names(self) -> List[str]:
        out_list = [var.get_name() for var in self.params]
        return out_list

    def add_state(self, name: str) -> ModelVariable:
        new_var = ModelVariable(len(self.states), name)
        self.states.append(new_var)
        self.num_states += 1
        self._state_th = th.zeros(self.num_states, requires_grad=True)
        return new_var

    def add_control(self, name: str) -> ModelVariable:
        new_var = ModelVariable(len(self.ctrls), name)
        self.ctrls.append(new_var)
        self.num_controls += 1
        self._control_th = th.zeros(self.num_controls, requires_grad=True)
        return new_var

    def add_param(self, name: str) -> ModelVariable:
        new_var = ModelVariable(len(self.params), name)
        self.params.append(new_var)
        self.num_params += 1
        self._params_th = th.zeros(self.num_params, requires_grad=True)
        return new_var

    def get_all_variables(self) -> List[ModelVariable]:
        """
        Get a list of all state and ctrl model variables
        """
        return self.states + self.ctrls

    def get_num_states(self) -> int:
        return self.num_states

    def get_num_controls(self) -> int:
        return self.num_controls

    def get_num_params(self) -> int:
        return self.num_params

    def get_states(self, need_grad=True) -> th.Tensor:
        return self._state_th

    def get_controls(self, need_grad=True) -> th.Tensor:
        return self._control_th

    def get_params(self, need_grad=True) -> th.Tensor:
        return self._params_th

    @abstractmethod
    def forward(self, states: th.Tensor, controls: th.Tensor) -> th.Tensor:
        pass 

    def forward_noparam_base(self, state_ctrl: th.Tensor) -> th.Tensor:
        states = state_ctrl[0:self.num_states].reshape((1, self.num_states))
        ctrls = state_ctrl[self.num_states:].reshape((1, self.num_controls))
        out_vals = self.forward_noparam(states, ctrls)[0, :]
        return out_vals

    def forward_jac(self, state_ctrl: th.Tensor) -> th.Tensor:
        return functorch.jacfwd(self.forward_noparam_base)(state_ctrl)
