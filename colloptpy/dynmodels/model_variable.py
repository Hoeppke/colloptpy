import torch as th
from typing import List, Tuple


class ModelVariable():


    def __init__(self, idx: int, name='unnamed variable'):
        self.idx = idx
        self.name = name
        #self.value: th.Tensor = th.tensor([value], requires_grad=True)
        self.bound_low: float = -1e10
        self.bound_upper: float = +1e10

    def get_name(self) -> str:
        return self.name

    def set_bounds(self, lb: float, ub: float):
        self.bound_low = lb
        self.bound_upper = ub

    def get_bounds(self) -> Tuple[float, float]:
        return (self.bound_low, self.bound_upper)

    def get_idx(self) -> int:
        return self.idx

    #def val(self) -> th.Tensor:
    #    return self.value

    def valf(self, vals_th) -> float:
        val = vals_th[self.get_idx()]
        return val

    #def set_value(self, value: float):
    #    self.value: th.Tensor = th.tensor([value], requires_grad=True)

    #def set(self, tensor_val: th.Tensor):
    #    self.value = tensor_val
