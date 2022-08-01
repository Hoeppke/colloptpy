import torch as th
from typing import List, Tuple
from .model_variable import ModelVariable


class LinearModelVariable(ModelVariable):


    def __init__(self, idx: int, name='unnamed variable', offset=0.0):
        super().__init__(idx, name=name)
        self.offset = offset
        self.facs = []
        self.others = []

    def add(self, other: 'LinearModelVariable', fac: float):
        self.facs.append(fac)
        self.others.append(other)
