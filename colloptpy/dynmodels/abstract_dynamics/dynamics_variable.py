from typing import List, Tuple


class DynamicsVariable():

    def __init__(self, name:str, idx: int, lb: float=-1e16, ub: float=1e16, desc:str=None):
        self.name: str = name
        self.idx: int = idx
        self.desc: str = desc
        self.lb: float = lb
        self.ub: float = ub

    def get_bounds(self) -> Tuple[float, float]:
        return (self.lb, self.ub)

    def get_idx(self) -> int:
        return self.idx
        
    def get_name(self) -> str:
        return self.name

