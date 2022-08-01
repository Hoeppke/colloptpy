from .dynamics_variable import DynamicsVariable


class StateVariable(DynamicsVariable):
    
    def __init__(self, name: str, idx: int, lb: float=-1e16, ub: float=+1e16, desc: str=None):
        super(StateVariable, self).__init__(name, idx, desc=desc, lb=lb, ub=ub)

    def get_state_bounds(self) -> Tuple[float, float]:
        return self.get_bounds()

