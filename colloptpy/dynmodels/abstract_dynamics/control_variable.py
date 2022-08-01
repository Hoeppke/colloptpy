from .dynamics_variable import DynamicsVariable


class ControlVariable(DynamicsVariable):

    def __init__(self, name: str, idx: int, lb:float=-1e16, ub: float=+1e16, desc: str=None):
        super(ControlVariable, self).__init__(name, idx, desc=desc, lb=lb, ub=ub)

    def get_action_bounds(self) -> Tuple[float, float]:
        return self.get_bounds()

