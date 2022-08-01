from ..dynmodels.dynamic_model import DynamicModel


class Node():
    
    def __init__(self, node_idx: int, pos: float, has_ctrl=False):
        self.idx = node_idx
        self.pos = pos
        self._has_ctrl = has_ctrl
        self.state_idxs: list[int] = []
        self.ctrl_idxs: list[int] = []

    def has_ctrl(self) -> bool:
        return self._has_ctrl

    def set_indices(self, start_idx: int, model: DynamicModel):
        c_start_idx = start_idx + model.get_num_states()
        s_idxs = [start_idx + k for k in range(model.get_num_states())]
        c_idxs = [c_start_idx+ k for k in range(model.get_num_controls())]
        self.state_idxs = s_idxs
        if self.has_ctrl():
            self.ctrl_idxs = c_idxs

    def get_state_idxs(self) -> list[int]:
        return self.state_idxs

    def get_ctrl_idxs(self) -> list[int]:
        if self.has_ctrl():
            return self.ctrl_idxs
        else:
            return None

    def get_last_vidx(self):
        if self.has_ctrl():
            return self.ctrl_idxs[-1]
        return self.state_idxs[-1]

    def get_pos(self) -> float:
        return self.pos

    def get_idx(self) -> int:
        return self.idx
