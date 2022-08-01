from .bicycle_model import BicycleModel
from .bicycle_model_v2 import BicycleModel2
from track_model.track import Track
from typing import Union, List, Dict
import numpy as np

BModel = Union[BicycleModel, BicycleModel2]



class BikeModelVarNamer():

    def __init__(self, model: BModel, track: Track):
        self.model = model
        self.track = track

    def _get_model_vars(self, state: np.ndarray) -> Dict[str, float]:
        '''
        Extract variables from bicycle model
        '''
        out_dict = {}
        for var in self.model.states:
            vidx, vname = var.get_idx(), var.get_name()
            out_dict[vname] = state[vidx]
        return out_dict

    def _get_track_variables(self, state: np.ndarray) -> Dict[str, float]:
        '''
        Extract the track variables (only called if necessary).
        '''
        out_dict = {}
        pxi, pyi = self.model.x.get_idx(), self.model.y.get_idx()
        px, py = state[[pxi, pyi]]
        s_dist = self.track.get_sdist(px, py)
        out_dict['s'] = s_dist
        if s_dist is None:
            out_dict['xi'] = None
            out_dict['nu'] = None
        else:
            yaw = state[self.model.yaw.get_idx()]
            out_dict['xi'] = self.track.get_xi_angle(px, py, yaw)
            out_dict['nu'] = self.track.get_nudist(px, py)
        return out_dict
            

    def get_variables(self, state: np.ndarray, time=None) -> Dict[str, float]:
        out_dict = self._get_model_vars(state)
        if isinstance(self.model, BicycleModel2):
            # update to include track variables
            tvars = self._get_track_variables(state)
            out_dict.update(tvars)
        if time is not None:
            out_dict['time'] = time
        return out_dict
