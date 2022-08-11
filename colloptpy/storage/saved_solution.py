from typing import List, Dict, Tuple, Union
import json
import numpy as np
import datetime as dt
import pandas as pd
import uuid
from scipy import interpolate as interp

class SavedSolution():

    def __init__(self, var_names: List[str], num_nodes: int):
        self.dtype=np.float64
        self.var_names: List[str] = var_names
        self.num_nodes: int = num_nodes
        self.num_vars = len(self.var_names)
        self.values = np.zeros((self.num_nodes, self.num_vars), dtype=self.dtype)
        self.uid: str = str(uuid.uuid4())
        self.dt_fmt = '%Y-%m-%d %H:%M:%S'
        self.time_stamp: dt.datetime = None

    def set_timestamp(self, solution_date: dt.datetime):
        if isinstance(solution_date, str):
            self.time_stamp = dt.datetime.strptime(solution_date, self.dt_fmt)
        else:
            self.time_stamp: dt.datetime = solution_date

    def get_timestamp(self) -> dt.datetime:
        return self.time_stamp

    def get_node_values(self, node: int) -> Dict[str, float]:
        node_values: np.array = self.values[node, :]
        out_dict = dict(zip(self.var_names, node_values))
        return out_dict

    def get_num_nodes(self) -> int:
        return self.num_nodes

    def set_values(self, var: str, values: np.ndarray):
        if var not in self.var_names:
            self.var_names.append(var)
            self.num_nodes += 1
            new_values = np.zeros((self.num_nodes, self.num_vars), dtype=self.dtype)
            new_values[:, :-1] = self.values
            self.values = new_values
        if values.shape[0] != self.num_nodes:
            msg = 'found {} nodes, expected {}'
            msg = msg.format(values.shape[0], self.num_nodes)
            raise ValueError(msg)
        idx = self.var_names.index(var)
        self.values[:, idx] = values 

    def get_variables(self) -> List[str]:
        return self.var_names

    def get_values(self, var: Union[str, int, list]):
        if isinstance(var, int):
            col_idx = var
            return self.values[:, col_idx]
        elif isinstance(var, str):
            if var not in self.var_names:
                raise ValueError('Variabel {} not in saved solution'.format(var))
            col_idx = self.var_names.index(var)
            return self.values[:, col_idx]
        elif isinstance(var, list):
            return [self.get_values(v) for v in var]
        raise ValueError('Undefined variable:', var)

    def get_values_interp(self, var: str, ivar:str, ivals: List[float]):
        '''
        Get the values of an interpolated variable:
        '''
        v_idx = self.var_names.index(var) 
        vi_idx = self.var_names.index(ivar)
        vix = self.get_values(vi_idx)
        viy = self.get_values(v_idx)
        out_vals = np.interp(ivals, vix, viy)
        return out_vals

    def set_values_interp(self, var: str, vals: List[float], ivar: str, ivals: List[float]):
        '''
        Set the values "vals" of variable "var" according the 'x' variable
        "ivar" at the positions "ivals".
        '''
        x_out_vals = self.get_values(ivar)
        y_vals = np.interp(x_out_vals, ivals, vals)
        self.set_values(var, y_vals)

    def save(self, fname):
        if '.json' in fname:
            out_dict = self.to_dict()
            with open(fname, 'w') as fp:
                json.dump(out_dict, fp, indent=4)
        elif '.csv' in fname:
            df = self.to_dataframe()
            df.to_csv(fname, index=False) 

    def to_dict(self) -> Dict[str, List[float]]:
        out_dict = {}
        for idx, var_names in enumerate(self.var_names):
            out_dict[var_names] = list(self.values[:, idx])
        if self.time_stamp is not None:
            out_dict['time_stamp'] = self.time_stamp.timestamp()
        out_dict['uid'] = self.uid
        return out_dict

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self.values, columns=self.var_names)
        return df

    def to_df_row(self) -> Dict[str, float]:
        out_dict = {}
        for vidx, var_name in enumerate(self.var_names):
            values = self.values[:, vidx]
            for tidx, value in enumerate(values):
                key = '{}_{}'.format(var_name, tidx)
                out_dict[key] = value
        return out_dict

    def interpolate(self, x_var: str, x_values, kind:str='linear') -> 'SavedSolution':
        '''
        Linear interpolation of solution to a different x array
        '''
        var_names = self.var_names
        num_nodes = len(x_values)
        out_sol = SavedSolution(var_names, num_nodes)
        x_arr = self.get_values(x_var)
        x_arr_new = np.array(x_values)
        other_names = list(set(self.var_names) - set([x_var]))
        other_names.sort()
        out_sol.set_values(x_var, x_values)
        for var_name in other_names:
            y_arr = self.get_values(var_name)
            spline = interp.interp1d(x_arr, y_arr, kind=kind, fill_value='extrapolate')
            y_arr_new = spline(x_arr_new)    
            out_sol.set_values(var_name, y_arr_new)
        return out_sol

    @staticmethod
    def load(fname: str) -> 'SavedSolution':
        '''
        Load a saved solution from either json or csv format
        '''
        if fname.endswith('json'):
            return SavedSolution.load_json(fname)
        if fname.endswith('csv'):
            return SavedSolution.load_csv(fname)

    @staticmethod
    def load_json(fname) -> 'SavedSolution':
        with open(fname, 'r') as fp:
            data = json.load(fp)
        return SavedSolution.load_dict(data)

    @staticmethod
    def load_csv(fname) -> 'SavedSolution':
        data = pd.read_csv(fname)
        var_names = list(data.columns)
        num_nodes = len(data[var_names[0]])
        sol = SavedSolution(var_names, num_nodes)
        for var in var_names:
            sol.set_values(var, data[var].values)
        return sol

    @staticmethod
    def load_dict(data: Dict[str, List[float]]) -> 'SavedSolution':
        var_names = list(data.keys())
        num_nodes = len(data[var_names[0]])
        sol: SavedSolution = SavedSolution(var_names, num_nodes)
        for name in data:
            if name != 'time_stamp':
                values = np.array(data[name])
                sol.set_values(name, values)
        if 'time_stamp' in data:
            ts = dt.datetime.fromtimestamp(data['time_stamp'])
            sol.set_timestamp(ts)
        if 'uuid' in data:
            sol.uid = data['uuid']
        return sol
