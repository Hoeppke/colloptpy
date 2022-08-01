from .saved_solution import SavedSolution
from typing import List, Dict, Tuple, Optional, Callable
import uuid
import pandas as pd
import json
import numpy as np
import time

class SolutionHistory():

    def __init__(self, var_names: List[str], num_nodes: int):
        self.var_names: List[str] = var_names
        self.num_nodes: int = num_nodes
        self.sol_steps: List[SavedSolution] = []
        self.index_arr: List[int] = []
        self.sol_times: List[float] = []

    def get_entry(self, idx: int) -> SavedSolution:
        return self.sol_steps[idx]

    def get_entries(self) -> List[SavedSolution]:
        return self.sol_steps

    def iter(self) -> List[Tuple[int, SavedSolution]]:
        out_lst = list(zip(self.index_arr, self.sol_steps))
        return out_lst

    def var_matches(self, sol: SavedSolution):
        var_names = set(self.var_names)
        sol_names = set(sol.var_names)
        return var_names == sol_names

    def add_solution(self, sol: SavedSolution, idx: int=None, tme: float=None):
        if not self.var_matches(sol):
            raise ValueError('could not match solution variables')
        if not self.num_nodes == sol.get_num_nodes():
            raise ValueError('solution has wrong number of nodes')
        if idx == None:
            idx = len(self.index_arr)
        idx = int(idx)
        if tme is None:
            my_time = time.time()
        else:
            my_time = tme
        self.sol_steps.append(sol)
        self.index_arr.append(idx)
        self.sol_times.append(my_time)

    def save_history(self, save_file: str):
        '''
        Save the history as a (potentially large) csv file
        '''
        out_list = []
        for idx, sol in enumerate(self.sol_steps):
            sol_idx = self.index_arr[idx]
            new_dict = sol.to_df_row()
            new_dict['opt_iter'] = sol_idx
            out_list.append(new_dict)
        out_df = pd.DataFrame(out_list)
        out_df.to_csv(save_file)

    def iter_entries(self):
        for k in range(len(self.sol_steps)):
            sol = self.sol_steps[k]
            entry = {}
            entry['index'] = self.index_arr[k]
            entry['solution'] = sol.to_dict()
            entry['time'] = self.sol_times[k]
            yield entry

    def save_json(self, save_file: str):
        '''
        Save the history as a (potentially large) json file
        '''
        print('Saving solution to: ', save_file)
        all_entries = list(self.iter_entries())
        with open(save_file, 'w') as fp:
            json.dump(all_entries, fp, indent=4)

    def compute_mean_std(self) -> Tuple[SavedSolution, SavedSolution]:
        '''
        Find, for each variable at each node the mean over the
        solution history.
        '''
        mean_history = SavedSolution(self.var_names, self.num_nodes)
        std_history = SavedSolution(self.var_names, self.num_nodes)
        num_sols = len(self.sol_steps)
        for varname in self.var_names:
            val_mtx = np.zeros((self.num_nodes, num_sols), dtype=np.float64)
            # Assemble value matrix
            for solidx, sol in enumerate(self.sol_steps):
                vec = sol.get_values(varname)
                val_mtx[:, solidx] = vec
            # Compute mean, std
            mean_vec = np.mean(val_mtx, axis=1)
            std_vec = np.std(val_mtx, axis=1)
            mean_history.set_values(varname, mean_vec)
            std_history.set_values(varname, std_vec)
        return mean_history, std_history

    @staticmethod
    def load_json(save_file: str) -> 'SolutionHistory':
        if not save_file.endswith('.json'):
            if '.' in save_file:
                ext = save_file.split('.')[-1]
            else:
                ext = ''
            msg = 'Found .{} extension, expected .json'.format(ext)
            raise ValueError(msg)
        # Load the file
        hist = None
        with open(save_file, 'r') as fp:
            data_dict = json.load(fp)
            for entry in data_dict:
                idx = entry['index']
                sol_dict = entry['solution']
                tme = entry['time']
                sol: SavedSolution = SavedSolution.load_dict(sol_dict)
                if hist is None:
                    var_names = sol.var_names
                    num_nodes = sol.num_nodes
                    hist = SolutionHistory(var_names, num_nodes)
                hist.add_solution(sol, idx, tme)
                # Load the solution
        return hist
