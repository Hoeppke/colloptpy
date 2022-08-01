import numpy as np
import pandas as pd
from .node import Node
from typing import List, Tuple, Dict
from dataclasses import dataclass
from dataclasses_json import dataclass_json
import json


class DomainLayout():
    '''
    Class to organise the layout of a domain, in terms of the segment and node placement.
    '''
    def __init__(self):
        self.segment_dicts = []

    def add_segment(self, width: float, method: str, order: int):
        '''
        Add a single method with the corresponding collocation method
        '''
        new_segment = {}
        new_segment['width'] = width
        new_segment['method'] = method
        new_segment['order'] = order
        self.segment_dicts.append(new_segment)

    def get_segments(self) :
        return self.segment_dicts

    @staticmethod
    def load_json(file_name) -> 'DomainLayout':
        new_layout = DomainLayout()
        with open(file_name, 'r') as fs:
            data = json.load(fs)
        for seg in data:
            width, method, order = seg['width'], seg['method'], seg['order']
            new_layout.add_segment(width, method, order)
        return new_layout
