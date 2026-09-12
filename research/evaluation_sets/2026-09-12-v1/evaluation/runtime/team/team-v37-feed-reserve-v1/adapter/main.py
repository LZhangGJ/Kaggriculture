"""Lifecycle adapter for the unchanged public Python candidate."""
import importlib.util
from pathlib import Path

class Agent:
    def __init__(self):
        path=Path(__file__).resolve().parent.parent/'main.py'
        spec=importlib.util.spec_from_file_location('_frozen_team_candidate',path)
        self.module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.config={}
    def __call__(self,observation,configuration=None):
        return self.module.agent(observation,configuration)
    def debug(self):return {}
    def close(self):
        reset=getattr(self.module,'reset',None)
        if reset:reset()

def create_agent():return Agent()
