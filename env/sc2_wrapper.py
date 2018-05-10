"""Wrapper around PySC2."""
import numpy as np
from pysc2.env import sc2_env
from pysc2.lib import actions, features

class SC2Env:
    def __init__(self, map_name='MoveToBeacon', screen_size=64, minimap_size=64, step_mul=8, visualize=False):
        self.map_name = map_name
        self.screen_size = screen_size
        self.env = sc2_env.SC2Env(
            map_name=map_name,
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=screen_size, minimap=minimap_size),
                use_feature_units=True,
            ),
            step_mul=step_mul, visualize=visualize,
        )
    def reset(self):
        obs = self.env.reset()
        return self._process(obs[0])
    def step(self, action_id, action_args=None):
        if action_args is None:
            action_args = [[0], [0, 0]]
        obs = self.env.step([actions.FunctionCall(action_id, action_args)])
        ob = obs[0]
        return self._process(ob), ob.reward, ob.last()
    def _process(self, obs):
        return {
            'screen': np.array(obs.observation.feature_screen, dtype=np.float32),
            'minimap': np.array(obs.observation.feature_minimap, dtype=np.float32),
        }
    def close(self):
        self.env.close()
