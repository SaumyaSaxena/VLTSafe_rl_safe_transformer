import gym
import numpy as np
from abc import ABC, abstractmethod
from gym.utils import EzPickle
from omegaconf import DictConfig, OmegaConf
from typing import Type, Any, Union, Optional
import mujoco

class BaseMujocoEnv(gym.Env, EzPickle, ABC):
    def __init__(self, device: Optional[Any], cfg: Type[Union[DictConfig, OmegaConf]]):
        super().__init__()
        EzPickle.__init__(self)
        self.device = "cpu" if device is None else device
        self.env_cfg = cfg

        self.reward = self.env_cfg.reward
        self.penalty = self.env_cfg.penalty
        self.doneType = self.env_cfg.doneType
        self.costType = self.env_cfg.costType
        self.return_type = self.env_cfg.return_type
        self.scaling_target = self.env_cfg.scaling_target
        self.scaling_safety = self.env_cfg.scaling_safety
    
    @property
    @abstractmethod
    def env_observation_shapes(self):
        pass
    
    @abstractmethod
    def reset(self, **kwargs):
        pass
    
    def get_cost(self, l_x, g_x, success, fail):
        if self.costType == 'dense_ell':
            cost = l_x
        elif self.costType == 'dense':
            cost = l_x + g_x
        elif self.costType == 'sparse':
            cost = 0.
        elif self.costType == 'max_ell_g':
            if 'reward' in self.return_type:
                cost = np.minimum(l_x, g_x)
            else:
                cost = np.maximum(l_x, g_x)
        else:
            raise ValueError("invalid cost type!")

        if self.env_cfg.get('shape_reward', False):
            if 'reward' in self.return_type:
                cost[success] = -1.*self.reward
                cost[fail] = -1.*self.penalty

                l_x[success] = -1.*self.reward
                g_x[fail] = -1.*self.penalty
            else:
                cost[success] = self.reward
                cost[fail] = self.penalty

                l_x[success] = self.reward
                g_x[fail] = self.penalty

        return cost, l_x, g_x

    def get_done(self, state, success, fail):
        # state: shape(batch,n)
        if self.doneType == 'toEnd':
            done = self.check_within_env(state)
        elif self.doneType == 'fail':
            done = fail
        elif self.doneType == 'TF':
            done = np.logical_or(fail, success)
        elif self.doneType == 'all':
            done = np.logical_or(np.logical_or(fail, success), self.check_within_env(state))
            if self.failure_mode is None and self.check_within_env(state):
                self.failure_mode = 'out_of_env'
            elif success:
                self.failure_mode = 'success'
        elif self.doneType == 'real':
            real_fail = self.check_real_failure()
            done = np.logical_or(np.logical_or(real_fail, success), self.check_within_env(state))
            if self.failure_mode is None and self.check_within_env(state):
                self.failure_mode = 'out_of_env'
            elif success:
                self.failure_mode = 'success'
        else:
            raise ValueError("invalid done type!")
        return done
    
    @abstractmethod
    def step(self, action):
        pass
    
    @abstractmethod
    def render(self, mode='human'):
        pass

    @abstractmethod
    def safety_margin(self, **kwargs):
        pass

    @abstractmethod
    def target_margin(self, **kwargs):
        pass

    @abstractmethod
    def check_failure(self, **kwargs):
        pass
    
    @abstractmethod
    def check_real_failure(self, **kwargs):
        pass

    @abstractmethod
    def check_success(self, **kwargs):
        pass

    @abstractmethod
    def check_within_env(self, **kwargs):
        pass

    @abstractmethod
    def plot_trajectory(self, **kwargs):
        pass