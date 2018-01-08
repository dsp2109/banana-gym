#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplified poker hand history environment.

Each episode is selling a hand from one player's perspective.
"""

# core modules
import random
import math

# 3rd party modules
import gym
import numpy as np
import pandas as pd
from gym import spaces

class PokerHistEnv(gym.Env):
    """
    Define a simple Banana environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        print("PokerHistEnv - Version {}".format(self.__version__))

        # General variables defining the environment
        self.SHAPE = (13, 13, 16)

        #DP EDITS
        self.curr_step = -1
        self.curr_episode = -1
        self.curr_action = -1
        self.is_hand_over = False
        self.is_hand_won = False 

        self.action_space = spaces.Discrete(11)
        # action_choices = {"fold":0, "check_call_0": 1, "bet 0.1":2, "bet 0.25": 3,
        # "bet 0.4": 4, "bet 0.55": 5, "bet 0.7": 6,"bet 1":7, "bet 1.5":8,"bet 2.5":9,
        # "bet all":10}

        # Observation is the game state and prior actions
        
        low = np.zeros(self.SHAPE)
        high = np.ones(self.SHAPE)
        self.observation_space = spaces.Box(low, high)
        #DP, https://github.com/openai/gym/blob/master/gym/spaces/box.py
        
        self.hh_df = pd.read_pickle(\
            "C:/Users/dsp21/NYDS/Project/Capstone Ideas/Poker bot/git_things/test_episodes1000.pickle")
        #columns = ["obs", "acts", "reward", "done", "num_steps"]


    def _step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.is_hand_over:
            raise RuntimeError("Episode is done")

        reward = self._get_reward()
        self._take_action(action) 
        self.curr_step += 1
        ob = self._get_state() #observation is next one

        return ob, reward, self.is_hand_over, {}

    def _get_action(self):
        self.curr_action = self.hh_df.loc[self.curr_episode,"acts"][0][self.curr_step]
        return self.curr_action

    def _take_action(self, action):
        self._get_action()
        if self.curr_action != action:
            import pdb; pdb.set_trace()
            raise RuntimeError("action taken is not same as saved hand_log action")
            #Does action meet what we planned?

        self.is_hand_over = self.hh_df.loc[self.curr_episode,"done"][0][self.curr_step]


    def _get_reward(self):
        """Reward is given for a sold banana."""
        return self.hh_df.loc[self.curr_episode,"reward"][0][self.curr_step]


    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_episode += 1
        self.curr_step = 0
        self._get_action()
        self.is_hand_over = False

        return self._get_state()

    def _get_state(self):
        """Get the observation."""
        ob = self.hh_df.loc[self.curr_episode,"obs"][0][self.curr_step] #+1 for next observation
        return ob

    def _render(self, mode='human', close=False):
        print("tried to render - no rendering function built") #DP edit
        return