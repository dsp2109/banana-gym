#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplifie Banana selling environment.

Each episode is selling a single banana.
"""

# core modules
import random
import math

# 3rd party modules
import gym
import numpy as np
from gym import spaces


def get_chance(x):
    """Get probability that a banana will be sold at price x."""
    raise RuntimeError("still accessing banana things in get_chance") #DP NOT NEEDED
    e = math.exp(1)
    return (1.0 + e) / (1. + math.exp(x + 1))


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
        self.MAX_PRICE = 2.0
        self.TOTAL_TIME_STEPS = 2

        self.curr_step = -1
        self.is_banana_sold = False

        #DP EDITS
        self.curr_hand = -1
        self.is_hand_over = False
        self.is_hand_won = False #replacememnt for self.is_banana_sold = False

        # Define what the agent can do
        # Sell at 0.00 EUR, 0.10 Euro, ..., 2.00 Euro
        self.action_space = spaces.Discrete(13)
        #DP, do we need this many actions? Need to figure out how to represent actions.
        #{"fold":0, "check": 1, "call":2, "bet 0.2":3, "bet 0.3": 4, "bet 0.4": 5, "bet 0.5": 6, "raise 0.6": 7,"raise 0.75":8, "raise 1":9, "raise 1.5":10, "raise 2":11,"raise 5":12, "raise all":12}
        #maybe reduce this for initial tests [0.2, 0.35. 0.5, 0.7, 1, 1.5, 4, all-in]

        # Observation is the remaining time
        low = np.array([0.0,  # remaining_tries
                        ])
        high = np.array([self.TOTAL_TIME_STEPS,  # remaining_tries
                         ])
        self.observation_space = spaces.Box(low, high)
        #DP, https://github.com/openai/gym/blob/master/gym/spaces/box.py
        #DP, will need to figure out low and high

        # Store what the agent tried
        self.curr_episode = -1
        self.action_episode_memory = []

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
        self.curr_step += 1
        self._take_action(action)
        reward = self._get_reward()
        ob = self._get_state()
        return ob, reward, self.is_hand_over, {}

    def _take_action(self, action):
        #DP check does action meet what we planned?
        self.action_episode_memory[self.curr_episode].append(action)
        self.price = ((float(self.MAX_PRICE) /
                      (self.action_space.n - 1)) * action)

        chance_to_take = get_chance(self.price)
        banana_is_sold = (random.random() < chance_to_take)

        if banana_is_sold:
            self.is_banana_sold = True

        remaining_steps = self.TOTAL_TIME_STEPS - self.curr_step
        time_is_over = (remaining_steps <= 0)
        throw_away = time_is_over and not self.is_banana_sold
        if throw_away:
            self.is_banana_sold = True  # abuse this a bit
            self.price = 0.0

    def _get_reward(self):
        """Reward is given for a sold banana."""
        if self.is_banana_sold:
            return self.price - 1
        else:
            return 0.0

        #DP edit
        if self.is_hand_over:


    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """
        self.curr_episode += 1
        self.action_episode_memory.append([]) #DP - huh? append empty list?
        self.is_banana_sold = False
        self.price = 1.00
        return self._get_state()

    def _render(self, mode='human', close=False):
        print("tried to render - no rendering function built") #DP edit
        return

    def _get_state(self):
        """Get the observation."""
        ob = [self.TOTAL_TIME_STEPS - self.curr_step]
        return ob
