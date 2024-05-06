import numpy as np
import gymnasium as gym

from gymnasium import spaces

# Reservoir class proposal
# - current_water_volume
# - capacity
# - demand (0 for reservoirs without demand)
# - deficit (value that is added to every month with deficit, to calculate yearly average)
# - evaporation_rate (evap per volume)
#
#   QUESTION: Do we need a separate class for power plants?
#       On the one hand, they are similar to other reservoirs, since they have a capacity and demand
#       On the other hand, the metrics for deficit are different and might be difficult to calculate without a separate class


class NileEnv(gym.Env):
    # See https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/
    # for example of continuous observation and action space

    # QUESTION: Should we add reservoirs at init, or have a method for adding reservoirs?
    def __init__(self, reservoirList):
        # Add all reservoirs to observation space (or only their water levels)
        # Agent 'location' is determined by water level at each reservoir
        # Use Box with continuous values (range with float32) for each reservoir (min and max water levels)
        self.observation_space = spaces.Box()

        # Action space with how much water can be released from each reservoir
        # Use Box with continuous values (range with float32)
        # Minimum value is 0 and maximum value is capacity of reservoir
        self.action_space = spaces.Box()

    def _get_info(self):
        # Return useful information about current state, such as:
        # - Water volumes in reservoirs
        # - Surplusses & Deficits
        # - Current reward
        return

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        # QUESTION: Should we do this in our implementation, or does initial state depend on specific case study
        super().reset(seed=seed)

        # Set agent's state (all reservoirs or all their water levels), corresponding to case study
        self._agent_state = []

        # In the docs, they create a separate _get_obs method that returns the observation (theirs contains agent as well as target)
        # QUESTION: Since I think we do not define a target for our problem, should we create a separate method?
        observation = self._agent_state
        info = self._get_info()

        return observation, info

    # Each step should be one month
    def step(self, action):
        # Actually doing something with the action --> the most important step
        # This is the place for all calculations
        # For each outflow determined in action, substract it from the corresponding reservoir
        # Outflow means inflow at the next reservoir, so add it to corresponding reservoir
        for i, flow in enumerate(action):
            self._agent_state[i] -= flow
            if i < len(action):
                # Optionally incorporate loss of water
                WATER_LOSS = 0.0
                self._agent_state[i + 1] += flow * (1 - WATER_LOSS)

        # Update action space based on new state
        self.action_space

        # Write method to check if terminated or truncated
        terminated = bool
        truncated = bool

        # Write reward function, based on model_nile.py's simulate/evaluate functions
        # Reward functions for the following objectives, simply add them up for single-objective purposes for now
        # - Egypt average yearly demand deficit (minimise)
        # - HAD frequency of months below minimum power generation level (minimise)
        # - Sudan average yearly demand deficit (minimise)
        # - Ethiopia yearly hydro-energy generation from GERD (maximisation)
        reward = float

        observation = self._agent_state
        info = self._get_info()

        return observation, reward, terminated, truncated, info
