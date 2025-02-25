import hockey.hockey_env as h_env
import numpy as np

class HockeyEnvWrapper:
    def __init__(self, mode=None, render=False):
        
        self.env = h_env.HockeyEnv(mode=mode)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.render = render

    def reset(self):
        # Resets the env
        obs, info = self.env.reset()
        if self.render:
            self.env.render()
        return obs, info

    def step(self, action):
       
       # Takes a step in the environment with the given action.
        
       # Takes action as an arg. The action to take, in the form of a NumPy array.
        
        # Retrns a Tuple: (next_state, reward, done, truncated, info)
        
        # Clip the action first
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        next_state, base_reward, done, truncated, info = self.env.step(action)

        # Reward shaping, should help to add numerical stability, without it the agent was not learning properly, 
        # after each episode where the reward was positive, in the next one it was significantly decreasing
        shaping_reward = 0.0
        try:
            agent_x, agent_y = next_state[0], next_state[1]
            puck_x, puck_y   = next_state[2], next_state[3]
            dist = np.sqrt((agent_x - puck_x)**2 + (agent_y - puck_y)**2)

            # If near the puck, add a bonus
            if dist < 0.5:
                shaping_reward += 0.01
        except:
            # If for some reason there is a problem with state format for example it doesn't match, do nothing.
            pass

        # Combine base reward + shaping
        reward = base_reward + shaping_reward

        if self.render:
            self.env.render()

        return next_state, reward, done, truncated, info

    def close(self):
        #closes the env
        self.env.close()
