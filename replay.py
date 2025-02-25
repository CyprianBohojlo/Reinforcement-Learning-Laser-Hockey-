import numpy as np

class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim):
        
        # Replay buffer for storing and sampling transitions
        # As args takes:
        #    max_size (int): Maximum number of transitions the buffer can hold.
        #    state_dim (int): Dimension of the state space.
        #    action_dim (int): Dimension of the action space.
        
        self.max_size = max_size
        self.current_idx = 0
        self.current_size = 0  # Remember it I renamed it from self.size

        #  memory for transitions
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros(max_size, dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros(max_size, dtype=bool)

    def add_transition(self, state, action, reward, next_state, done):
        
        # Adds a new transition to the buffer, should overwrite the oldest if full.
        # As args takes:
        #    state (np.ndarray): Current state.
        #    action (np.ndarray): Action taken.
        #    reward (float): Reward received.
        #    next_state (np.ndarray): Next state after action.
        #    done (bool): Whether the episode ended.
        
        self.states[self.current_idx] = state
        self.actions[self.current_idx] = action
        self.rewards[self.current_idx] = float(reward)  # Ensure correct dtype, before it was a error
        self.next_states[self.current_idx] = next_state
        self.dones[self.current_idx] = float(done)  # Ensure correct dtype, before it w as error

        # Updating buffer
        self.current_idx = (self.current_idx + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)  

    def sample(self, batch_size):
        
        #Samples a random batch of transitions from the buffer.
       
        # Returns a tuple of arrays: (states, actions, rewards, next_states, dones)
        
        if batch_size > self.current_size:  
            batch_size = self.current_size
        indices = np.random.choice(range(self.current_size), size=batch_size, replace=False)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices].reshape(-1, 1),  # Fixed shape (64,1)
            self.next_states[indices],
            self.dones[indices].reshape(-1, 1)  # Fixed shape (64,1)
        )

    def get_size(self):  
        """
        Returns the current size of the buffer.
        """
        return self.current_size  

    def get_all_transitions(self):
        """
        Returns all stored transitions.
        """
        return self.transitions[0:self.current_size]  
