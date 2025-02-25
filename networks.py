import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        
        #Actor network for the SAC algorithm.
        
        # As arg takes:    
        # state_dim: Dimension of the state space.
        #   action_dim: Dimension of the action space.
        #   action_bound: Bound for the actions (assumed symmetric).
        
        super(ActorNetwork, self).__init__()
        
        self.action_bound = action_bound

        layers = []
        layers.append(nn.Linear(state_dim, 256))  # Input layer (state_dim: 256)
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 256))  
        layers.append(nn.ReLU())

        self.fc = nn.Sequential(*layers)

        self.mu = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        

    def forward(self, state):
    
        x = self.fc(state) 
        mu = self.mu(x)  # Computes mean
        log_std = self.log_std(x)  # Computes log standard deviation
        log_std = torch.clamp(log_std, -20, 2)  # Clips for numerical stability
        std = torch.exp(log_std)  # Computes standard deviation
        return mu, std



    def sample_action(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        action = normal.rsample()  # This is for Reparameterization trick
        log_prob = normal.log_prob(action).sum(axis=-1)
        log_prob -= torch.sum(2 * (torch.log(torch.tensor(2.0)) - action - F.softplus(-2 * action)), axis=-1)
        action = torch.tanh(action) * self.action_bound
        return action, log_prob # Returns the action made by our Actor and log porbability


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        
        #Critic network for SAC (used for Q-value estimation).
        # As arg takes:
        #    state_dim: Dimension of the state space.
        #    action_dim: Dimension of the action space.
        
        super(CriticNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim + action_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
    
        q_value = self.fc(x)
        return q_value # Returns our Q-value


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        
        #Value network for SAC (used for state value estimation).
        # As an arg takes state_dim: Dimension of the state space.
        
        super(ValueNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 256))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(256, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, state):
        value = self.fc(state)
        return value
