import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from replay import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork

class SACAgent:
    def __init__(self, state_dim, action_dim, action_bound, replay_buffer, learning_rate=1e-4,
                 gamma=0.99, tau=0.01, alpha=0.05, device=None, target_entropy=None):
        # Defines the SAC agent
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.replay_buffer = replay_buffer
        self.current_lr = learning_rate  #Store current LR, used later for adaptive LR
        self.device = device if device is not None else torch.device('cpu')

        # If didn't specify target_entropy, the default is default to -action_dim
        if target_entropy is None:
            self.target_entropy = -action_dim
            self.auto_alpha = False
        else:
            self.target_entropy = target_entropy
            self.auto_alpha = True

        # Defining Actor, Value Nets and Critics, I used two critics in order to avoid overestimation bias 
        self.actor = ActorNetwork(state_dim, action_dim, action_bound).to(self.device)
        self.critic_1 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.critic_2 = CriticNetwork(state_dim, action_dim).to(self.device)
        self.value = ValueNetwork(state_dim).to(self.device)
        self.target_value = ValueNetwork(state_dim).to(self.device)

        # For optimizers I used Adam as the msot popular optimizer in Deep Learning
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.current_lr, weight_decay=1e-4)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=self.current_lr, weight_decay=1e-4)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=self.current_lr, weight_decay=1e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=self.current_lr, weight_decay=1e-4)

        # Copies parameters to the target value network, used later
        self.target_value.load_state_dict(self.value.state_dict())

        # If auto-tuning alpha enabled in train.py , it maintain log_alpha
        if self.auto_alpha:
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.current_lr)
        else:
            self.log_alpha = torch.tensor(np.log(alpha), requires_grad=False, device=self.device)
            self.alpha_optimizer = None

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def set_learning_rate(self, new_lr):
        
        # Adjusts the LR of all SAC optimizers at once.
        
        self.current_lr = new_lr
        for opt in [
            self.actor_optimizer,
            self.critic_1_optimizer,
            self.critic_2_optimizer,
            self.value_optimizer,
            self.alpha_optimizer if self.alpha_optimizer is not None else None
        ]:
            if opt is not None:
                for param_group in opt.param_groups:
                    param_group['lr'] = new_lr

    # Saving chekcpoints so resuming the training can be possible
    def save_full_checkpoint(self, path, save_replay=False):
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "value": self.value.state_dict(),
            "target_value": self.target_value.state_dict(),
            "log_alpha": self.log_alpha,
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_1_opt": self.critic_1_optimizer.state_dict(),
            "critic_2_opt": self.critic_2_optimizer.state_dict(),
            "value_opt": self.value_optimizer.state_dict(),
            "auto_alpha": self.auto_alpha
        }
        if self.auto_alpha:
            checkpoint["alpha_opt"] = self.alpha_optimizer.state_dict()

        
        torch.save(checkpoint, path)
        print(f"Full training state saved to {path}")

    # Loads checkpoints if I resume training
    def load_full_checkpoint(self, path, load_replay=False):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic_1.load_state_dict(checkpoint["critic_1"])
        self.critic_2.load_state_dict(checkpoint["critic_2"])
        self.value.load_state_dict(checkpoint["value"])
        self.target_value.load_state_dict(checkpoint["target_value"])

        self.auto_alpha = checkpoint.get("auto_alpha", False)
        self.log_alpha = checkpoint["log_alpha"]
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_1_optimizer.load_state_dict(checkpoint["critic_1_opt"])
        self.critic_2_optimizer.load_state_dict(checkpoint["critic_2_opt"])
        self.value_optimizer.load_state_dict(checkpoint["value_opt"])

        if self.auto_alpha and "alpha_opt" in checkpoint:
            if not hasattr(self, "alpha_optimizer") or self.alpha_optimizer is None:
                self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.current_lr)
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_opt"])

        print(f"Loaded full training state from {path}")

    

    def select_action(self, state, evaluate=False):
        
        #Selects an action for the given state.
        #If evaluate=True, returns a deterministic action (actor mean),
        # otherwise it should sample from the actor's Gaussian.
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.sample_action(state_tensor)
        return action.cpu().numpy().flatten()

    def update(self, batch_size):
        
        #Updates the actor, critic, and value networks using a sampled batch from the replay buffer
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).view(-1, 1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).view(-1, 1).to(self.device)

        # Updating the critic
        with torch.no_grad():
            next_actions, log_probs_next = self.actor.sample_action(next_states)
            log_probs_next = log_probs_next.view(-1, 1)
            target_v = self.target_value(next_states).view(-1, 1)
            target_values = target_v - self.alpha * log_probs_next
            y = rewards + self.gamma * (1 - dones) * target_values

        q1_pred = self.critic_1(states, actions)
        q2_pred = self.critic_2(states, actions)

        critic_1_loss = nn.MSELoss()(q1_pred, y)
        critic_2_loss = nn.MSELoss()(q2_pred, y)
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        #Updating the value
        with torch.no_grad():
            sampled_actions, log_probs = self.actor.sample_action(states)
            log_probs = log_probs.view(-1, 1)
            q1_pi = self.critic_1(states, sampled_actions)
            q2_pi = self.critic_2(states, sampled_actions)
            q_pi = torch.min(q1_pi, q2_pi)
            target_v_values = q_pi - self.alpha * log_probs

        value_pred = self.value(states)
        value_loss = nn.MSELoss()(value_pred, target_v_values)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Updating the actor
        sampled_actions, log_probs = self.actor.sample_action(states)
        log_probs = log_probs.view(-1, 1)
        q1_pi = self.critic_1(states, sampled_actions)
        q2_pi = self.critic_2(states, sampled_actions)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * log_probs - q_pi).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Updating the adaptive alpha
        if self.auto_alpha:
            alpha_loss = (self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)).mean() * (-1)
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update 
        self.soft_update(self.value, self.target_value)

        # Returns the losses needed  for controller of the adaptive learning rate
        return (
            critic_1_loss.item(),
            critic_2_loss.item(),
            value_loss.item(),
            actor_loss.item()
        )

    def soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1 - self.tau) * target_param.data
            )
