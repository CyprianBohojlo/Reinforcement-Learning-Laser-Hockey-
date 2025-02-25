import os
import numpy as np
import torch
from environment import HockeyEnvWrapper
from replay import ReplayBuffer
from sac import SACAgent
import hockey.hockey_env as h_env
import matplotlib.pyplot as plt
import datetime 

BEST_EVAL_REWARD = float("-inf")

# Controller for adaptive learning rate
class AdaptiveLRController(torch.nn.Module):
    
    # Controller computes a scaling factor for the adaptive learning rate, using a simple one-layer policy network and a REINFORCE update.
    # This is a simplified version of the paper's approach as they used PPO
    

    def __init__(self):
        super(AdaptiveLRController, self).__init__()
        # Defining the netowrk for the controller, the input size should be 5
        self.hidden = torch.nn.Linear(5, 16)  
        self.output = torch.nn.Linear(16, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        # Storing transitions for each episode
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        
        h = torch.relu(self.hidden(x))
        scale = torch.sigmoid(self.output(h))  
        return scale

    def select_action(self, state_vec):
        
        #Forward pass that returns the scale factor plus the log_prob for REINFORCE.
        
        scale = self.forward(state_vec)

        #Creates a distribution around it for a REINFORCE approach update
        dist = torch.distributions.Normal(scale, 0.01)  
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return torch.clamp(action, 0.0, 2.0), log_prob  # clipping scale to [0, 2] for stability, if not it sometimes explodes

    def store_outcome(self, log_prob, reward):
        self.saved_log_probs.append(log_prob)
        self.rewards.append(reward)

    def finish_episode(self):
        
        # REINFORCE update after each episode
        returns = self.rewards  

        loss = 0
        for log_prob, r in zip(self.saved_log_probs, returns):
            loss += -log_prob * r  # REINFORCE objective as described in the paper and report

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clearing buffers
        self.saved_log_probs = []
        self.rewards = []


def train_sac(
    mode=h_env.Mode.NORMAL,
    max_episodes=1000,
    max_steps=500,
    eval_interval=10,
    eval_episodes=10,
    save_path='./models',
    load_model=None,
    render=False,
    device=None,
    checkpoint_mode='none'
):
    global BEST_EVAL_REWARD

    
    # Open a file in write mode to print the logs
    log_file = open("score_logs.txt", "w")
    

    # Hyperparameters
    replay_buffer_size = 10000
    batch_size = 128
    action_bound = 1.0
    learning_rate = 1e-4

    total_wins = 0
    total_draws = 0
    total_losses = 0

    device = device or torch.device('cpu')

    # Initializing the environment 
    env = HockeyEnvWrapper(mode=mode, render=render)
    state_dim = env.state_dim
    action_dim = env.action_dim

    # Creating replay buffer 
    replay_buffer = ReplayBuffer(
        max_size=replay_buffer_size,
        state_dim=state_dim,
        action_dim=action_dim
    )

    # Creating the SAC agent
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        action_bound=action_bound,
        replay_buffer=replay_buffer,
        learning_rate=learning_rate,
        gamma=0.99,
        tau=0.005,
        alpha=0.05,
        device=device,
        target_entropy=-action_dim  # None to disable adaptive alpha tuning 
    )

    if checkpoint_mode == 'actor' and load_model:
        model_path = os.path.join(save_path, load_model)
        if os.path.exists(model_path):
            agent.actor.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f"Loaded ONLY ACTOR weights from {model_path}")
            log_file.write(f"Loaded ONLY ACTOR weights from {model_path}\n")  
        else:
            print(f"Warning: Specified model '{model_path}' not found. Starting from scratch.")
            log_file.write(f"Warning: Specified model '{model_path}' not found. Starting from scratch.\n")  
    else:
        print("No checkpoint is loaded; starting from scratch.")
        log_file.write("No checkpoint is loaded; starting from scratch.\n")  

    def maybe_switch_mode(episode, max_ep, env_wrapper):
        
        # Dynamically change env_wrapper.env.mode 
        # Splits training into:
        # - 0%-25% = TRAIN_SHOOTING
        # - 25%-50% = TRAIN_DEFENSE
        # - 50%-100% = NORMAL
        
        if mode not in [h_env.Mode.NORMAL, "weak_opponent", "strong_opponent"]: 
            return
        fraction = episode / float(max_ep)
        if fraction < 0.25:
            env_wrapper.env.mode = h_env.Mode.TRAIN_SHOOTING
        elif fraction < 0.5:
            env_wrapper.env.mode = h_env.Mode.TRAIN_DEFENSE
        else:
            env_wrapper.env.mode = h_env.Mode.NORMAL

    rewards_history = []
    eval_rewards_history = []

    # Create adaptive learning rate controller
    lr_controller = AdaptiveLRController().to(device)

    for episode in range(max_episodes):
        maybe_switch_mode(episode, max_episodes, env)

        state, _ = env.reset()
        episode_reward = 0.0

        # At the start of each episode, pick a scaling factor
        if episode == 0:
            last_c1, last_c2, last_vl, last_al = 0, 0, 0, 0
        input_vec = torch.FloatTensor([
            [
                agent.current_lr,
                last_c1,
                last_c2,
                last_vl,
                last_al
            ]
        ]).to(device)

        scale_factor, log_prob = lr_controller.select_action(input_vec)
        new_lr = float(agent.current_lr * scale_factor.item())
        agent.set_learning_rate(max(new_lr, 1e-8))

        sum_c1, sum_c2, sum_vl, sum_al = 0, 0, 0, 0
        updates_this_episode = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _, info = env.step(action)

            replay_buffer.add_transition(state, action, reward, next_state, done)

            if replay_buffer.get_size() > batch_size:
                c1l, c2l, vl, al = agent.update(batch_size)
                sum_c1 += c1l
                sum_c2 += c2l
                sum_vl += vl
                sum_al += al
                updates_this_episode += 1

            state = next_state
            episode_reward += reward
            if done:
                if "winner" in info:
                    if info["winner"] == 1:
                        total_wins += 1
                    elif info["winner"] == 0:
                        total_draws += 1
                    elif info["winner"] == -1:
                        total_losses += 1
                break

        rewards_history.append(episode_reward)

        if updates_this_episode > 0:
            avg_c1 = sum_c1 / updates_this_episode
            avg_c2 = sum_c2 / updates_this_episode
            avg_vl = sum_vl / updates_this_episode
            avg_al = sum_al / updates_this_episode
        else:
            avg_c1, avg_c2, avg_vl, avg_al = 0, 0, 0, 0

        last_c1, last_c2, last_vl, last_al = avg_c1, avg_c2, avg_vl, avg_al
        total_loss = (avg_c1 + avg_c2 + avg_vl + avg_al)
        reward_for_lr = -total_loss
        lr_controller.store_outcome(log_prob, reward_for_lr)
        lr_controller.finish_episode()

        if episode % eval_interval == 0 or episode == max_episodes - 1:
            eval_reward, wins, draws, losses = evaluate_agent(env, agent, eval_episodes, max_steps)
            eval_rewards_history.append(eval_reward)
            # Printing the results
            print(
                f"Episode {episode}, Training Reward: {episode_reward:.2f}, Eval Reward: {eval_reward:.2f}, "
                f"Wins: {wins}, Draws: {draws}, Losses: {losses}"
            )
            # Writing the same results into the log file
            log_file.write(
                f"Episode {episode}, Training Reward: {episode_reward:.2f}, Eval Reward: {eval_reward:.2f}, "
                f"Wins: {wins}, Draws: {draws}, Losses: {losses}\n"
            )

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            model_path = os.path.join(save_path, f"sac_agent_episode_{episode}_{timestamp}.pth")
            torch.save(agent.actor.state_dict(), model_path)
            print(f"Model saved at {model_path}")
            log_file.write(f"Model saved at {model_path}\n")

            if eval_reward > BEST_EVAL_REWARD:
                BEST_EVAL_REWARD = eval_reward
                best_model_path = os.path.join(save_path, f"best_sac_agent_{timestamp}.pth")
                torch.save(agent.actor.state_dict(), best_model_path)
                print(f"New best model saved with evaluation reward {BEST_EVAL_REWARD:.2f}")
                log_file.write(f"New best model saved with evaluation reward {BEST_EVAL_REWARD:.2f}\n")

    final_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    final_model_path = os.path.join(save_path, f"sac_agent_{mode}_{final_timestamp}.pth")
    torch.save(agent.actor.state_dict(), final_model_path)
    print(f"Final model saved at {final_model_path}")
    log_file.write(f"Final model saved at {final_model_path}\n")

    env.close()
    print(f"Final Totals => Wins: {total_wins}, Draws: {total_draws}, Losses: {total_losses}")
    log_file.write(f"Final Totals => Wins: {total_wins}, Draws: {total_draws}, Losses: {total_losses}\n")

    # Close the file
    log_file.close()
    return agent


def evaluate_agent(env, agent, num_episodes=5, max_steps=500):
   
    total_reward = 0.0
    wins = 0
    draws = 0
    losses = 0

    for _ in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = agent.select_action(state, evaluate=True)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                if "winner" in info:
                    if info["winner"] == 1:
                        wins += 1
                    elif info["winner"] == 0:
                        draws += 1
                    elif info["winner"] == -1:
                        losses += 1
                break
        total_reward += episode_reward

    avg_reward = total_reward / num_episodes
    return avg_reward, wins, draws, losses

if __name__ == "__main__":
    train_sac()
