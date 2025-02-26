from argparse import ArgumentParser
from environment import HockeyEnvWrapper
from sac import SACAgent
from train import train_sac, evaluate_agent
import hockey.hockey_env as h_env
from hockey.hockey_env import HockeyEnv_BasicOpponent

import os
import torch

def parse_arguments():
    
    parser = ArgumentParser()

    # Training params wit h default options
    parser.add_argument('--mode', choices=['normal', 'shooting', 'defense', 'weak_opponent', 'strong_opponent'], default='normal',
                        help='Training mode: normal, shooting, defense, or opponent')
    parser.add_argument('--max_episodes', type=int, default=1000, help='Maximum number of episodes for training')
    parser.add_argument('--max_steps', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes for evaluation')
    parser.add_argument('--eval_interval', type=int, default=50, help='Interval for evaluation during training')
    parser.add_argument('--render', action='store_true', help='Render the environment during training/evaluation')
    parser.add_argument('--save_path', type=str, default='./models', help='Directory to save the trained models')
    parser.add_argument('--load_model', type=str, default=None, help='Name of the .pth file to load (partial or none)')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use for training')
    parser.add_argument('--checkpoint_mode', choices=['actor', 'none'], default='none',
                        help='actor=load only actor weights (skip training), none=no loading')

    return parser.parse_args()


def main():
    
    #Main script to train the SAC agent for hockey game modes.
    
    args = parse_arguments()

    # Ensures save directory exists
    os.makedirs(args.save_path, exist_ok=True)

    # Lists the modes possible
    modes = {
        'normal': h_env.Mode.NORMAL,
        'shooting': h_env.Mode.TRAIN_SHOOTING,
        'defense': h_env.Mode.TRAIN_DEFENSE
    }

    # Defining the modes to choose froom
    if args.mode == "weak_opponent":
        #env = h_env.BasicOpponent(weak=True)
        env = HockeyEnv_BasicOpponent(mode=h_env.Mode.NORMAL, weak_opponent=True)

        env_mode = h_env.Mode.NORMAL  # Force normal mode
    elif args.mode == "strong_opponent":
        env = HockeyEnv_BasicOpponent(mode=h_env.Mode.NORMAL, weak_opponent=False)
        env_mode = h_env.Mode.NORMAL  # Force normal mode
    else:
        if args.mode not in modes:
            raise ValueError(f"Invalid mode: {args.mode}. Use one of: normal, shooting, defense, weak_opponent, strong_opponent.")
        env_mode = modes[args.mode]
        env = HockeyEnv_BasicOpponent(mode=h_env.Mode.NORMAL, weak_opponent=True)
 

    # Code to continue training (skipped if checkpoint_mode='actor', then only weights are loaded to play against other agents)
    agent = train_sac(
        mode=env_mode,
        max_episodes=args.max_episodes,
        max_steps=args.max_steps,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        save_path=args.save_path,
        load_model=args.load_model,
        device=args.device,
        render=args.render,
        checkpoint_mode=args.checkpoint_mode
    )

    # Evaluating agent's performance and displaying it
    if args.eval_episodes > 0:
        print(f"Evaluating the agent for {args.eval_episodes} episodes...")
        eval_reward, wins, draws, losses = evaluate_agent(env, agent, args.eval_episodes, args.max_steps)
        print(f"Average evaluation reward: {eval_reward:.2f}, Wins: {wins}, Draws: {draws}, Losses: {losses}")

    env.close()

if __name__ == "__main__":
    main()
