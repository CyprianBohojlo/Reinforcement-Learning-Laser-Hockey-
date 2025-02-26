# Laser Hockey Reinforcement Learning Project   
The final project about the development of a Reinforcement Learning agent for a laser hockey game in the course Reinforcement Learning WiSe 24/25 at the University of Tübingen, Germany.
## Install   
The base environment can be installed using 
```
python3 -m pip install git+https://github.com/martius-lab/hockey-env.git

or add the following line to your pip file

hockey = {editable = true, git = "https://git@github.com/martius-lab/hockey-env.git"}
```
Additionally, we would recommend you to install   
```
pip install swig # when an error occurs while installing box2d-py
pip install torch
```

```
# Setup
# Linter and Formatter
pip install ruff
pip install pre-commit
```

## Laser Hockey game
The Hockey environment is implemented using the Gymnasium API
(formerly Open AI Gym, https://gymnasium.farama.org/, which features a custom environment developed by the
Martius Lab. The environment simulates a two-player hockey game in which agents compete to score goals against
each other. It presents numerous challenges, including continuous state spaces, complex dynamics, and the need
for both defensive and offensive strategies.
## Algorithm
This project implements the **Soft Actor-Critic (SAC)** algorithm to train an agent to play a laser hockey game. SAC is an off-policy reinforcement learning algorithm that optimizes for both reward maximization and entropy, encouraging exploration.

The key components of the SAC implementation in this project include:

- **Environment Wrapper (`environment.py`)**  
  A wrapper around the `hockey-env` environment that manages state observations, actions, and reward shaping.

- **Neural Networks (`networks.py`)**  
  - **Actor Network**: Learns a policy to output actions based on states and policy.
  - **Critic Networks**: Two critics are used to estimate Q-values and mitigate overestimation bias.
  - **Value Network**: Estimates the expected return from a given state.

- **Replay Buffer (`replay.py`)**  
  A memory buffer storing past experiences to enable batch updates and break correlation in training data.

- **SAC Agent (`sac.py`)**  
  Implements the Soft Actor-Critic agent, including:
  - Policy learning with reparameterization trick.
  - Twin Q-learning for stability.
  - Adaptive temperature alpha tuning to balance exploration and exploitation.

- **Training Pipeline (`train.py`)**  
  - Implements training loops, evaluation, logs saving, and model saving.
  - Uses an **adaptive learning rate controller** to auto-tune learning rate automatically.

- **Training (`training_sac_agent.py`)**  
  - Provides command-line arguments for training and evaluation.
  - Supports multiple training modes (normal, shooting, defense, different opponent modes).
  - Handles model saving, resuming training, and performance evaluation.

## Usage

To train the SAC agent, the user can run the following command:

```bash
python training_sac_agent.py --mode weak_opponent --max_episodes 20000 --max_steps 500 \
        --save_path ./models --checkpoint_mode none --device cuda


