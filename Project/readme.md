# Reinforcement Learning Project

## Overview

This project explores reinforcement learning algorithms through custom grid-world environments and decision-making simulations. It focuses on learning optimal actions through interaction with an environment, reward-based feedback, and iterative policy improvement.

The project contains implementations of **Monte Carlo methods, Temporal-Difference learning, Deep Q-Networks (DQN), and Modified Policy Iteration**.

Two main simulation setups are explored:

1. **Goal-Oriented Decision-Making Simulation:** A 7×7 grid-world environment in which an agent observes the movement of another agent toward one of two goals and learns how to distribute its consumption actions based on the goals' relative distances and reward feedback.
2. **Grid-World Reinforcement Learning:** A 5×5 environment with stochastic state transitions, multiple actions, and different reward values. Monte Carlo methods and policy iteration are used to estimate value functions and improve policies.

## Project Objectives

- Understand reinforcement learning through practical simulations.
- Implement state, action, transition, and reward representations.
- Explore value-function estimation and policy improvement.
- Apply Monte Carlo prediction and control methods.
- Implement on-policy and off-policy Monte Carlo approaches.
- Explore Temporal-Difference learning with function approximation.
- Develop a Deep Q-Network using PyTorch.
- Compare different approaches to learning action-selection policies.
- Study how reward design influences an agent's decisions.

## 1. Goal-Oriented Decision-Making Simulation

### Environment Description

The first simulation uses a **7×7 grid-world environment** containing two goal locations, G1 and G2.

The environment models a moving agent whose movement follows a predefined policy. A second agent, referred to as the eater, observes the mover's position and chooses how to distribute its consumption between the two goals.

The goal locations are represented by grid states:

- G1: State 37
- G2: State 49

The eater makes decisions based on the mover's position, its distance from each goal, and the reward associated with the selected action.

### State Representation

The state includes:

- Current position of the moving agent.
- Previous or current movement action.

The environment converts between grid-state numbers and two-dimensional grid coordinates to calculate positions and distances.

### Action Space

The eater has nine possible consumption strategies:

| Action | Consumption at G1 | Consumption at G2 |
|---|---:|---:|
| 1 | 100% | 0% |
| 2 | 0% | 100% |
| 3 | 50% | 50% |
| 4 | 20% | 80% |
| 5 | 60% | 40% |
| 6 | 80% | 20% |
| 7 | 40% | 60% |
| 8 | 70% | 30% |
| 9 | 30% | 70% |

These actions allow the agent to select different consumption distributions depending on the observed environment.

### Reward Calculation

The reward function considers the relative distances between the mover and the two goals.

It calculates an ideal consumption ratio using the distance to G1 and G2 and compares that ratio with the selected action.

The difference between the ideal and actual consumption ratios determines the reward.

In some implementations, the reward is mapped to discrete values, including positive rewards and penalties.

### Deception Detection and Path Prediction

Some notebook implementations explore additional mechanisms:

- Tracking the mover's action history.
- Predicting a future position based on previous movement.
- Monitoring the minimum distance to each goal.
- Detecting when the mover moves away from a goal after approaching it.
- Updating the assumed true goal after detecting a possible deceptive movement.
- Adjusting the consumption decision based on the detected behavior.

These are exploratory extensions of the environment rather than a single finalized deception-detection system.

## 2. Deep Q-Network (DQN)

The project includes a Deep Q-Network implementation using PyTorch.

DQN uses a neural network to estimate Q-values for available actions. The agent selects actions based on these estimated values and improves its decision-making through reward feedback.

### Neural Network Architecture

The implemented Q-network contains three fully connected layers:

- Input layer: 2 state features.
- First hidden layer: 16 neurons.
- Second hidden layer: 128 neurons.
- Output layer: 9 action values.

ReLU activation is applied to the hidden layers.

The output represents the estimated Q-value for each available consumption action.

### Experience Replay

A replay buffer is implemented using Python's `deque`.

It stores transitions containing:

- Current state.
- Selected action.
- Received reward.
- Next state.
- Episode completion indicator.

Random batches are sampled from the buffer during training to update the network.

### Training Process

The DQN implementation follows these steps:

1. Initialize the environment and Q-network.
2. Reset the environment at the beginning of an episode.
3. Observe the current state.
4. Select an action using an epsilon-greedy strategy.
5. Execute the action in the environment.
6. Observe the next state and reward.
7. Store the transition in the replay buffer.
8. Sample a batch of experiences.
9. Calculate the predicted and target Q-values.
10. Calculate the mean squared error loss.
11. Update the network using the Adam optimizer.
12. Repeat the process across training episodes.

The implementation uses a discount factor to incorporate future rewards into the Q-value target.

## 3. Monte Carlo Methods

Monte Carlo methods are explored in the 5×5 grid-world environment.

The environment contains 25 states and five possible actions:

- Up
- Down
- Left
- Right
- Stay

The transition model introduces stochastic movement. The intended action is followed with a probability of 0.7, while alternative movements can occur because of transition uncertainty.

The environment uses the following reward values:

| Reward | Value |
|---|---:|
| Intended movement | -1 |
| Other movement | -0.5 |
| Special states 13, 14, 15 | 0 |
| Special states 3, 4, 5 | -100 |

### First-Visit Monte Carlo Prediction

The first-visit Monte Carlo implementation estimates action-value functions by generating episodes and calculating the discounted return.

For each episode:

1. Generate a sequence of state-action-reward transitions.
2. Calculate returns by traversing the episode backward.
3. Identify the first occurrence of each state-action pair.
4. Store the return associated with that occurrence.
5. Update the corresponding Q-value using the average observed return.

The implementation also plots the evolution of a selected Q-value across episodes to examine its behavior during learning.

### First-Visit Monte Carlo Control

The project also explores Monte Carlo control using an epsilon-greedy policy.

The implementation:

- Initializes a Q-value table.
- Generates episodes using the current policy.
- Calculates returns for state-action pairs.
- Updates Q-values using average returns.
- Identifies the best action based on the estimated Q-values.
- Updates the policy using epsilon-greedy action probabilities.
- Reduces epsilon over training episodes to gradually change exploration behavior.

### Off-Policy Monte Carlo

An off-policy Monte Carlo implementation is also included.

It uses:

- A target policy.
- A behavior policy.
- Importance sampling.
- Cumulative importance weights.
- State-action value estimates.

The importance-sampling weights are used to update Q-values based on experience generated by a behavior policy while estimating the value of a target policy.

## 4. Temporal-Difference Learning

The project includes a Semi-Gradient Temporal-Difference learning implementation with linear function approximation.

Instead of maintaining a separate value for every state-action combination, the implementation represents the value function using a parameter vector.

### Feature Representation

The feature vector encodes:

- A bias term.
- The moving agent's grid position.
- The eater's selected action.

A weight vector is used to estimate the value of each state-action combination.

### Semi-Gradient TD Update

The implementation calculates:

- Current value estimate.
- Next-state value estimate.
- TD target.
- TD error.
- Gradient-based weight update.

The learning process repeatedly updates the weight vector using the observed reward and estimated future value.

The notebook also plots weight evolution across episodes to visualize how the parameters change during learning.

## 5. Modified Policy Iteration

The project includes a Modified Policy Iteration implementation for the 5×5 grid-world environment.

The algorithm combines policy evaluation and policy improvement.

### Policy Evaluation

The value function is updated iteratively using the current policy and the environment's transition and reward models.

The implementation uses:

- A convergence threshold (`theta`).
- A limited number of policy-evaluation iterations (`N`).
- A value function initialized to zeros.

### Policy Improvement

After policy evaluation, the algorithm calculates action values for each state and selects the action with the highest estimated value.

The policy is updated accordingly.

The process continues until the policy becomes stable or the convergence condition is satisfied.

The implementation reports:

- Estimated value function.
- Updated policy.
- Number of iterations.
- Execution time.

## Technologies Used

- **Python** – Implementation of environments, algorithms, and simulations.
- **NumPy** – Numerical computation, arrays, and value-function calculations.
- **PyTorch** – Neural network implementation and DQN training.
- **Matplotlib** – Visualization of learning progress and Q-value convergence.
- **Collections (deque)** – Experience replay buffer.
- **Random** – Random action selection and exploration.

## Key Reinforcement Learning Concepts Explored

| Concept | Application |
|---|---|
| State and Action Representation | Modeling grid positions and agent decisions |
| Reward Design | Evaluating consumption strategies and grid movements |
| State Transition | Modeling environment movement and uncertainty |
| Value Function | Estimating expected future rewards |
| Q-Value Estimation | Evaluating state-action combinations |
| Monte Carlo Prediction | Estimating values from complete episodes |
| Monte Carlo Control | Improving policies using sampled returns |
| Epsilon-Greedy Policy | Balancing exploration and exploitation |
| Importance Sampling | Off-policy Monte Carlo estimation |
| Temporal-Difference Learning | Updating estimates using bootstrapping |
| Function Approximation | Representing value estimates using weights |
| Deep Q-Network | Learning action values using a neural network |
| Policy Iteration | Alternating evaluation and improvement |

## Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/indhu68/ReinforcementLearning.git
cd ReinforcementLearning/Project
```

### 2. Install Dependencies

```bash
pip install numpy matplotlib torch jupyter
```

### 3. Run Python Implementations

For example:

```bash
python environment.py
python reward.py
python MC_Qestimate.py
python MC_Uestimate.py
python MC_mu_estimation.py
python offpolicy_MC_mu_estimate.py
python policy_value_eval.py
```

Some scripts import other modules from the same directory. Keep the related Python files together when running them.

### 4. Run Jupyter Notebooks

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks in the `Project` directory to explore the simulations and algorithm implementations.

The DQN notebooks include the neural network, replay buffer, environment interaction, and training logic.

## Learning Outcomes

This project provides practical experience with:

- Designing custom reinforcement learning environments.
- Defining states, actions, rewards, and transition behavior.
- Implementing Monte Carlo prediction and control.
- Understanding on-policy and off-policy learning.
- Applying importance sampling.
- Implementing Temporal-Difference learning.
- Using linear function approximation.
- Building and training a Deep Q-Network with PyTorch.
- Understanding policy evaluation and policy improvement.
- Visualizing learning behavior and value estimates.

## Conclusion

This project explores multiple reinforcement learning approaches through custom grid-world simulations and goal-oriented decision-making scenarios.

By implementing Monte Carlo methods, Temporal-Difference learning, Deep Q-Networks, and Modified Policy Iteration, it provides a practical foundation for understanding how agents learn action-selection strategies from environmental interaction and reward feedback.

The different implementations also demonstrate how state representation, reward design, exploration strategies, and value estimation influence reinforcement learning behavior.
