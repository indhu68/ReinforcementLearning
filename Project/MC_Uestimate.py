import numpy as np
import environment
from collections import defaultdict

def generate_episode(max_steps, policy):
    episode = []
    state = np.random.choice(environment.States)  # Randomly select a starting state
    #state = np.random.choice([12,13, 14, 15])
    action = np.random.choice(environment.Actions)
    for i in range(max_steps):
        next_state, reward_index = environment.get_transition_probability(state, action)
        reward = environment.Rewards[reward_index]
        episode.append((state, action, reward))
        state = next_state
        action = choose_action(state, policy)
    return episode

def choose_action(state, policy):
    action_probabilities = policy[state - 1]
    action = np.random.choice([1, 2, 3, 4, 5], p=action_probabilities)
    return action

def first_visit_mc_epsilon_greedy(total_episodes, gamma=0.99, max_steps=1000, epsilon_start=1.0, epsilon_decay=0.9, epsilon_min=0.1):
    Q = np.zeros((environment.Total_states, environment.Total_actions))
    mu = np.ones((environment.Total_states, environment.Total_actions)) / environment.Total_actions
    epsilon = epsilon_start
    stateAction_reward = defaultdict(list)
    for episode_num in range(total_episodes):
        episode = generate_episode(max_steps=max_steps, policy=mu)
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if not (state, action) in [(x[0], x[1]) for x in episode[:t]]:
                stateAction_reward[(state, action)].append(G)
                Q[state - 1][action - 1] = np.mean(stateAction_reward[(state, action)])
                A_star = np.argmax(Q[state - 1])
                for a in range(environment.Total_actions):
                    if a == A_star:
                        mu[state - 1][a] = 1 - epsilon + (epsilon / environment.Total_actions)
                    else:
                        mu[state - 1][a] = epsilon / environment.Total_actions

        epsilon = max(epsilon * epsilon_decay, epsilon_min)

    return mu, Q
total_episodes = 1000
optimal_policy_mu, Q = first_visit_mc_epsilon_greedy(total_episodes)
print("Optimal policy estimates:")
print(optimal_policy_mu)
print("Q Value estimates:")
print(Q)
