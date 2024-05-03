import numpy as np
from environment import *
def generate_episode(policy, max_steps=500):
    episode = []
    state = np.random.choice(States)  # Start from a random state
    action= np.random.choice(Actions)
    for _ in range(max_steps):
        next_state, reward_index = get_transition_probability(state, action)
        reward = Rewards[reward_index]
        episode.append((state, action, reward))
        state = next_state
        action = policy[state - 1]
    return episode

def monte_carlo_first_visit(epsilon, episodes):
    Q = np.full((Total_states, Total_actions), -9000.2)
    returns = [[[] for _ in range(Total_actions)] for _ in range(Total_states)]
    policy = np.random.choice(Actions, Total_states)  # Random initial policy
    gamma = 0.99
    for j in range(episodes):
        episode = generate_episode(policy)
        G = 0
        epsilon = 1/(j+1)
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if not (state, action) in [(x[0], x[1]) for x in episode[:t]]:
                returns[state - 1][action - 1].append(G)
                Q[state - 1][action - 1] = np.mean(returns[state - 1][action - 1])
                A_star = np.argmax(Q[state - 1])
                if np.random.rand() < epsilon:
                    policy[state - 1] = np.random.choice(Actions)  # Explore
                else:
                    policy[state - 1] = A_star + 1  # Exploit

    return policy, Q


final_policy, final_Q = monte_carlo_first_visit(epsilon=0.20, episodes=7000)
print("Final policy:")
print(final_policy)
print("Final Q-values:")
print(final_Q)
