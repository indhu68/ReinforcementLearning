import environment
import policy
import numpy as np
import matplotlib.pyplot as plt
Q_plot = []
def generate_episode(max_steps=100):
    """Generates an episode following the policy defined in policy.py."""
    episode = []
    state = np.random.choice(environment.States)# Randomly select a starting state
    action0 = np.random.choice(environment.Actions)
    for i in range(max_steps):
        if i == 0:
            action = action0
        else:
            action = policy.choose_action(state)
        next_state, reward_index = environment.get_transition_probability(state, action)
        reward = environment.Rewards[reward_index]
        episode.append((state, action, reward))
        state = next_state
    return episode



def first_visit_mc_prediction(total_episodes, gamma=0.99, max_steps=150):
    # Step 1: Initialize Q^mu(s, a)
    Q = np.full((environment.Total_states, environment.Total_actions), np.nan)
    returns = {}  # Dictionary to store returns for each state-action pair

    for _ in range(total_episodes):
        # Step 2: Generate an episode
        episode = generate_episode(max_steps=max_steps)
        G = 0  # Initialize the return

        # Step 3: Update Q^mu estimates
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward  # Update the return

            # Check if this is the first visit to the state-action pair in the episode
            if (state, action) not in [(s, a) for (s, a, _) in episode[:t]]:
                if (state, action) not in returns:
                    returns[(state, action)] = []
                returns[(state, action)].append(G)
                Q[state - 1][action - 1] = np.mean(returns[(state, action)])
        Q_plot.append(Q[0][3])

    return Q


# Example usage
total_episodes = 19500
Q_mu = first_visit_mc_prediction(total_episodes)
print("Q^mu estimates:")
print(Q_mu)


plt.figure(figsize=(10, 6))
plt.plot(Q_plot, label="Qμ(s=1, a='right')")
plt.xlabel('Episodes')
plt.ylabel('Qμ(s=1, a=\'right\')')
plt.title('Convergence of Qμ(s=1, a=\'right\') over episodes')
plt.legend()
plt.grid(True)
plt.show()


