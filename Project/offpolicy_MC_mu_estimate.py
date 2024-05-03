import numpy as np
import environment
import policy

ep = 0.00000001 # Small constant to avoid division by zero

def generate_episode(max_steps):
    episode = []
    state = np.random.choice(environment.States)  # Randomly select a starting state
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

def offpolicy_mc_prediction(total_episodes, gamma=0.99, max_steps=170):
    Q =  np.zeros((environment.Total_states, environment.Total_actions))
    C = np.zeros((environment.Total_states, environment.Total_actions))
    mu = np.ones((environment.Total_states, environment.Total_actions)) / environment.Total_actions
    b = [policy.get_policy(state) for state in environment.States]
    for episode_num in range(total_episodes):
        epsilon = max(0, 1 - episode_num / (total_episodes * 0.8))
        #epsilon = 1/(episode_num+1)
        #epsilon = 0.1
        episode = generate_episode(max_steps)
        G = 0
        W = 1
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            state_idx, action_idx = state - 1, action - 1
            G = gamma * G + reward
            C[state_idx][action_idx] += W
            Q[state_idx][action_idx] += (W / C[state_idx][action_idx]) * (G - Q[state_idx][action_idx])
            a_star = np.argmax(Q[state_idx])
            for a in range(environment.Total_actions):
                if a == a_star:
                    mu[state_idx][a] = 1 - epsilon + (epsilon / environment.Total_actions)
                else:
                    mu[state_idx][a] = epsilon / environment.Total_actions
            W *= mu[state_idx][action_idx] / (b[state_idx][action_idx] + ep)
            if W == 0:
                break

    return mu,Q


offpolicy = offpolicy_mc_prediction(30000)
print(offpolicy[0])
print(offpolicy[1])