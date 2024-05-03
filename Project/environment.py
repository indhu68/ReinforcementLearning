import numpy as np

N = 5
Total_states = N ** 2
Total_actions = 5
Total_rewards = 4
States = np.arange(1, Total_states + 1)
Actions = np.arange(1, Total_actions + 1)
Rewards = [-1, -0.5, 0, -100]

# Initialize P(s', r | s, a)
P = np.zeros((Total_states, Total_rewards, Total_states, Total_actions))

def get_transition_probability(s, a):
    r = np.random.rand()
    s_prime = s
    reward_index = 1  # Default reward is -0.5

    if a == 1:  # Up
        if r <= 0.7:
            if s <= Total_states - N:
                s_prime = s + N
                reward_index = 0
            else:
                s_prime = s
                reward_index = 1
        elif r <= 0.8:
            s_prime = s + 1 if s % N != 0 else s
        elif r <= 0.9:
            s_prime = s - 1 if s % N != 1 else s
    elif a == 2:  # Down
        if r <= 0.7:
            if s > N:
                s_prime = s - N
                reward_index = 0
            else:
                s_prime = s
                reward_index = 1
        elif r <= 0.8:
            s_prime = s + 1 if s % N != 0 else s
        elif r <= 0.9:
            s_prime = s - 1 if s % N != 1 else s
    elif a == 3:  # Left
        if r <= 0.7:
            if s % N != 1:
                s_prime = s - 1
                reward_index = 0
            else:
                s_prime = s
                reward_index = 1
        elif r <= 0.8:
            s_prime = s + N if s <= Total_states - N else s
        elif r <= 0.9:
            s_prime = s - N if s > N else s
    elif a == 4:  # Right
        if r <= 0.7:
            if s % N != 0:
                s_prime = s + 1
                reward_index = 0
            else:
                s_prime = s
                reward_index = 1
        elif r <= 0.8:
            s_prime = s + N if s <= Total_states - N else s
        elif r <= 0.9:
            s_prime = s - N if s > N else s
    else:  # Stay
        reward_index = 0
        s_prime = s

    # Special states with different rewards
    if s_prime in [3, 4, 5]:
        reward_index = 3
    elif s_prime in [13, 14, 15]:
        reward_index = 2

    return s_prime, reward_index

