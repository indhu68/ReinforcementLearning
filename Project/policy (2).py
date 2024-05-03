import numpy as np

def get_policy(state):
    if state in [13, 14, 15]:
        return [0, 0, 0, 0, 1]
    else:
        return [0.2] * 5

def choose_action(state):
    action_probabilities = get_policy(state)
    action = np.random.choice([1, 2, 3, 4, 5], p=action_probabilities)
    return action

