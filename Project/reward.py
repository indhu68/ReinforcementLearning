import numpy as np

# Define the grid size and actions
grid_size = 5  # Since it's a 5x5 grid
num_states = grid_size ** 2
actions = ['up', 'down', 'left', 'right', 'stay']
action_indices = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stay': 4}

# Initialize the reward matrix R
R = np.zeros((num_states, len(actions), num_states))

# Define the special reward values
REWARD_DESIRED_DIRECTION = -1  # Reward for moving in the desired direction
REWARD_OTHER = -0.5  # Reward for all other moves
REWARD_SPECIAL = -100  # Placeholder for the special reward R
REWARD_ZERO = 0  # Reward for the special states 13, 14, 15

# Define special state indices (considering Python uses 0-based indexing)
special_states_R = [2, 3, 4]  # States 3, 4, 5 in 0-based index
special_states_0 = [12, 13, 14]  # States 13, 14, 15 in 0-based index

# Helper function to get the resulting state index given row and column numbers
def get_state(row, col, grid_size):
    return row * grid_size + col

# Helper function to determine the intended resulting state from a given action
def get_intended_state(s, action, grid_size):
    row, col = divmod(s, grid_size)
    if action == 'up' :
        return get_state(row + 1, col, grid_size)
    elif action == 'down' :
        return get_state(row - 1, col, grid_size)
    elif action == 'left' :
        return get_state(row, col - 1, grid_size)
    elif action == 'right' :
        return get_state(row, col + 1, grid_size)
    return s  # For 'stay' or any action that would go off the grid

for s in range(num_states):
    for action, action_idx in action_indices.items():
        intended_state = get_intended_state(s, action, grid_size)
        for next_s in range(num_states):
            if next_s in special_states_R:
                # Assign special reward R
                R[s, action_idx, next_s] = REWARD_SPECIAL
            elif next_s in special_states_0:
                # Assign special reward 0
                R[s, action_idx, next_s] = REWARD_ZERO
            elif next_s == intended_state:
                # Assign reward for desired direction
                R[s, action_idx, next_s] = REWARD_DESIRED_DIRECTION
            else:
                # Assign reward for other directions
                R[s, action_idx, next_s] = REWARD_OTHER

