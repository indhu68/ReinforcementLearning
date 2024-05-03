# Correcting the transition probability matrix generation logic for the grid world
# Constants for the grid and actions
import numpy as np
num_states = 25  # 5x5 grid
actions = ['up', 'down', 'left', 'right', 'stay']
action_indices = {'up': 0, 'down': 1, 'left': 2, 'right': 3, 'stay': 4}
num_actions = 5  # up, down, left, right, stay
grid_size = 5  # Grid is 5x5
# Initialize the transition probabilities matrix P with zeros
P = np.zeros((num_states, num_actions, num_states))

# Define the actions in terms of row and column offsets
action_to_offset = {
    'up': (1, 0),
    'down': (-1, 0),
    'left': (0, -1),
    'right': (0, 1),
    'stay': (0, 0),
}

# Populate the transition probabilities
for s in range(num_states):
    current_row, current_col = divmod(s, grid_size)

    for action, (row_offset, col_offset) in action_to_offset.items():
        if action == 'stay':
            # If the action is 'stay', set the transition probability to the current state to 1
            P[s, action_indices[action], s] = 1
            continue

        # Calculate the row and column of the new state after the action
        new_row = current_row + row_offset
        new_col = current_col + col_offset

        # If the new state is within bounds, set the transition probability
        if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
            new_state = new_row * grid_size + new_col
            P[s, action_indices[action], new_state] = 0.7
        else:
            # Reflective boundary; stay in the current state
            P[s, action_indices[action], s] += 0.7

        # For orthogonal directions, the transitions depend on the specific action
        if action in ['up', 'down']:
            # For 'up' and 'down', the orthogonal directions are 'left' and 'right'
            for orth_action in ['left', 'right']:
                orth_row_offset, orth_col_offset = action_to_offset[orth_action]
                orth_new_row = current_row + orth_row_offset
                orth_new_col = current_col + orth_col_offset

                if 0 <= orth_new_col < grid_size:
                    orth_new_state = current_row * grid_size + orth_new_col
                    P[s, action_indices[action], orth_new_state] = 0.1
                else:
                    # Reflective boundary for orthogonal direction
                    P[s, action_indices[action], s] += 0.1

        elif action in ['left', 'right']:
            # For 'left' and 'right', the orthogonal directions are 'up' and 'down'
            for orth_action in ['up', 'down']:
                orth_row_offset, orth_col_offset = action_to_offset[orth_action]
                orth_new_row = current_row + orth_row_offset
                orth_new_col = current_col + orth_col_offset

                if 0 <= orth_new_row < grid_size:
                    orth_new_state = orth_new_row * grid_size + current_col
                    P[s, action_indices[action], orth_new_state] = 0.1
                else:
                    # Reflective boundary for orthogonal direction
                    P[s, action_indices[action], s] += 0.1
        else:
            # The 'stay' action probability is already set to 1, no need to add orthogonal transitions
            pass

        # Ensure probabilities sum to 1 for each state-action pair
        probability_sum = P[s, action_indices[action], :].sum()
        if probability_sum < 1:
            P[s, action_indices[action], s] += (1 - probability_sum)
        elif probability_sum > 1:
            # In case sum of probabilities exceeds 1 due to reflective boundaries, normalize it
            P[s, action_indices[action], :] /= probability_sum

# Now let's check the transition probabilities for state 1 taking action 'up'
print(P[0, action_indices['down']])
