import time

import numpy as np
import environment
import reward
import time

# Parameters for the modified policy iteration algorithm
theta = 0.0000000001  # Small threshold parameter for convergence
N = 10  # Number of iterations for policy evaluation

# State and action space
num_states = 25  # 5x5 grid world
num_actions = 5  # up, down, left, right, stay

# Initialize V(s) and policy arbitrarily
V = np.zeros(num_states)  # Value function
policy = np.ones((num_states, num_actions))  # Policy
def modified_policy_iteration(V, policy, P, R, theta, N, num_states, num_actions):
    start_time = time.time()  # Start timing
    iteration_count = 0  # Initialize iteration counter

    while True:
        delta = 0
        for i in range(N):  # Perform up to N iterations
            iteration_count += 1  # Increment iteration count each time
            V_old = V.copy()
            for s in range(num_states):
                V[s] = sum(
                    policy[s, a]* P[s, a, s_prime] * (R[s, a, s_prime] + V_old[s_prime]) for a in range(num_actions) for s_prime in
                    range(num_states))
                delta = max(delta, abs(V[s] - V_old[s]))

            # Check for convergence within this N loop
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for s in range(num_states):
            old_action = np.argmax(policy[s])
            new_action_values = [
                sum(P[s, a, s_prime] * (R[s, a, s_prime] + V[s_prime]) for s_prime in range(num_states)) for a in
                range(num_actions)]
            new_action = np.argmax(new_action_values)
            if new_action != old_action:
                policy_stable = False
            policy[s] = np.eye(num_actions)[new_action]

        # If policy is stable or value change is less than theta, then stop
        if policy_stable or delta < theta:
            break

    end_time = time.time()  # End timing
    time_taken = end_time - start_time  # Calculate total time taken
    return V, policy, iteration_count, time_taken



# Assume P and R are provided by the environment and reward modules
P = environment.P
R = reward.R

# Run the modified policy iteration algorithm
V, policy, iteration_count, time_taken = modified_policy_iteration(V, policy, P, R, theta, N, num_states, num_actions)

# Print the results
print(f"Value Function: {V}")
print(f"Policy: {policy}")
print(f"Number of iterations: {iteration_count}")
print(f"Total time taken: {time_taken} seconds")
