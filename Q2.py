import numpy as np
import matplotlib.pyplot as plt

# Define the payoff matrix
payoff_matrix = [
    [(0, 0), (0, 1), (1, 0)],  # Row 0 - Rock
    [(1, 0), (0, 0), (0, 1)],  # Row 1 - Paper
    [(0, 1), (1, 0), (0, 0)]   # Row 2 - Scissors
]

# Never leave all the values as 0 in either of these
counts_p1 = np.array([0, 1, 0])  # Player 1's counts of Player 2's actions
counts_p2 = np.array([0, 1, 0])  # Player 2's counts of Player 1's actions

# Number of iterations
N = 1000

# History to track empirical distributions
history_p1 = [counts_p2 / counts_p2.sum()]  # Player 1's mixed strategy
history_p2 = [counts_p1 / counts_p1.sum()]  # Player 2's mixed strategy

for _ in range(N):
    # Player 1 computes best response to Player 2's empirical distribution
    p2 = counts_p1 / counts_p1.sum()
    utilities_p1 = [sum(p2[j] * payoff_matrix[i][j][0] for j in range(3)) for i in range(3)]
    max_util_p1 = max(utilities_p1)
    best_actions_p1 = [i for i, u in enumerate(utilities_p1) if u == max_util_p1]
    action_p1 = np.random.choice(best_actions_p1)  # Pick random in case of ties
     
    # Player 2 computes best response to Player 1's empirical distribution
    p1 = counts_p2 / counts_p2.sum()
    utilities_p2 = [sum(p1[i] * payoff_matrix[i][j][1] for i in range(3)) for j in range(3)]
    max_util_p2 = max(utilities_p2)
    best_actions_p2 = [j for j, u in enumerate(utilities_p2) if u == max_util_p2]
    action_p2 = np.random.choice(best_actions_p2) # Pick random in case of ties
    
    # Uncomment the section below for breaking the ties as R>P>S
    
    action_p1 = min(best_actions_p1)  # Favor R > P > S for 1st player
    action_p2 = min(best_actions_p2)  # Favor R > P > S for 2nd player
    
    
    # Update counts
    counts_p1[action_p2] += 1
    counts_p2[action_p1] += 1
    
    # Record current empirical distributions
    history_p1.append(counts_p2 / counts_p2.sum())
    history_p2.append(counts_p1 / counts_p1.sum())

# Convert to numpy arrays for plotting
history_p1 = np.array(history_p1)
history_p2 = np.array(history_p2)

# Plotting the results
plt.figure(figsize=(12, 6))

# Player 1's strategies
plt.subplot(1, 2, 1)
for i in range(3):
    if(i==0): 
        action = "Rock"
    elif i == 1:
        action = "Paper"
    else:
        action = "Scissor"
    plt.plot(history_p1[:, i], label=action)
plt.title("Player 1's Strategy Over Time")
plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.legend()

# Player 2's strategies
plt.subplot(1, 2, 2)
for i in range(3):
    if(i==0): 
        action = "Rock"
    elif i == 1:
        action = "Paper"
    else:
        action = "Scissor"
    plt.plot(history_p2[:, i], label=action)
plt.title("Player 2's Strategy Over Time")
plt.xlabel('Iteration')
plt.ylabel('Probability')
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()
