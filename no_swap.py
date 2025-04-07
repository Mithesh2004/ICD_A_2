import numpy as np
import matplotlib.pyplot as plt

# Configuration
NUM_ACTIONS = 3
T = 2000
ETA = 0.1

U = np.array([
    [(0, 0), (2,3 ), (0, 0)],
    [(1, 2), (0, 0), (2, 1)],
    [(0, 0), (3, 2), (0, 0)]
])

def normalize(weights):
    total = np.sum(weights)
    return weights / total if total > 0 else np.full(len(weights), 1.0 / len(weights))

def choose_action(strategy):
    return np.random.choice(len(strategy), p=strategy)

def update_weights(swap_regret, eta):
    new_weights = np.zeros(NUM_ACTIONS)
    for i in range(NUM_ACTIONS):
        regret_sum = sum(max(0, swap_regret[i][j]) for j in range(NUM_ACTIONS) if i != j)
        new_weights[i] = np.exp(eta * regret_sum)
    return new_weights

def plot_heatmap(ax, joint_distribution, step):
    ax.clear()
    heatmap = ax.imshow(joint_distribution, cmap='viridis', vmin=0, vmax=1)
    ax.set_title(f'Joint Probability Distribution (Step {step})')
    ax.set_xlabel('Player B Actions')
    ax.set_ylabel('Player A Actions')
    ax.set_xticks(np.arange(NUM_ACTIONS))
    ax.set_yticks(np.arange(NUM_ACTIONS))
    for i in range(NUM_ACTIONS):
        for j in range(NUM_ACTIONS):
            ax.text(j, i, f"{joint_distribution[i][j]:.2f}", ha='center', va='center', color='white')
    plt.pause(0.01)

def run_simulation():
    weights_A = np.ones(NUM_ACTIONS)
    weights_B = np.ones(NUM_ACTIONS)
    strategy_sum_A = np.zeros(NUM_ACTIONS)
    strategy_sum_B = np.zeros(NUM_ACTIONS)
    joint_counts = np.zeros((NUM_ACTIONS, NUM_ACTIONS))
    swap_regret_A = np.zeros((NUM_ACTIONS, NUM_ACTIONS))
    swap_regret_B = np.zeros((NUM_ACTIONS, NUM_ACTIONS))

    plt.ion()
    fig, ax = plt.subplots()

    for t in range(T):
        strategy_A = normalize(weights_A)
        strategy_B = normalize(weights_B)

        strategy_sum_A += strategy_A
        strategy_sum_B += strategy_B

        action_A = choose_action(strategy_A)
        action_B = choose_action(strategy_B)

        joint_counts[action_A][action_B] += 1

        payoff_A, payoff_B = U[action_A][action_B]

        # Swap regret updates
        for i in range(NUM_ACTIONS):
            for j in range(NUM_ACTIONS):
                if i != j:
                    if action_A == i:
                        alt_A, _ = U[j][action_B]
                        swap_regret_A[i][j] += alt_A - payoff_A
                    if action_B == i:
                        _, alt_B = U[action_A][j]
                        swap_regret_B[i][j] += alt_B - payoff_B

        # Update weights
        weights_A = update_weights(swap_regret_A, ETA)
        weights_B = update_weights(swap_regret_B, ETA)

        # Plot every 10 steps
        if (t + 1) % 10 == 0:
            joint_distribution = joint_counts / np.sum(joint_counts)
            plot_heatmap(ax, joint_distribution, t + 1)

    plt.ioff()
    plt.show()

    avg_strategy_A = strategy_sum_A / np.sum(strategy_sum_A)
    avg_strategy_B = strategy_sum_B / np.sum(strategy_sum_B)
    final_joint_distribution = joint_counts / np.sum(joint_counts)

    print("Average Strategy - Player A:", avg_strategy_A)
    print("Average Strategy - Player B:", avg_strategy_B)
    print("\nFinal Joint Probability Distribution:\n", final_joint_distribution)

if __name__ == "__main__":
    run_simulation()
