from collections import defaultdict

def simulate_algorithm(cost_matrix, tie_break='lowest'):
    n_arms = 4
    T = 7
    cumulative_costs = [0] * n_arms
    selected_arms = []
    total_cost = 0

    for t in range(T):  # t is 0-based (0 to 6)
        current_time = t + 1  # 1-based time
        if current_time == 1:
            selected = 0  # Arm 1 (0-based index)
        else:
            min_cost = min(cumulative_costs)
            candidates = [i for i in range(n_arms) if cumulative_costs[i] == min_cost]
            if tie_break == 'lowest':
                selected = min(candidates)
            elif tie_break == 'highest':
                selected = max(candidates)
            else:
                raise ValueError("Invalid tie_break")
        selected_arm = selected + 1  # Convert to 1-based
        selected_arms.append(selected_arm)
        total_cost += cost_matrix[t][selected]
        # Update cumulative costs with current time's costs for all arms
        for i in range(n_arms):
            cumulative_costs[i] += cost_matrix[t][i]
    return selected_arms, total_cost

def compute_external_regret(cost_matrix, algorithm_total):
    arm_totals = [sum(col) for col in zip(*cost_matrix)]
    best_arm_total = min(arm_totals)
    return algorithm_total - best_arm_total

def compute_swap_regret(selected_arms, cost_matrix):
    arm_times = defaultdict(list)
    for t, arm in enumerate(selected_arms):
        arm_times[arm].append(t)
    
    num_arms = len(cost_matrix[0])
    
    # Find best swap for each arm
    best_swaps = {}
    for arm in arm_times:
        min_cost = float('inf')
        best_target = None
        for target_arm in range(1, num_arms + 1):
            total = sum(cost_matrix[t][target_arm - 1] for t in arm_times[arm])
            if total < min_cost:
                min_cost = total
                best_target = target_arm
        best_swaps[arm] = (best_target, min_cost)
    
    # Total cost using best swaps
    swapped_total_cost = sum(cost for _, cost in best_swaps.values())
    algorithm_total = sum(cost_matrix[t][arm - 1] for t, arm in enumerate(selected_arms))
    
    return algorithm_total - swapped_total_cost

# Generate cost matrices for parts (i) and (ii)
n_arms = 4
T = 7

# Cost sequence C (part i)
cost_matrix_i = []
for t in range(1, T + 1):
    row = []
    for i in range(1, n_arms + 1):
        row.append(1 if (t - i) % 4 == 0 else 0)
    cost_matrix_i.append(row)

# Cost sequence C' (part ii)
cost_matrix_ii = []
for t in range(1, T + 1):
    row = []
    for i in range(1, n_arms + 1):
        row.append(0 if (t - i) % 4 == 0 else 1)
    cost_matrix_ii.append(row)

# Simulate all scenarios

# Original greedy for part (i)
selected_i_greedy, total_i_greedy = simulate_algorithm(cost_matrix_i, 'lowest')
external_i_greedy = compute_external_regret(cost_matrix_i, total_i_greedy)
swap_i_greedy = compute_swap_regret(selected_i_greedy, cost_matrix_i)

# Original greedy for part (ii)
selected_ii_greedy, total_ii_greedy = simulate_algorithm(cost_matrix_ii, 'lowest')
external_ii_greedy = compute_external_regret(cost_matrix_ii, total_ii_greedy)
swap_ii_greedy = compute_swap_regret(selected_ii_greedy, cost_matrix_ii)

# Modified greedy for part (i)
selected_i_modified, total_i_modified = simulate_algorithm(cost_matrix_i, 'highest')
external_i_modified = compute_external_regret(cost_matrix_i, total_i_modified)
swap_i_modified = compute_swap_regret(selected_i_modified, cost_matrix_i)

# Modified greedy for part (ii)
selected_ii_modified, total_ii_modified = simulate_algorithm(cost_matrix_ii, 'highest')
external_ii_modified = compute_external_regret(cost_matrix_ii, total_ii_modified)
swap_ii_modified = compute_swap_regret(selected_ii_modified, cost_matrix_ii)

print("Part (i) with original greedy:")
print(f"External regret: {external_i_greedy}, Swap regret: {swap_i_greedy}")
print("\nPart (ii) with original greedy:")
print(f"External regret: {external_ii_greedy}, Swap regret: {swap_ii_greedy}")
print("\nPart (i) with modified greedy:")
print(f"External regret: {external_i_modified}, Swap regret: {swap_i_modified}")
print("\nPart (ii) with modified greedy:")
print(f"External regret: {external_ii_modified}, Swap regret: {swap_ii_modified}")
