import numpy as np
from scipy.optimize import linprog

# Combined payoff matrix: each entry is a tuple (payoff_A, payoff_B)
U = np.array([
    [(0, 0), (2,3 ), (0, 0)],
    [(1, 2), (0, 0), (2, 1)],
    [(0, 0), (3, 2), (0, 0)]
])

n = U.shape[0]  # number of strategies

# Extract individual payoff matrices for players A and B
U_A = np.array([[U[i, j][0] for j in range(n)] for i in range(n)])
U_B = np.array([[U[i, j][1] for j in range(n)] for i in range(n)])

A_ub = []

# Player A's incentive constraints (row deviations for fixed column)
for i in range(n):
    for ip in range(n):
        if i != ip:
            row = np.zeros(n * n)
            for j in range(n):
                idx = i * n + j
                idx_p = ip * n + j
                row[idx] -= U_A[i, j]
                row[idx_p] += U_A[ip, j]
            A_ub.append(row)

# Player B's incentive constraints (column deviations for fixed row)
for j in range(n):
    for jp in range(n):
        if j != jp:
            row = np.zeros(n * n)
            for i in range(n):
                idx = i * n + j
                idx_p = i * n + jp
                row[idx] -= U_B[i, j]
                row[idx_p] += U_B[i, jp]
            A_ub.append(row)

A_ub = np.array(A_ub)
b_ub = np.zeros(A_ub.shape[0])

# Equality constraint: sum of probabilities = 1
A_eq = np.ones((1, n * n))
b_eq = np.array([1])

# Bounds for each probability: 0 <= p_i <= 1
bounds = [(0, 1) for _ in range(n * n)]

# Objective function: arbitrary (we just want any feasible point)
f = np.random.randn(n * n)

# Solve using linprog
result = linprog(c=f, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

if result.success:
    print("Correlated equilibrium distribution:")
    print(np.round(result.x.reshape(n, n), 4))  # reshape for readability
else:
    print("No solution found:", result.message)
