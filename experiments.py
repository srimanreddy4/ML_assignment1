import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_avg_time = 100

def gen_data(N, P, case):
    if case == 1:
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randn(N))
    elif case == 2:
        X = pd.DataFrame(np.random.randn(N, P))
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif case == 3:
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randint(P, size=N), dtype="category")
    elif case == 4:
        X = pd.DataFrame({i: pd.Series(np.random.randint(P, size=N), dtype="category") for i in range(P)})
        y = pd.Series(np.random.randn(N))
    else:
        raise ValueError("Invalid case number")
    return X, y

def meas_learn(X_tr, y_tr, case):
    start = time.time()
    crit = "MSE" if case in [1, 3] else "gini_index"
    tree = DecisionTree(criterion=crit)
    tree.fit(X_tr, y_tr)
    learn_time = time.time() - start
    return learn_time, tree

def meas_pred(tree, X_te):
    start = time.time()
    y_pred = tree.predict(X_te)
    pred_time = time.time() - start
    return pred_time

Ns = [20, 60, 100]
Ms = [10, 40, 70]
cases = [1, 2, 3, 4]
l_time = []
p_time = []

for case in cases:
    case_l = []
    case_p = []
    
    for N in Ns:
        for M in Ms:
            X, y = gen_data(N, M, case)
            y_df = pd.DataFrame(y, columns=['new_col'])
            X = X.join(y_df, rsuffix='_y')
            
            split = int(0.7*N)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]
            
            lt, tree = meas_learn(X_tr, y_tr, case)
            case_l.append(lt)
            
            pt = meas_pred(tree, X_te)
            case_p.append(pt)
    
    l_time.append(case_l)
    p_time.append(case_p)

fig, axes = plt.subplots(nrows=len(cases), ncols=2, figsize=(15, 8))
fig.suptitle('Time Complexity Analysis')

for i, case in enumerate(cases):
    axes[i, 0].set_title(f"Case {case} Learning Time")
    for j, N in enumerate(Ns):
        axes[i, 0].plot(Ms, l_time[i][j * len(Ms):(j + 1) * len(Ms)], label=f'N={N}')
    axes[i, 0].set_xlabel('P (Features)')
    axes[i, 0].set_ylabel('Time (s)')
    axes[i, 0].legend()

    axes[i, 1].set_title(f"Case {case} Prediction Time")
    for j, N in enumerate(Ns):
        axes[i, 1].plot(Ms, p_time[i][j * len(Ms):(j + 1) * len(Ms)], label=f'N={N}')
    axes[i, 1].set_xlabel('P (Features)')
    axes[i, 1].set_ylabel('Time (s)')
    axes[i, 1].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()