import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eig
import random

# --- Load Transition Matrix ---
df_matrix = pd.read_csv("extended_matrix_outputs/extended_transition_matrix.csv", index_col=0)
df_matrix.columns = [eval(col) for col in df_matrix.columns]
df_matrix.index = [eval(idx) for idx in df_matrix.index]

P = df_matrix.values
states = list(df_matrix.index)

# --- Compute Steady-State Distribution ---
eigvals, left_evecs = eig(P.T, left=True, right=False)
steady_state = np.real(left_evecs[:, np.isclose(eigvals, 1)])
steady_state = steady_state[:, 0]
steady_state /= steady_state.sum()

# --- Thresholds & Parameters ---
thresholds = np.arange(600, 1800 + 50, 50)

def random_parameters():
    k_per_m3 = random.uniform(100, 200)
    K1 = random.randint(150, 250)
    K2 = random.randint(250, 450)
    c = random.uniform(0.1, 0.4) * k_per_m3
    return k_per_m3, K1, K2, c

results = []

# --- Evaluate 20 Random Policies ---
for _ in range(20):
    k_per_m3, K1, K2, c = random_parameters()
    holding_cost_per_cuft_per_day = c / 35.3147
    fixed_3pl_cost = 800
    variable_3pl_cost = k_per_m3 / 35.3147

    for threshold in thresholds:
        total_cost_truck = 0
        total_cost_3pl = 0
        total_shipments = 0

        for (v, due), prob in zip(states, steady_state):
            if prob < 1e-8:
                continue

            holding_days = due if due > 1 else 0
            holding_cost = v * holding_days * holding_cost_per_cuft_per_day
            ship = v >= threshold or due == 1

            if ship:
                total_shipments += prob
                fixed_truck = K1 if v <= 900 else K2
                variable_truck = (k_per_m3 / 35.3147) * v
                fixed_3pl = fixed_3pl_cost
                variable_3pl = variable_3pl_cost * v
            else:
                fixed_truck = variable_truck = fixed_3pl = variable_3pl = 0

            total_cost_truck += prob * (holding_cost + fixed_truck + variable_truck)
            total_cost_3pl += prob * (holding_cost + fixed_3pl + variable_3pl)

        results.append({
            "k_per_m3": round(k_per_m3, 2),
            "K1": K1,
            "K2": K2,
            "c": round(c, 2),
            "Threshold": threshold,
            "Truck_Cost": round(total_cost_truck, 2),
            "3PL_Cost": round(total_cost_3pl, 2),
            "Expected_Shipments_per_Day": round(total_shipments, 4)
        })

# --- Summarize Optimal Thresholds ---
df_all = pd.DataFrame(results)
df_min_3pl = df_all.loc[df_all.groupby(["k_per_m3", "K1", "K2", "c"])["3PL_Cost"].idxmin()].copy()
df_min_3pl.rename(columns={"Threshold": "Best_Threshold_3PL", "3PL_Cost": "Min_3PL_Cost"}, inplace=True)

df_min_truck = df_all.loc[df_all.groupby(["k_per_m3", "K1", "K2", "c"])["Truck_Cost"].idxmin()].copy()
df_min_truck.rename(columns={"Threshold": "Best_Threshold_Truck", "Truck_Cost": "Min_Truck_Cost"}, inplace=True)

df_summary = pd.merge(df_min_3pl, df_min_truck, on=["k_per_m3", "K1", "K2", "c"])
df_summary.to_csv("extended_matrix_outputs/threshold_policy_summary.csv", index=False)

print("✅ Summary generated and saved.")
