import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# --- Cost Parameter Generation ---
def random_parameters():
    k_per_m3 = random.uniform(100, 200)  
    K1 = random.randint(150, 250)  
    K2 = random.randint(250, 450)  
    c = random.uniform(0.1, 0.4) * k_per_m3  
    return k_per_m3, K1, K2, c

# --- Load Transition Matrix ---
df_P = pd.read_csv('extended_matrix_outputs/extended_transition_matrix.csv', index_col=0)
state_space = [eval(s) for s in df_P.index]
P = df_P.values

# --- Compute Steady State ---
eigvals, eigvecs = np.linalg.eig(P.T)
steady_state = np.real(eigvecs[:, np.isclose(eigvals, 1)])
steady_state = steady_state[:, 0]
steady_state = steady_state / steady_state.sum()

# --- Thresholds ---
thresholds = np.arange(0, 1800 + 50, 50)

# --- Result Storage ---
results = []

def calculate_costs(k_per_m3, K1, K2, c):
    fixed_3pl = 800
    var_3pl = k_per_m3 / 35.3147
    hold_cost = c / 35.3147
    
    for threshold in thresholds:
        total_3pl_cost = 0
        total_truck_cost = 0
        ship_freq = 0

        for i, (v, due) in enumerate(state_space):
            prob = steady_state[i]
            if prob < 1e-8:
                continue

            holding = v * due * hold_cost
            ship = v >= threshold or due == 1

            if ship:
                ship_freq += prob
                truck_fixed = K1 if v <= 900 else K2
                truck_var = (k_per_m3 / 35.3147) * v

                # 3PL cost
                cost_3pl = fixed_3pl + var_3pl * v
                cost_truck = truck_fixed + truck_var
            else:
                cost_3pl = 0
                cost_truck = 0

            total_3pl_cost += prob * (holding + cost_3pl)
            total_truck_cost += prob * (holding + cost_truck)

        results.append({
            "k_per_m3": k_per_m3,
            "K1": K1,
            "K2": K2,
            "c": c,
            "Threshold": threshold,
            "3PL_Cost": round(total_3pl_cost, 2),
            "Truck_Cost": round(total_truck_cost, 2),
            "Expected_Shipments_per_Day": round(ship_freq, 4)
        })

# Original parameters
original_k = 120
original_K1 = 190
original_K2 = 200
original_c = 0.1 * original_k

calculate_costs(original_k, original_K1, original_K2, original_c)

# Run 20 random scenarios
for _ in range(20):
    k, K1, K2, c = random_parameters()
    calculate_costs(k, K1, K2, c)

# --- Compile Results ---
df_results = pd.DataFrame(results)

# --- Find Optimal Thresholds ---
df_min_3pl = df_results.loc[df_results.groupby(["k_per_m3", "K1", "K2", "c"])["3PL_Cost"].idxmin()]
df_min_3pl.rename(columns={"Threshold": "Best_Threshold_3PL", "3PL_Cost": "Min_3PL_Cost"}, inplace=True)

df_min_truck = df_results.loc[df_results.groupby(["k_per_m3", "K1", "K2", "c"])["Truck_Cost"].idxmin()]
df_min_truck.rename(columns={"Threshold": "Best_Threshold_Truck", "Truck_Cost": "Min_Truck_Cost"}, inplace=True)

# Merge and Save
df_summary = pd.merge(df_min_3pl, df_min_truck, on=["k_per_m3", "K1", "K2", "c"])
df_summary.to_csv("extended_matrix_outputs/threshold_policy_summary.csv", index=False)

# --- Plot Example ---
plt.figure(figsize=(10, 6))
sample = df_results[df_results["k_per_m3"] == original_k]
plt.plot(sample["Threshold"], sample["3PL_Cost"], label="3PL", marker='o')
plt.plot(sample["Threshold"], sample["Truck_Cost"], label="Truck", marker='s')
plt.xlabel("Shipment Threshold")
plt.ylabel("Expected Daily Cost")
plt.title("Policy Cost Curve (Original Parameters)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
