import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random

# --- Original Cost Parameters ---
original_k_per_m3 = 120  # Original variable shipment cost per m³
original_K1 = 190  # Original 900 ft³ truck cost
original_K2 = 200  # Original 1800 ft³ truck cost
original_c = 0.1 * original_k_per_m3  # Original holding cost per m³ per day
original_holding_cost_per_cuft_per_day = original_c / 35.3147  # Convert m³ to ft³

thresholds = np.arange(0, 1800 + 50, 50)

# --- Random Parameter Generation ---
def random_parameters():
    k_per_m3 = random.uniform(100, 200)  # Random variable shipment cost between 100 and 200
    K1 = random.randint(150, 250)  # Random 900 ft³ truck cost between 150 and 250
    K2 = random.randint(250, 450)  # Random 1800 ft³ truck cost between 250 and 350
    c = random.uniform(0.1, 0.4) * k_per_m3  # Random holding cost between 10% and 40% of k_per_m3
    return k_per_m3, K1, K2, c

# --- Load Transition Matrix ---
P = pd.read_csv('outputs/initial_model.csv', index_col=0).values
volume_states = np.arange(0, 1850, 50)  # 0 to 1800 in 50-step increments
n_states = len(volume_states)

# --- Steady State Distribution ---
eigvals, eigvecs = np.linalg.eig(P.T)
steady_state = np.real(eigvecs[:, np.isclose(eigvals, 1)])
steady_state = steady_state[:, 0]
steady_state = steady_state / steady_state.sum()

# --- Results Storage ---
results = []

def calculate_costs(k_per_m3, K1, K2, c):
    holding_cost_per_cuft_per_day = c / 35.3147  # Convert m³ to ft³
    for threshold in thresholds:
        total_daily_3pl_cost = 0
        total_daily_truck_cost = 0
        shipment_freq = 0

        for i, v in enumerate(volume_states):
            prob = steady_state[i]
            holding_cost = v * holding_cost_per_cuft_per_day

            if v >= threshold:
                shipment_freq += prob
                truck_cost = K1 if v <= 900 else K2
                variable_cost = (k_per_m3 / 35.3147) * v
            else:
                truck_cost = 0
                variable_cost = 0

            total_daily_3pl_cost += prob * (holding_cost + variable_cost)
            total_daily_truck_cost += prob * (holding_cost + truck_cost)

        results.append({
            "k_per_m3": k_per_m3,
            "K1": K1,
            "K2": K2,
            "c": c,
            "Threshold": threshold,
            "3PL_Cost": round(total_daily_3pl_cost, 2),
            "Truck_Cost": round(total_daily_truck_cost, 2),
            "Expected_Shipments_per_Day": round(shipment_freq, 4)
        })

# Calculate for original parameters
calculate_costs(original_k_per_m3, original_K1, original_K2, original_c)

# Calculate for random parameter sets
for _ in range(20):
    k_per_m3, K1, K2, c = random_parameters()
    calculate_costs(k_per_m3, K1, K2, c)

# Convert to DataFrame
df_results = pd.DataFrame(results)


# --- Find Minimum Costs and Corresponding Thresholds ---
df_min_3pl = df_results.loc[df_results.groupby(["k_per_m3", "K1", "K2", "c"])["3PL_Cost"].idxmin(), ["k_per_m3", "K1", "K2", "c", "Threshold", "3PL_Cost"]]
df_min_3pl.rename(columns={"Threshold": "Best_Threshold_3PL", "3PL_Cost": "Min_3PL_Cost"}, inplace=True)

df_min_truck = df_results.loc[df_results.groupby(["k_per_m3", "K1", "K2", "c"])["Truck_Cost"].idxmin(), ["k_per_m3", "K1", "K2", "c", "Threshold", "Truck_Cost"]]
df_min_truck.rename(columns={"Threshold": "Best_Threshold_Truck", "Truck_Cost": "Min_Truck_Cost"}, inplace=True)

# Merge the two summaries
df_summary = pd.merge(df_min_3pl, df_min_truck, on=["k_per_m3", "K1", "K2", "c"])

# Overwrite summary CSV
df_summary.to_csv("outputs/threshold_policy_summary.csv", index=False)

# --- Plot Costs vs Threshold --- 
plt.figure(figsize=(10, 6))
plt.plot(df_results["Threshold"], df_results["3PL_Cost"], marker='o', label="3PL Cost")
plt.plot(df_results["Threshold"], df_results["Truck_Cost"], marker='s', label="Truck Rental Cost")
plt.xlabel("Shipment Threshold (ft³)")
plt.ylabel("Expected Daily Cost ($)")
plt.title("Cost vs Shipment Threshold")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Display summary results
print(df_summary)