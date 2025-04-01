import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# --- Load Transition Matrix ---
P = pd.read_csv('outputs/initial_model.csv', index_col=0).values
volume_states = np.arange(0, 1850, 50)  # 0 to 1800 in 50-step increments
n_states = len(volume_states)

# --- Cost Parameters (Updateable) ---
k_per_m3 = 156                         # Variable shipment cost per m³
K1 = 300                                 # 900 ft³ truck cost
K2 = 400                                # 1800 ft³ truck cost
c = 0.15 * k_per_m3                     # Holding cost per m³ per day
holding_cost_per_cuft_per_day = c / 35.3147  # Convert m³ to ft³

thresholds = np.arange(0, 1800 + 50, 50)

# --- Steady State Distribution ---
eigvals, eigvecs = np.linalg.eig(P.T)
steady_state = np.real(eigvecs[:, np.isclose(eigvals, 1)])
steady_state = steady_state[:, 0]
steady_state = steady_state / steady_state.sum()

# --- Results Storage ---
results = []

for threshold in thresholds:
    total_daily_3pl_cost = 0
    total_daily_truck_cost = 0
    shipment_freq = 0

    for i, v in enumerate(volume_states):
        prob = steady_state[i]

        # Holding cost
        holding_cost = v * holding_cost_per_cuft_per_day

        # Shipping costs (if threshold met)
        if v >= threshold:
            shipment_freq += prob
            # Choose truck size
            if v <= 900:
                truck_cost = K1
            else:
                truck_cost = K2

            variable_cost = (k_per_m3 / 35.3147) * v
        else:
            truck_cost = 0
            variable_cost = 0

        # Accumulate daily expected costs
        total_daily_3pl_cost += prob * (holding_cost + variable_cost)
        total_daily_truck_cost += prob * (holding_cost + truck_cost)

    results.append({
        "Threshold": threshold,
        "3PL_Cost": round(total_daily_3pl_cost, 2),
        "Truck_Cost": round(total_daily_truck_cost, 2),
        "Expected_Shipments_per_Day": round(shipment_freq, 4),
        "k_per_m3": k_per_m3,
        "K1": K1,
        "K2": K2
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)

# --- Append to CSV ---
csv_path = "outputs/threshold_policy_costs.csv"

if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path)
    df_combined = pd.concat([df_existing, df_results], ignore_index=True)
else:
    df_combined = df_results

# Save updated results
df_combined.to_csv(csv_path, index=False)

# --- Find Minimum Costs and Corresponding Thresholds ---
df_min_3pl = df_combined.loc[df_combined.groupby(["k_per_m3", "K1", "K2"])["3PL_Cost"].idxmin(), ["k_per_m3", "K1", "K2", "Threshold", "3PL_Cost"]]
df_min_3pl.rename(columns={"Threshold": "Best_Threshold_3PL", "3PL_Cost": "Min_3PL_Cost"}, inplace=True)

df_min_truck = df_combined.loc[df_combined.groupby(["k_per_m3", "K1", "K2"])["Truck_Cost"].idxmin(), ["k_per_m3", "K1", "K2", "Threshold", "Truck_Cost"]]
df_min_truck.rename(columns={"Threshold": "Best_Threshold_Truck", "Truck_Cost": "Min_Truck_Cost"}, inplace=True)

# Merge the two summaries
df_summary = pd.merge(df_min_3pl, df_min_truck, on=["k_per_m3", "K1", "K2"])

# Save summary
# df_summary.to_csv("outputs/threshold_policy_summary.csv", index=False)

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