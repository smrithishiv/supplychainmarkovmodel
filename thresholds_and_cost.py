import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

P = pd.read_csv('outputs/initial_model.csv', index_col=0).values
volume_states = np.arange(0, 1850, 50)  # 0 to 1800 in 50-step increments
n_states = len(volume_states)

# --- Cost Parameters ---
k_per_m3 = 120                          # Variable shipment cost per m³
K1 = 150                                # 900 ft³ truck cost
K2 = 200                                # 1800 ft³ truck cost
c = 0.10 * k_per_m3                     # Holding cost per m³ per day
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
        "Expected_Shipments_per_Day": round(shipment_freq, 4)
    })

# Convert to DataFrame and display
df_results = pd.DataFrame(results)
print(df_results)

# Save to CSV
df_results.to_csv("outputs/threshold_policy_costs.csv", index=False)

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