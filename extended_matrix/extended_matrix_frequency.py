import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# --- Load Markov transition matrix ---
df_matrix = pd.read_csv("extended_matrix_outputs/extended_transition_matrix.csv", index_col=0)
P = df_matrix.values
states = [eval(idx) for idx in df_matrix.index]

# --- Compute steady-state distribution ---
eigvals, left_evecs = eig(P.T, left=True, right=False)
steady_state = np.real(left_evecs[:, np.isclose(eigvals, 1)])
steady_state = steady_state[:, 0]
steady_state /= steady_state.sum()

# --- Shipment cost parameters ---
thresholds = np.arange(600, 1800 + 50, 50)
k_per_m3 = 120
K1 = 190  # Small truck (900 ft³)
K2 = 200  # Large truck (1800 ft³)
c = 0.10 * k_per_m3
holding_cost_per_cuft_per_day = c / 35.3147

# 3PL parameters
fixed_3pl_cost = 800
variable_cost_per_cuft_3pl = k_per_m3 / 35.3147

# --- Evaluate cost under each threshold ---
results = []

for threshold in thresholds:
    total_cost_truck = 0
    total_cost_3pl = 0
    total_shipments = 0

    for (volume, due), prob in zip(states, steady_state):
        if prob < 1e-8:
            continue

        holding_days = due if due > 1 else 0
        holding_cost = volume * holding_days * holding_cost_per_cuft_per_day

        ship = volume >= threshold or due == 1

        if ship:
            total_shipments += prob
            # --- Truck costs ---
            fixed_truck = K1 if volume <= 900 else K2
            variable_truck = (k_per_m3 / 35.3147) * volume

            # --- 3PL costs ---
            fixed_3pl = fixed_3pl_cost
            variable_3pl = variable_cost_per_cuft_3pl * volume
        else:
            fixed_truck = 0
            variable_truck = 0
            fixed_3pl = 0
            variable_3pl = 0

        total_cost_truck += prob * (holding_cost + fixed_truck + variable_truck)
        total_cost_3pl += prob * (holding_cost + fixed_3pl + variable_3pl)

    results.append({
        "Threshold": threshold,
        "Valid": True,
        "Truck_Cost": round(total_cost_truck, 2),
        "3PL_Cost": round(total_cost_3pl, 2),
        "Expected_Shipments_per_Day": round(total_shipments, 4)
    })

# --- Output results ---
df_results = pd.DataFrame(results)
df_results.to_csv("extended_matrix_outputs/q4_threshold_policy_costs_with_3pl.csv", index=False)
# --- Identify optimal policies ---
min_truck_cost = df_results["Truck_Cost"].min()
min_3pl_cost = df_results["3PL_Cost"].min()

df_results["IsOptimal_Truck"] = df_results["Truck_Cost"] == min_truck_cost
df_results["IsOptimal_3PL"] = df_results["3PL_Cost"] == min_3pl_cost

print(df_results)

# --- Plot results ---
plt.figure(figsize=(10, 6))
plt.plot(df_results["Threshold"], df_results["Truck_Cost"], marker='o', label="Truck Rental Cost")
plt.plot(df_results["Threshold"], df_results["3PL_Cost"], marker='s', label="3PL Cost")
plt.xlabel("Shipment Threshold (ft³)")
plt.ylabel("Expected Daily Cost ($)")
plt.title("Q4 Shipment Cost Comparison: Truck vs 3PL")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
