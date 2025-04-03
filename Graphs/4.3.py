import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig

# --- Load results ---
df = pd.read_csv("extended_matrix_outputs/q4_threshold_policy_costs_with_3pl.csv")

# ❌ Remove use of missing "Valid" column
# ✅ Instead, just show where costs are minimized
df["IsOptimalTruck"] = df["Truck_Cost"] == df["Truck_Cost"].min()
df["IsOptimal3PL"] = df["3PL_Cost"] == df["3PL_Cost"].min()

# --- Plot optimal threshold positions ---
plt.figure(figsize=(10, 6))
plt.plot(df["Threshold"], df["Truck_Cost"], label="Truck Cost", marker='o')
plt.plot(df["Threshold"], df["3PL_Cost"], label="3PL Cost", marker='s')

# Mark optimal thresholds
for _, row in df[df["IsOptimalTruck"]].iterrows():
    plt.axvline(row["Threshold"], color='green', linestyle='--', alpha=0.5, label="Optimal Truck Threshold")
for _, row in df[df["IsOptimal3PL"]].iterrows():
    plt.axvline(row["Threshold"], color='blue', linestyle='--', alpha=0.5, label="Optimal 3PL Threshold")

plt.xlabel("Threshold (ft³)")
plt.ylabel("Expected Daily Cost ($)")
plt.title("Shipment Cost vs Threshold (Truck vs 3PL)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/q4_cost_thresholds.png")
plt.show()

# --- Load Markov matrix ---
df_matrix = pd.read_csv("extended_matrix_outputs/extended_transition_matrix.csv", index_col=0)
P = df_matrix.values
states = [eval(s) for s in df_matrix.index]

# --- Steady-state distribution ---
eigvals, eigvecs = eig(P.T, left=True, right=False)
ss = np.real(eigvecs[:, np.isclose(eigvals, 1)])
ss = ss[:, 0]
ss = ss / ss.sum()

# --- Expected inventory volume by due date ---
due_volume = {2: 0, 3: 0, 4: 0}

for (v, due), p in zip(states, ss):
    if due == 1:
        due_volume[2] += v * p  # will be due in 2 days
    elif due == 2:
        due_volume[3] += v * p  # due in 3 days
    elif due >= 3:
        due_volume[4] += v * p  # due in 4+ days

# --- Bar Plot ---
plt.figure(figsize=(8, 5))
plt.bar(due_volume.keys(), due_volume.values())
plt.xlabel("Days Until Due")
plt.ylabel("Expected Inventory Volume (ft³)")
plt.title("Expected Volume Distribution by Due Date")
plt.xticks([2, 3, 4])
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/due_day_volume.png")
plt.show()
