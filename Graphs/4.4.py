import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("outputs/threshold_policy_summary.csv")

# Create short scenario labels (S1, S2, ...)
df["Scenario"] = [f"S{i+1}" for i in range(len(df))]

# Plot optimal thresholds
plt.figure(figsize=(10, 6))
plt.plot(df["Scenario"], df["Best_Threshold_Truck"], marker='o', label="Truck Rental")
plt.plot(df["Scenario"], df["Best_Threshold_3PL"], marker='s', label="3PL")

plt.xlabel("Scenario")
plt.ylabel("Optimal Threshold (ft³)")
plt.title("Optimal Thresholds Across Parameter Scenarios - Basic Model")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/scenario_mapping_basic_matrix.png")
plt.show()

# Optionally save the mapping
df[["Scenario", "k_per_m3", "K1", "K2", "c"]].to_csv("graphs/scenario_mapping_basic_matrix.csv", index=False)


# Load data
df = pd.read_csv("extended_matrix_outputs/threshold_policy_summary.csv")

# Create short scenario labels (S1, S2, ...)
df["Scenario"] = [f"S{i+1}" for i in range(len(df))]

# Plot optimal thresholds
plt.figure(figsize=(10, 6))
plt.plot(df["Scenario"], df["Best_Threshold_Truck"], marker='o', label="Truck Rental")
plt.plot(df["Scenario"], df["Best_Threshold_3PL"], marker='s', label="3PL")

plt.xlabel("Scenario")
plt.ylabel("Optimal Threshold (ft³)")
plt.title("Optimal Thresholds Across Parameter Scenarios - Extended Model")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/scenario_mapping_extended_matrix.png")
plt.show()
