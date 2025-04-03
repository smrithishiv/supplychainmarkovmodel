import pandas as pd
import matplotlib.pyplot as plt

# Load both datasets
df_threshold = pd.read_csv("extended_matrix_outputs/q4_threshold_policy_costs_with_3pl.csv")
df_periodic = pd.read_csv("4_day_policy/periodic_policy_costs.csv")  # Make sure it includes Truck_Cost and 3PL_Cost columns

# Set benchmark values from periodic policy
periodic_truck = df_periodic["Truck_Daily_Cost"].values[0]
periodic_3pl = df_periodic["3PL_Daily_Cost"].values[0]

plt.figure(figsize=(10, 6))
plt.plot(df_threshold["Threshold"], df_threshold["Truck_Cost"], label="Dynamic Policy – Truck", marker='o')
plt.plot(df_threshold["Threshold"], df_threshold["3PL_Cost"], label="Dynamic Policy – 3PL", marker='s')
plt.axhline(periodic_truck, color='blue', linestyle='--', label=f"Periodic Policy – Truck (${periodic_truck:.2f})")
plt.axhline(periodic_3pl, color='orange', linestyle='--', label=f"Periodic Policy – 3PL (${periodic_3pl:.2f})")

plt.xlabel("Shipment Threshold (ft³)")
plt.ylabel("Expected Daily Cost ($)")
plt.title("Dynamic vs Periodic Policy Cost Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/dynamic_vs_periodic_comparison.png")
plt.show()

# Compute savings
df_threshold["Truck_Savings"] = periodic_truck - df_threshold["Truck_Cost"]
df_threshold["3PL_Savings"] = periodic_3pl - df_threshold["3PL_Cost"]

plt.figure(figsize=(10, 6))
plt.plot(df_threshold["Threshold"], df_threshold["Truck_Savings"], marker='o', label="Truck Cost Savings")
plt.plot(df_threshold["Threshold"], df_threshold["3PL_Savings"], marker='s', label="3PL Cost Savings")
plt.xlabel("Shipment Threshold (ft³)")
plt.ylabel("Daily Cost Savings ($)")
plt.title("Cost Savings of Dynamic Policies vs Periodic Policy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("graphs/savings_vs_threshold.png")
plt.show()
