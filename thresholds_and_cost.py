import pandas as pd
import numpy as np

df = pd.read_csv("csv/labelledmatrix.csv", header=0, index_col=0)
P = df.to_numpy()

# === Load steady-state probabilities ===
steady_state_df = pd.read_csv("csv/steady_state_probabilities.csv")
π = steady_state_df['Probability'].to_numpy()

# === Extract volumes directly from transition matrix ===
volumes = df.columns.astype(int).tolist()  # Extract volume labels from the columns

holding_cost_per_m3 = 12  # $/m³/day
shipment_cost_3pl = 800    # $ for 3PL shipment
K1 = 190                  # $ for 900 ft³ truck
K2 = 200                  # $ for 1800 ft³ truck

# Function to calculate expected cost for a given threshold
def calculate_expected_cost(threshold):
    expected_cost_3pl = 0
    expected_cost_k1 = 0
    expected_cost_k2 = 0
    shipment_frequency_3pl = 0
    shipment_frequency_k1 = 0
    shipment_frequency_k2 = 0
    
    # Iterate through volume levels and compute costs
    for vol_ft3 in range(50, 1801, 50):
        vol_m3 = vol_ft3 * (1 / 35.31)
        holding_cost = vol_m3 * holding_cost_per_m3

        # Determine shipment cost based on threshold
        if vol_ft3 >= threshold:
            expected_cost_3pl += π[vol_ft3 // 50 - 1] * (holding_cost + shipment_cost_3pl)
            shipment_frequency_3pl += π[vol_ft3 // 50 - 1]
            
            if vol_ft3 <= 900:
                expected_cost_k1 += π[vol_ft3 // 50 - 1] * (holding_cost + K1)
                shipment_frequency_k1 += π[vol_ft3 // 50 - 1]
            elif vol_ft3 <= 1800:
                expected_cost_k2 += π[vol_ft3 // 50 - 1] * (holding_cost + K2)
                shipment_frequency_k2 += π[vol_ft3 // 50 - 1]
        else:
            expected_cost_3pl += π[vol_ft3 // 50 - 1] * holding_cost
            expected_cost_k1 += π[vol_ft3 // 50 - 1] * holding_cost
            expected_cost_k2 += π[vol_ft3 // 50 - 1] * holding_cost
    
    return expected_cost_3pl, expected_cost_k1, expected_cost_k2

# Iterate over possible thresholds and calculate expected costs
thresholds = list(range(50, 1801, 50))
costs = []
for threshold in thresholds:
    cost_3pl, cost_k1, cost_k2 = calculate_expected_cost(threshold)
    costs.append([threshold, cost_3pl, cost_k1, cost_k2])

# Find the optimal threshold with the lowest total cost
cost_df = pd.DataFrame(costs, columns=["Threshold (ft³)", "Cost/Day (3PL)", "Cost/Day (Truck 900ft³)", "Cost/Day (Truck 1800ft³)"])
optimal_threshold_3pl = cost_df.loc[cost_df["Cost/Day (3PL)"].idxmin()]
optimal_threshold_k1 = cost_df.loc[cost_df["Cost/Day (Truck 900ft³)"].idxmin()]
optimal_threshold_k2 = cost_df.loc[cost_df["Cost/Day (Truck 1800ft³)"].idxmin()]

# Display optimal thresholds
print(f"Optimal Threshold for 3PL: {optimal_threshold_3pl}")
print(f"Optimal Threshold for 900ft³ Truck: {optimal_threshold_k1}")
print(f"Optimal Threshold for 1800ft³ Truck: {optimal_threshold_k2}")