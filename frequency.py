import pandas as pd
import numpy as np

df = pd.read_csv("outputs/initial_model.csv", header=0, index_col=0)
P = df.to_numpy()

steady_state_df = pd.read_csv("outputs/steady_state_probabilities.csv")
π = steady_state_df['Probability'].to_numpy()

volumes = [0] + df.columns.astype(int).tolist()  
print(f"Volumes: {volumes}")

# === Define the threshold for shipment ===
threshold = 1800  

# states with volume >= threshold as absorbing (shipment triggered)
absorbing_states = [i for i, vol in enumerate(volumes) if vol >= threshold]
transient_states = [i for i in range(len(volumes)) if i not in absorbing_states]

# === Compute the fundamental matrix ===
Q = P[np.ix_(transient_states, transient_states)]

I = np.eye(len(transient_states)) 
N = np.linalg.inv(I - Q)  

# Mean time to absorption is the sum of each row of the fundamental matrix
mean_time_to_absorption = N.sum(axis=1) - 1  

# === STEP 4: Map back to original states and calculate frequencies ===
state_to_volume = {i: volumes[i] for i in transient_states}

mean_time_before_threshold = {state_to_volume[i]: mean_time_to_absorption[j] for j, i in enumerate(transient_states)}

shipment_frequency = {volume: 1 / time for volume, time in mean_time_before_threshold.items()}

# Display results
for volume, mean_time in mean_time_before_threshold.items():
    print(f"Volume {volume} ft³: {mean_time:.2f} days")

print("=" * 50)

# Display results for shipment frequency
for volume, frequency in shipment_frequency.items():
    print(f"Volume {volume} ft³: {frequency:.4f} shipments per day")

# Save to CSV
output_df = pd.DataFrame({
    "Volume (ft³)": list(mean_time_before_threshold.keys()),
    "Mean Time to Absorption (days)": list(mean_time_before_threshold.values()),
    "Shipment Frequency (shipments/day)": list(shipment_frequency.values())
})

output_df.to_csv("outputs/shipment_frequency_results_1800.csv", index=False)
print("Results saved to shipment_frequency_results.csv")
