import numpy as np
import pandas as pd
import itertools
import math
from collections import defaultdict

# --- Parameters ---
component_volumes = {"A": 50, "B": 100, "C": 150}
component_probs = {"A": 0.3, "B": 0.5, "C": 0.2}
lambda_orders = 2
max_daily_production = 3
k_per_m3 = 120
c = 0.10 * k_per_m3
K1 = 190  # Truck (900 ft³)
K2 = 200  # Truck (1800 ft³)
K_3PL = 800  # 3PL flat shipment cost
holding_cost_per_cuft_per_day = c / 35.3147
volume_step = 50
max_volume = 1800

# --- Generate daily volume distribution ---
poisson_probs = [((lambda_orders ** k) * np.exp(-lambda_orders)) / math.factorial(k) for k in range(6)]
poisson_probs.append(1 - sum(poisson_probs))
component_list = list(component_volumes.keys())

daily_volume_dist = defaultdict(float)
for n_orders, p_orders in enumerate(poisson_probs):
    actual_orders = min(n_orders, max_daily_production)
    combos = list(itertools.product(component_list, repeat=actual_orders))
    for combo in combos:
        prob = p_orders
        for c in component_list:
            prob *= component_probs[c] ** combo.count(c)
        total_vol = sum(component_volumes[c] for c in combo)
        daily_volume_dist[total_vol] += prob

# Normalize
total_p = sum(daily_volume_dist.values())
for k in daily_volume_dist:
    daily_volume_dist[k] /= total_p

# --- Simulate accumulation over 4 days ---
accumulated_dist = defaultdict(float)
for v1, p1 in daily_volume_dist.items():
    for v2, p2 in daily_volume_dist.items():
        for v3, p3 in daily_volume_dist.items():
            for v4, p4 in daily_volume_dist.items():
                total_vol = v1 + v2 + v3 + v4
                total_vol = min(total_vol, max_volume)
                prob = p1 * p2 * p3 * p4
                accumulated_dist[total_vol] += prob

# --- Compute expected costs (truck and 3PL) ---
total_truck_cost = 0
total_3pl_cost = 0
expected_volume_per_4days = 0

for vol, prob in accumulated_dist.items():
    holding_cost = vol * 1.5 * holding_cost_per_cuft_per_day
    truck_fixed = K1 if vol <= 900 else K2
    variable_cost = (k_per_m3 / 35.3147) * vol

    truck_cost = holding_cost + truck_fixed + variable_cost
    pl_cost = holding_cost + K_3PL + variable_cost

    total_truck_cost += prob * truck_cost
    total_3pl_cost += prob * pl_cost
    expected_volume_per_4days += prob * vol

# Normalize to daily values
truck_daily = total_truck_cost / 4
pl_daily = total_3pl_cost / 4
daily_vol = expected_volume_per_4days / 4

# Print
print("📦 4-Day Periodic Policy Results")
print(f"🚚 Truck Rental Expected Daily Cost: ${truck_daily:.2f}")
print(f"📦 3PL Expected Daily Cost:           ${pl_daily:.2f}")
print(f"📈 Avg Volume Shipped Every 4 Days:   {expected_volume_per_4days:.1f} ft³")
print(f"📅 Avg Daily Shipment Volume:         {daily_vol:.1f} ft³")

# Save to CSV
df = pd.DataFrame([{
    "Policy": "4-Day Periodic",
    "Truck_Daily_Cost": round(truck_daily, 2),
    "3PL_Daily_Cost": round(pl_daily, 2),
    "Avg_Shipment_Volume_4_Days": round(expected_volume_per_4days, 1),
    "Avg_Shipment_Volume_per_Day": round(daily_vol, 1)
}])

df.to_csv("4_day_policy/periodic_policy_costs.csv", index=False)
print("\n✅ Results saved to '4_day_policy/periodic_policy_costs.csv")
