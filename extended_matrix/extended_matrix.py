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
volume_step = 50
max_volume = 1800  # Extended to cover up to large truck capacity
max_due_days = 4   # Include due days 2, 3, and 4

def generate_poisson_distribution(lambda_orders, max_orders=6):
    poisson_probs = [((lambda_orders ** k) * np.exp(-lambda_orders)) / math.factorial(k) for k in range(max_orders)]
    poisson_probs.append(1 - sum(poisson_probs))  # Bucket for 6+ orders
    return poisson_probs

# --- Step 1: Daily Volume Distribution ---
volume_dist = defaultdict(float)
component_list = list(component_volumes.keys())
poisson_probs = generate_poisson_distribution(lambda_orders)

for n_orders, p in enumerate(poisson_probs):
    actual_orders = min(n_orders, max_daily_production)
    combos = list(itertools.product(component_list, repeat=actual_orders))
    for combo in combos:
        vol = sum(component_volumes[c] for c in combo)
        prob = p
        for c in component_list:
            if actual_orders > 0:
                prob *= (component_probs[c] ** combo.count(c))
        volume_dist[vol] += prob

# --- Step 2: Define Extended States (Volume, Due Days Remaining) ---
volume_states = list(range(0, max_volume + 1, volume_step))  # 0 to 1800
due_states = list(range(0, max_due_days + 1))  # 0 to 4
state_space = list(itertools.product(volume_states, due_states))
state_index = {state: i for i, state in enumerate(state_space)}
n_states = len(state_space)

# --- Step 3: Build Transition Matrix ---
P = np.zeros((n_states, n_states))

for (v, due), i in state_index.items():
    if due == 1:
        # Force shipment → reset to (0, 0)
        j = state_index[(0, 0)]
        P[i, j] = 1.0
    else:
        for added_v, prob in volume_dist.items():
            new_total = v + added_v
            new_due = max(due - 1, 0)  # Inventory ages by one day

            if new_total > max_volume:
                # Ship up to 1800 ft³, carry over remainder
                carryover = new_total - 1800
                carryover = min(carryover, max_volume)
                new_state = (carryover, new_due)
            else:
                new_state = (new_total, new_due)

            j = state_index[new_state]
            P[i, j] += prob

# --- Step 4: Save Transition Matrix ---
df = pd.DataFrame(P, index=state_space, columns=state_space)
df.to_csv("extended_matrix_outputs/extended_transition_matrix.csv")
print("✅ Extended transition matrix saved successfully.")
