import numpy as np
import pandas as pd
import itertools
import math

# --- Parameters ---
component_volumes = {"A": 50, "B": 100, "C": 150}
component_probs = {"A": 0.3, "B": 0.5, "C": 0.2}
due_date_probs = {2: 0.3, 3: 0.4, 4: 0.3}
lambda_orders = 2
max_daily_production = 3
volume_step = 50
max_volume = 1800
max_due_days = 4

# --- Generate Poisson Arrival Distribution ---
def generate_poisson_distribution(lmbda, max_orders=6):
    probs = [((lmbda ** k) * np.exp(-lmbda)) / math.factorial(k) for k in range(max_orders)]
    probs.append(1 - sum(probs))  # Bucket for 6+ orders
    return probs

poisson_probs = generate_poisson_distribution(lambda_orders)
components = list(component_volumes.keys())

# --- Step 1: Define State Space ---
volume_states = list(range(0, max_volume + 1, volume_step))
due_states = list(range(0, max_due_days + 1))  # 0–4 (0 = must ship today)
state_space = [(v, d) for v in volume_states for d in due_states]
state_index = {state: i for i, state in enumerate(state_space)}
n_states = len(state_space)

# --- Step 2: Build Transition Matrix ---
P = np.zeros((n_states, n_states))

for (v, due), i in state_index.items():

    # Force shipment for any volume with due = 0 (must ship today)
    if due == 0 and v > 0:
        j = state_index[(0, 0)]
        P[i, j] = 1.0
        continue

    for n_orders, p_orders in enumerate(poisson_probs):
        actual_orders = min(n_orders, max_daily_production)

        if actual_orders == 0:
            if v == 0:
                j = state_index[(0, 0)]
            elif due == 1:
                j = state_index[(0, 0)]
            else:
                new_due = max(due - 1, 0)
                j = state_index[(v, new_due)]
            P[i, j] += p_orders
            continue

        for combo in itertools.product(components, repeat=actual_orders):
            combo_prob = p_orders
            for c in components:
                combo_prob *= component_probs[c] ** combo.count(c)
            added_volume = sum(component_volumes[c] for c in combo)
            total_volume = v + added_volume  # Moved here before referencing

            for due_combo in itertools.product([2, 3, 4], repeat=actual_orders):
                prob = combo_prob
                for d in due_combo:
                    prob *= due_date_probs[d]

                new_order_due = min(due_combo)

                if due == 1:
                    if total_volume > 0:
                        j = state_index[(0, 0)]
                        P[i, j] += prob
                        continue
                    else:
                        next_due = new_order_due
                else:
                    aged_due = max(due - 1, 0)
                    next_due = min(aged_due, new_order_due)

                if total_volume > max_volume:
                    carryover = total_volume - max_volume
                    carryover = min(carryover, max_volume)
                    new_state = (carryover, next_due)
                else:
                    new_state = (total_volume, next_due)

                # Prevent invalid state: inventory due today without shipping
                if new_state[1] == 0 and new_state[0] > 0:
                    j = state_index[(0, 0)]
                    P[i, j] += prob
                    continue

                j = state_index[new_state]
                P[i, j] += prob

# --- Step 3: Save to CSV ---
df = pd.DataFrame(P, index=[str((v, d)) for v, d in state_space],
                  columns=[str((v, d)) for v, d in state_space])

df.to_csv("extended_matrix_outputs/extended_transition_matrix.csv")
print("Final matrix (fully corrected) saved as extended_transition_matrix.csv")
