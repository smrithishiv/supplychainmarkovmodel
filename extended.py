import itertools
import pandas as pd
import numpy as np

# Define component volumes and probabilities
component_volumes = [50, 100, 150]
component_probs = [0.3, 0.5, 0.2]

# Due date distribution: P(due in 2/3/4 days)
due_date_probs = [0.3, 0.4, 0.3]

# Maximum total inventory (in ft³)
max_volume = 1800  # Change this to 1000 for full model
volume_step = 50

# Generate all possible inventory states (inv_2, inv_3, inv_4)
possible_states = []
for v2 in range(0, max_volume + volume_step, volume_step):
    for v3 in range(0, max_volume - v2 + volume_step, volume_step):
        for v4 in range(0, max_volume - v2 - v3 + volume_step, volume_step):
            if v2 + v3 + v4 <= max_volume:
                possible_states.append((v2, v3, v4))

# Poisson arrival probabilities for 0–3 orders
arrival_prob = [np.exp(-2) * (2 ** k) / np.math.factorial(k) for k in range(4)]

# All combinations of 0–3 orders, each with (volume, due day prob)
order_types = list(itertools.product(component_volumes, due_date_probs))

# Generate all possible order combinations for 0–3 orders
order_combinations = []
for k in range(4):
    combos = list(itertools.product(order_types, repeat=k))
    for combo in combos:
        total_prob = arrival_prob[k]
        inv_add = [0, 0, 0]  # (inv_2, inv_3, inv_4)
        for (vol, due_prob) in combo:
            due_day = [2, 3, 4][due_date_probs.index(due_prob)]
            idx = due_day - 2
            inv_add[idx] += vol
            total_prob *= component_probs[component_volumes.index(vol)]
            total_prob *= due_prob
        if sum(inv_add) <= max_volume and total_prob > 0:
            order_combinations.append((tuple(inv_add), total_prob))

# Build Markov transition matrix
transitions = []
for from_state in possible_states:
    # Step 1: Ship products due today (inv_2 is removed), and shift due dates
    after_ship = (from_state[1], from_state[2], 0)

    for inv_add, prob in order_combinations:
        to_state = tuple(np.array(after_ship) + np.array(inv_add))
        if sum(to_state) <= max_volume:
            transitions.append({
                "From State": str(from_state),
                "To State": str(to_state),
                "Probability": round(prob, 5)
            })

# Convert to DataFrame
df = pd.DataFrame(transitions)

# Save to CSV
df.to_csv("markov_chain_due_dates.csv", index=False)
print("CSV saved as 'markov_chain_due_dates.csv'")
