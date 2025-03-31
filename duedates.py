import pandas as pd
import numpy as np
import itertools

# --- Parameters ---
components = {'A': 50, 'B': 100, 'C': 150}
comp_probs = {'A': 0.3, 'B': 0.5, 'C': 0.2}
due_date_probs = {2: 0.5, 3: 0.5}  # we'll simplify to only use 2-day due dates
arrival_probs = {0: 0.1353, 1: 0.2707, 2: 0.2707, 3: 0.1805}  # truncated Poisson(2)

MAX_VOLUME = 900  # Showcase limit
VOLUMES = list(range(0, MAX_VOLUME + 50, 50))
STATE_SPACE = [(inv, d2, d3) for inv in VOLUMES for d2 in VOLUMES for d3 in VOLUMES if inv == d2 + d3]

# --- Generate All Demand Scenarios (0–3 orders) ---
order_combos = []
for n in range(4):  # 0 to 3 orders
    for orders in itertools.product(components.items(), repeat=n):
        prob = arrival_probs[n]
        vol_due_pairs = []
        for (comp, vol) in orders:
            vol_due_pairs.append((vol, 2))  # simplified: all orders go to D3
            prob *= comp_probs[comp] * due_date_probs[2]
        order_combos.append((vol_due_pairs, prob))

# --- Transition Function ---
def get_next_state(state, demand):
    inv, d2, d3 = state
    shipped = d2
    inv = max(inv - shipped, 0)
    d2 = d3
    d3 = 0
    for vol, due in demand:
        if due == 2:
            d3 += vol
        elif due == 1:
            d2 += vol
        inv += vol
    if shipped > MAX_VOLUME:
        return None
    if inv > MAX_VOLUME:
        inv = MAX_VOLUME
    return (inv, d2, d3)

# --- Build Transition Matrix ---
STATE_LABELS = [str(s) for s in STATE_SPACE]
transition_matrix = pd.DataFrame(0.0, index=STATE_LABELS, columns=STATE_LABELS)

for from_state in STATE_SPACE:
    for demand, prob in order_combos:
        to_state = get_next_state(from_state, demand)
        if to_state in STATE_SPACE:
            from_label = str(from_state)
            to_label = str(to_state)
            transition_matrix.loc[from_label, to_label] += prob

# Normalize rows to ensure valid transition probabilities
transition_matrix = transition_matrix.div(transition_matrix.sum(axis=1), axis=0).fillna(0)

# Save to CSV
transition_matrix.to_csv("outputs/markov_transition_matrix_showcase.csv")
print("Matrix saved as 'markov_transition_matrix_showcase.csv'")
