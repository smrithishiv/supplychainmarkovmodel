import numpy as np
import pandas as pd

# === STEP 1: Load base transition matrix ===
df = pd.read_csv("labelledmatrix.csv", header=0, index_col=0)
P_base = df.to_numpy()
volumes = df.columns.astype(int).tolist()  # Extract volume labels from the columns

# === STEP 2: Define Due Date Probabilities ===
# Each order has a due date of 2, 3, or 4 days
due_date_probs = [0.3, 0.4, 0.3]
due_dates = [2, 3, 4]  # Possible due date states

# === STEP 3: Extend State Space ===
# New state representation: (Volume, Days Until Due)
state_labels = []
state_index = {}

for i, vol in enumerate(volumes):
    for days in due_dates:
        state_labels.append((vol, days))
        state_index[(vol, days)] = len(state_labels) - 1

# Number of new states
num_states = len(state_labels)

# === STEP 4: Construct Extended Transition Matrix ===
P_extended = np.zeros((num_states, num_states))

for (vol, days), idx in state_index.items():
    base_idx = volumes.index(vol)  # Index in the base transition matrix
    
    for j, next_vol in enumerate(volumes):
        prob = P_base[base_idx, j]  # Transition probability from base matrix

        if days == 2:  # If the due date is 2 days away, the order must ship tomorrow
            P_extended[idx, state_index.get((next_vol, 1), idx)] += prob
        elif days == 1:  # Must ship today (absorbing state)
            P_extended[idx, idx] = 1.0
        else:  # Otherwise, countdown
            P_extended[idx, state_index.get((next_vol, days - 1), idx)] += prob

# === STEP 5: Analyze the Extended Model ===
# Convert to DataFrame for better visualization
df_extended = pd.DataFrame(P_extended, index=state_labels, columns=state_labels)
print("Extended Transition Matrix with Due Date States:")
print(df_extended)

pd.DataFrame(df_extended).to_csv("extended_transition_matrix.csv", index=True)  # Save to CSV file

print("Extended transition matrix constructed and saved.")