import pandas as pd
import matplotlib.pyplot as plt

# Load results from Q4 with validity column
df = pd.read_csv("extended_matrix_outputs/q4_threshold_policy_costs_with_3pl.csv")

# Plot valid vs invalid
plt.figure(figsize=(10, 6))
colors = df["Valid"].astype(str).map({"True": "green", "False": "red"})


plt.bar(df["Threshold"], [1]*len(df), color=colors, width=40)
plt.yticks([])
plt.xlabel("Threshold (ft³)")
plt.title("Q4 Compliance: Valid Threshold Policies (Green = Valid, Red = Invalid)")
plt.tight_layout()
plt.savefig("graphs/q4_validity.png")
plt.show()

# Assuming you already generated steady-state distribution and state labels
import numpy as np

df_matrix = pd.read_csv("extended_matrix_outputs/extended_transition_matrix.csv", index_col=0)
P = df_matrix.values
states = [eval(idx) for idx in df_matrix.index]

# Compute steady-state
from scipy.linalg import eig
eigvals, eigvecs = eig(P.T, left=True, right=False)
ss = np.real(eigvecs[:, np.isclose(eigvals, 1)])
ss = ss[:, 0]
ss /= ss.sum()

# Aggregate volume by due days
# Reset due_volume
due_volume = {2: 0, 3: 0, 4: 0}

for (v, due), p in zip(states, ss):
    if due == 1:
        due_volume[2] += v * p  # will be due in 2 days
    elif due == 2:
        due_volume[3] += v * p  # due in 3 days
    elif due >= 3:
        due_volume[4] += v * p  # due in 4+ days

# Plot corrected expected volume
plt.figure(figsize=(8, 5))
plt.bar(due_volume.keys(), due_volume.values())
plt.xlabel("Days Until Due")
plt.ylabel("Expected Volume in Inventory (ft³)")
plt.title("Expected Inventory Volume by Due Date Remaining")
plt.xticks([2, 3, 4])
plt.grid(True)
plt.tight_layout()
plt.savefig("graphs/due_day_volume.png")
plt.show()

